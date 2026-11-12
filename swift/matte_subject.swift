import AVFoundation; import Vision; import CoreImage; import Foundation
// matte_subject: high-quality single-subject (or N-subject) alpha matte for AE roto pop-outs.
// usage: matte_subject in.mov out_matte.mp4 [cx cy] [--keep N] [--track face.json] [--reach PX] [--debug dir]
//   cx,cy  : subject anchor at frame 0 in SOURCE pixel coords (2160x3840, y from top)
//   --keep : number of subject instances to keep (2 for duets)            default 1
//   --track: per-frame face json [{t,cx,cy},...] (source coords); used only to re-acquire when lost
//   --nocarry: disable person-seg carry when the instance shrinks (use when a foreground person crosses the subject)
//   --reach: how far (px @1080 wide) person-segmentation may extend beyond the instance mask   default 16
// Method per frame (1080x1920):
//   1. VNGenerateForegroundInstanceMaskRequest -> instances; pick the ones that overlap last frame's matte
//      (frame 0 / lost: instance containing / nearest the anchor). Soft mask via generateScaledMaskForImage.
//   2. VNGeneratePersonSegmentationRequest(.accurate); keep only person pixels connected to the chosen
//      instances, inside a dilated region around them, and not belonging to other instances.
//      final = max(instance, person * keep)   -> recovers hair tips / hands / arms.
//   3. motion-adaptive EMA (0.6 cur / 0.4 prev on stable pixels, none on large changes), 2 px dilate, 1.5 px feather.
// Output: 1080x1920 grayscale (as RGB) H.264 40 Mbps, every frame, PTS = source PTS - first PTS.
let A=CommandLine.arguments
var pos:[String]=[]; var keepN=1; var trackPath:String?=nil; var reach=16; var debugDir:String?=nil; var noCarry=false
var i=1
while i<A.count { let s=A[i]
  if s=="--keep" { keepN=Int(A[i+1])!; i+=2 } else if s=="--track" { trackPath=A[i+1]; i+=2 }
  else if s=="--reach" { reach=Int(A[i+1])!; i+=2 } else if s=="--debug" { debugDir=A[i+1]; i+=2 } else if s=="--nocarry" { noCarry=true; i+=1 }
  else { pos.append(s); i+=1 } }
guard pos.count>=2 else { print("usage: matte_subject in.mov out_matte.mp4 [cx cy] [--keep N] [--track f.json] [--reach px]"); exit(1) }
let W=1080,H=1920,N=W*H
let QW=270,QH=480,QN=QW*QH   // quarter grid for region logic
var anchor0:(Double,Double)?=nil
if pos.count>=4, let x=Double(pos[2]), let y=Double(pos[3]) { anchor0=(x/2,y/2) }
// ---- face track
var track:[(Double,Double,Double)]=[]
if let tp=trackPath, let d=try? Data(contentsOf:URL(fileURLWithPath:tp)), let arr=try? JSONSerialization.jsonObject(with:d) as? [Any] {
  for e in arr { if let o=e as? [String:Any], let t=o["t"] as? Double, let x=o["cx"] as? Double, let y=o["cy"] as? Double { track.append((t,x/2,y/2)) } } }
func facePoint(_ t:Double)->(Double,Double)? {
  var best:(Double,Double)?=nil; var bd=0.06
  for (tt,x,y) in track { let d=abs(tt-t); if d<bd { bd=d; best=(x,y) } }
  return best }
// ---- IO
let asset=AVURLAsset(url:URL(fileURLWithPath:pos[0]))
let vtrack=asset.tracks(withMediaType:.video)[0]
let fps=vtrack.nominalFrameRate
// media-time ranges actually used by the edit list (ProRes exports often carry a 1-frame tail outside the edit)
let mediaRanges=vtrack.segments.filter{ !$0.isEmpty }.map{ $0.timeMapping.source }
func inEdit(_ t:CMTime)->Bool { if mediaRanges.isEmpty { return true }
  let half=CMTime(value:1,timescale:CMTimeScale(max(1,Int32((fps*2).rounded()))))
  return mediaRanges.contains{ CMTimeCompare(CMTimeAdd(t,half),$0.start)>=0 && CMTimeCompare(CMTimeAdd(t,half),$0.end)<=0 } }
var skipped=0
let reader=try! AVAssetReader(asset:asset)
let rout=AVAssetReaderTrackOutput(track:vtrack,outputSettings:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA])
rout.alwaysCopiesSampleData=false
reader.add(rout); reader.startReading()
let wurl=URL(fileURLWithPath:pos[1]); try? FileManager.default.removeItem(at:wurl)
let writer=try! AVAssetWriter(outputURL:wurl,fileType:.mp4)
let win=AVAssetWriterInput(mediaType:.video,outputSettings:[AVVideoCodecKey:AVVideoCodecType.h264,AVVideoWidthKey:W,AVVideoHeightKey:H,
  AVVideoCompressionPropertiesKey:[AVVideoAverageBitRateKey:40_000_000,AVVideoProfileLevelKey:AVVideoProfileLevelH264HighAutoLevel,
  AVVideoExpectedSourceFrameRateKey:Int(fps.rounded()),AVVideoMaxKeyFrameIntervalKey:Int(fps.rounded())]])
win.expectsMediaDataInRealTime=false
let ad=AVAssetWriterInputPixelBufferAdaptor(assetWriterInput:win,sourcePixelBufferAttributes:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA,kCVPixelBufferWidthKey as String:W,kCVPixelBufferHeightKey as String:H])
writer.add(win); writer.startWriting(); writer.startSession(atSourceTime:.zero)
let ctx=CIContext(options:[.workingColorSpace:NSNull(),.outputColorSpace:NSNull()])
func newBuf()->CVPixelBuffer { var p:CVPixelBuffer?; CVPixelBufferCreate(nil,W,H,kCVPixelFormatType_32BGRA,[kCVPixelBufferIOSurfacePropertiesKey as String:[:]] as CFDictionary,&p); return p! }
// ---- helpers
func sample8(_ pb:CVPixelBuffer)->(Int,Int,[Float]) {   // read OneComponent8 or Float32 buffer to float array
  CVPixelBufferLockBaseAddress(pb,.readOnly); defer{CVPixelBufferUnlockBaseAddress(pb,.readOnly)}
  let w=CVPixelBufferGetWidth(pb),h=CVPixelBufferGetHeight(pb),bpr=CVPixelBufferGetBytesPerRow(pb)
  let base=CVPixelBufferGetBaseAddress(pb)!; let fmt=CVPixelBufferGetPixelFormatType(pb)
  var o=[Float](repeating:0,count:w*h)
  if fmt==kCVPixelFormatType_OneComponent32Float {
    for y in 0..<h { let r=(base+y*bpr).assumingMemoryBound(to:Float.self); for x in 0..<w { o[y*w+x]=r[x] } }
  } else { for y in 0..<h { let r=(base+y*bpr).assumingMemoryBound(to:UInt8.self); for x in 0..<w { o[y*w+x]=Float(r[x])/255 } } }
  return (w,h,o) }
func resize(_ s:[Float],_ sw:Int,_ sh:Int,_ dw:Int,_ dh:Int)->[Float] {   // bilinear
  if sw==dw && sh==dh { return s }
  var o=[Float](repeating:0,count:dw*dh)
  let fx=Float(sw)/Float(dw), fy=Float(sh)/Float(dh)
  for y in 0..<dh { let sy=max(0,min(Float(sh-1),(Float(y)+0.5)*fy-0.5)); let y0=Int(sy); let y1=min(sh-1,y0+1); let wy=sy-Float(y0)
    for x in 0..<dw { let sx=max(0,min(Float(sw-1),(Float(x)+0.5)*fx-0.5)); let x0=Int(sx); let x1=min(sw-1,x0+1); let wx=sx-Float(x0)
      let a=s[y0*sw+x0]*(1-wx)+s[y0*sw+x1]*wx, b=s[y1*sw+x0]*(1-wx)+s[y1*sw+x1]*wx
      o[y*dw+x]=a*(1-wy)+b*wy } }
  return o }
func maxFilter(_ s:[Float],_ w:Int,_ h:Int,_ r:Int)->[Float] {   // separable square max (van Herk-ish naive, r small or grid small)
  if r<=0 { return s }
  var t=[Float](repeating:0,count:w*h), o=[Float](repeating:0,count:w*h)
  for y in 0..<h { for x in 0..<w { var m:Float=0; for k in max(0,x-r)...min(w-1,x+r) { let v=s[y*w+k]; if v>m {m=v} }; t[y*w+x]=m } }
  for y in 0..<h { for x in 0..<w { var m:Float=0; for k in max(0,y-r)...min(h-1,y+r) { let v=t[k*w+x]; if v>m {m=v} }; o[y*w+x]=m } }
  return o }
func dilateDisk(_ s:[Float],_ w:Int,_ h:Int,_ r:Int)->[Float] {   // roughly round: square r*0.7 then plus-shapes
  var o=s; var left=r; while left>0 { let st=min(left,3); o=maxFilter(o,w,h,st); left-=st }; return o }
func gauss(_ s:[Float],_ w:Int,_ h:Int,_ sigma:Float)->[Float] {
  let r=Int(ceil(sigma*3)); var k=[Float](); var sum:Float=0
  for j in -r...r { let v=exp(-Float(j*j)/(2*sigma*sigma)); k.append(v); sum+=v }; k=k.map{$0/sum}
  var t=[Float](repeating:0,count:w*h), o=[Float](repeating:0,count:w*h)
  for y in 0..<h { for x in 0..<w { var a:Float=0; for j in -r...r { a+=s[y*w+min(w-1,max(0,x+j))]*k[j+r] }; t[y*w+x]=a } }
  for y in 0..<h { for x in 0..<w { var a:Float=0; for j in -r...r { a+=t[min(h-1,max(0,y+j))*w+x]*k[j+r] }; o[y*w+x]=a } }
  return o }
func pool(_ s:[Float],_ w:Int,_ h:Int,_ dw:Int,_ dh:Int,_ useMax:Bool)->[Float] {
  var o=[Float](repeating:0,count:dw*dh); let fx=w/dw, fy=h/dh
  for y in 0..<dh { for x in 0..<dw { var a:Float=0
    for yy in 0..<fy { for xx in 0..<fx { let v=s[(y*fy+yy)*w+x*fx+xx]; if useMax { if v>a {a=v} } else { a+=v } } }
    o[y*dw+x]=useMax ? a : a/Float(fx*fy) } }
  return o }
func savePGM(_ s:[Float],_ w:Int,_ h:Int,_ path:String) {
  var d=Data("P5 \(w) \(h) 255\n".utf8); d.append(contentsOf:s.map{UInt8(max(0,min(255,$0*255)))}); try? d.write(to:URL(fileURLWithPath:path)) }

let fgReq=VNGenerateForegroundInstanceMaskRequest()
let segReq=VNGeneratePersonSegmentationRequest(); segReq.qualityLevel = .accurate; segReq.outputPixelFormat=kCVPixelFormatType_OneComponent8

var prevFinal:[Float]?=nil          // full res, after EMA
var prevQ:[Float]?=nil              // quarter grid binary of previous subject
var lastCentroid:(Double,Double)?=anchor0
var runArea:Double?=nil
var areaShort=false
var areaJump=false
var gatedFrames=0
var runQ:Double?=nil
var rejQ=[Float](repeating:0,count:QN)
var n=0; var firstPTS:CMTime?=nil; var lostFrames=0
var log:[String]=[]
while let sb=rout.copyNextSampleBuffer() { autoreleasepool {
  guard let pb=CMSampleBufferGetImageBuffer(sb) else {return}
  let pts=CMSampleBufferGetPresentationTimeStamp(sb)
  if !inEdit(pts) { skipped+=1; return }
  if firstPTS==nil { firstPTS=pts }
  let tsec=CMTimeGetSeconds(CMTimeSubtract(pts,firstPTS!))
  var img=CIImage(cvPixelBuffer:pb)
  img=img.transformed(by:CGAffineTransform(scaleX:Double(W)/img.extent.width,y:Double(H)/img.extent.height))
  let rpb=newBuf(); ctx.render(img,to:rpb)
  let handler=VNImageRequestHandler(cvPixelBuffer:rpb,options:[:])
  try? handler.perform([fgReq,segReq])
  // person seg -> full res
  var person=[Float](repeating:0,count:N)
  if let m=segReq.results?.first?.pixelBuffer { let (sw,sh,s)=sample8(m); person=resize(s,sw,sh,W,H) }
  // instances
  var inst=[Float](repeating:0,count:N); var others=[Float](repeating:0,count:N)
  var chosen=IndexSet()
  if let obs=fgReq.results?.first {
    let (lw,lh,lab)=sample8(obs.instanceMask)   // labels/255
    var cnt=[Int:Int](), sx=[Int:Double](), sy=[Int:Double](), ovl=[Int:Int](), hitA=[Int:Bool]()
    var minD=[Int:Double]()
    if let p=prevQ, !p.contains(where:{$0>0}) { prevQ=nil }
    let anchor:(Double,Double)? = (prevQ==nil) ? (lastCentroid ?? facePoint(tsec)) : lastCentroid
    for y in 0..<lh { for x in 0..<lw { let l=Int((lab[y*lw+x]*255).rounded()); if l==0 {continue}
      cnt[l,default:0]+=1; let px=(Double(x)+0.5)*Double(W)/Double(lw), py=(Double(y)+0.5)*Double(H)/Double(lh)
      sx[l,default:0]+=px; sy[l,default:0]+=py
      if let p=prevQ { let qx=min(QW-1,Int(px/4)), qy=min(QH-1,Int(py/4)); if p[qy*QW+qx]>0.5 { ovl[l,default:0]+=1 } }
      if let a=anchor { let d=hypot(px-a.0,py-a.1); if d<minD[l] ?? 1e9 { minD[l]=d } } } }
    _ = hitA
    let labels=Array(cnt.keys)
    areaShort=false; areaJump=false
    if prevQ != nil {
      // keep instances that mostly lie on last frame's subject
      var sel=labels.filter{ Double(ovl[$0] ?? 0)/Double(cnt[$0]!) > 0.45 }
      if sel.isEmpty { sel=labels.filter{ Double(ovl[$0] ?? 0)/Double(cnt[$0]!) > 0.2 }.sorted{ (ovl[$0] ?? 0) > (ovl[$1] ?? 0) }.prefix(keepN).map{$0} }
      if sel.isEmpty, anchor != nil { // re-acquire near the last known subject position
        sel=labels.filter{ (minD[$0] ?? 1e9) < 60 }.sorted{ (minD[$0] ?? 1e9) < (minD[$1] ?? 1e9) }.prefix(keepN).map{$0} }
      // area guard: a subject that vanished (merged/missed for some frames) comes back as an unselected instance
      var selArea=Double(sel.reduce(0){$0+cnt[$1]!})
      if keepN>1, sel.count<keepN, let ra=runArea, selArea<0.75*ra, let lc=lastCentroid {
        let pxs=Double(W*H)/Double(lw*lh)
        let rest=labels.filter{ !sel.contains($0) }.map{ l->(Int,Double) in (l,hypot(sx[l]!/Double(cnt[l]!)-lc.0, sy[l]!/Double(cnt[l]!)-lc.1)) }
          .filter{ $0.1 < 1.2*sqrt(ra*pxs) && Double(cnt[$0.0]!) < 1.3*(ra-selArea)+0.1*ra }.sorted{ $0.1<$1.1 }
        for (l,_) in rest { if selArea>=0.85*ra { break }; sel.append(l); selArea+=Double(cnt[l]!) } }
      areaJump = runArea != nil && selArea>1.25*runArea!
      if let ra=runArea { runArea = selArea>=0.85*ra ? 0.8*ra+0.2*selArea : 0.99*ra } else if selArea>0 { runArea=selArea }
      chosen=IndexSet(sel)
    } else if !labels.isEmpty {
      var a=anchor
      if a==nil { // largest * central
        let best=labels.max{ l1,l2 in
          func sc(_ l:Int)->Double { let cx=sx[l]!/Double(cnt[l]!), cy=sy[l]!/Double(cnt[l]!); return Double(cnt[l]!)/(1+hypot(cx-540,cy-960)/400) }
          return sc(l1)<sc(l2) }!
        a=(sx[best]!/Double(cnt[best]!), sy[best]!/Double(cnt[best]!))
        for l in labels { let cx=sx[l]!/Double(cnt[l]!), cy=sy[l]!/Double(cnt[l]!); minD[l]=hypot(cx-a!.0,cy-a!.1) }
      }
      let order=labels.sorted{ (minD[$0] ?? 1e9) < (minD[$1] ?? 1e9) }
      chosen=IndexSet(order.prefix(keepN)); runArea=Double(chosen.reduce(0){$0+cnt[$1]!})
    }
    if !chosen.isEmpty { var cx=0.0,cy=0.0,c=0.0; for l in chosen { cx+=sx[l] ?? 0; cy+=sy[l] ?? 0; c+=Double(cnt[l] ?? 0) }; if c>0 { lastCentroid=(cx/c,cy/c) } }
    if !chosen.isEmpty, let sm=try? obs.generateScaledMaskForImage(forInstances:chosen,from:handler) {
      let (mw,mh,m)=sample8(sm); inst=resize(m,mw,mh,W,H) }
    // other instances (hard, nearest)
    for y in 0..<H { let ly=min(lh-1,y*lh/H); for x in 0..<W { let l=Int((lab[ly*lw+min(lw-1,x*lw/W)]*255).rounded())
      if l != 0 && !chosen.contains(l) { others[y*W+x]=1 } } }
    log.append(String(format:"%d t=%.2f inst=%@ chosen=%@",n,tsec,labels.sorted().map{"\($0):\(cnt[$0]!)/\(ovl[$0] ?? 0)"}.joined(separator:","),chosen.map{String($0)}.joined(separator:",")))
  }
  // ---- region logic on quarter grid
  var instQ=pool(inst,W,H,QW,QH,true).map{ $0>0.5 ? Float(1):0 }
  // drop instance fragments that are disconnected from last frame's subject (background limbs merged into the instance)
  if let p=prevQ, !chosen.isEmpty {
    let pd=maxFilter(p,QW,QH,2)
    var comp=[Int](repeating:-1,count:QN); var keepC=[Bool](); var nc=0
    for s0 in 0..<QN where instQ[s0]>0 && comp[s0]<0 {
      var st=[s0]; comp[s0]=nc; var hit=false
      while let k=st.popLast() { if pd[k]>0 { hit=true }; let x=k%QW, y=k/QW
        for (dx,dy) in [(1,0),(-1,0),(0,1),(0,-1)] { let nx=x+dx, ny=y+dy; if nx<0||ny<0||nx>=QW||ny>=QH {continue}
          let j=ny*QW+nx; if instQ[j]>0 && comp[j]<0 { comp[j]=nc; st.append(j) } } }
      keepC.append(hit); nc+=1 }
    if keepC.contains(true) && keepC.contains(false) {
      var kq=[Float](repeating:0,count:QN); for k in 0..<QN where comp[k]>=0 && keepC[comp[k]] { kq[k]=1 }
      kq=maxFilter(kq,QW,QH,2); let kf=resize(kq,QW,QH,W,H)
      for k in 0..<N { inst[k]*=min(1,kf[k]*1.5) }
      for k in 0..<QN { instQ[k]*=kq[k] } } }
  for k in 0..<QN { rejQ[k]*=0.92 }
  if let p=prevQ, !chosen.isEmpty {   // keep previously rejected territory out (stops 'creeping' back in 32 px per frame)
    let rd=maxFilter(rejQ.map{ $0>0.3 ? Float(1):0 },QW,QH,1); var hit=false
    for k in 0..<QN where instQ[k]>0 && rd[k]>0 && p[k]==0 { instQ[k]=0; hit=true }
    if hit { let rf=resize(rd,QW,QH,W,H); let pf=resize(p,QW,QH,W,H); for k in 0..<N where rf[k]>0.5 && pf[k]<0.5 { inst[k]=0 } } }
  if let p=prevQ, !chosen.isEmpty {   // growth limit: the subject may only grow ~32 px (@1080) per frame beyond last frame's matte.
    // Stops background limbs / neighbours that the instance merges with from popping in; rejected territory is remembered (rejQ).
    let g=dilateDisk(p,QW,QH,8); let gf=resize(g,QW,QH,W,H)
    let soft=pool(inst,W,H,QW,QH,true); var any=false
    for k in 0..<QN where soft[k]>0.2 && g[k]==0 { rejQ[k]=1; any=true }
    for k in 0..<N { inst[k]*=min(1,gf[k]*1.5) }; for k in 0..<QN { instQ[k]*=g[k] }; if any || areaJump { gatedFrames+=1 } }
  // area check on the gated instance (quarter grid): a frame where Vision drops part of the subject -> carry it via person seg
  let effArea=Double(instQ.reduce(0,+))
  areaShort = false
  if !chosen.isEmpty { if let ra=runQ { areaShort = !noCarry && effArea<0.85*ra; runQ = effArea>=0.85*ra ? 0.8*ra+0.2*effArea : 0.99*ra } else { runQ=effArea } }
  var seedQ=instQ
  if chosen.isEmpty, let p=prevQ { seedQ=p; lostFrames+=1 }   // instance lost: carry from previous subject
  let personQ=pool(person,W,H,QW,QH,false)
  let othersQ=pool(others,W,H,QW,QH,false)
  var regionQ=dilateDisk(seedQ,QW,QH,chosen.isEmpty ? 1 : reach/4)
  if areaShort && !chosen.isEmpty, let p=prevQ {   // a subject (or part of it) missed by Vision this frame -> carry it via person seg
    let pr=dilateDisk(p,QW,QH,2)
    for k in 0..<QN { if pr[k]>0 { regionQ[k]=1 }; if p[k]>0 && personQ[k]>0.5 && othersQ[k]<0.5 { seedQ[k]=1 } } }
  var keepQ=[Float](repeating:0,count:QN); var stack=[Int]()
  for k in 0..<QN where seedQ[k]>0 { keepQ[k]=1; stack.append(k) }
  while let k=stack.popLast() { let x=k%QW, y=k/QW
    for (dx,dy) in [(1,0),(-1,0),(0,1),(0,-1)] { let nx=x+dx, ny=y+dy; if nx<0||ny<0||nx>=QW||ny>=QH {continue}
      let j=ny*QW+nx; if keepQ[j]>0 {continue}
      if regionQ[j]>0 && personQ[j]>0.4 && othersQ[j]<0.5 { keepQ[j]=1; stack.append(j) } } }
  if keepN==1 && !chosen.isEmpty {   // single subject: drop recovered pieces not connected to the instance
    var lab=[Int](repeating:-1,count:QN); var ok=[Bool](); var nc=0
    for s0 in 0..<QN where keepQ[s0]>0 && lab[s0]<0 { var st=[s0]; lab[s0]=nc; var hit=false
      while let k=st.popLast() { if instQ[k]>0 { hit=true }; let x=k%QW, y=k/QW
        for (dx,dy) in [(1,0),(-1,0),(0,1),(0,-1)] { let nx=x+dx, ny=y+dy; if nx<0||ny<0||nx>=QW||ny>=QH {continue}
          let j=ny*QW+nx; if keepQ[j]>0 && lab[j]<0 { lab[j]=nc; st.append(j) } } }
      ok.append(hit); nc+=1 }
    for k in 0..<QN where lab[k]>=0 && !ok[lab[k]] { keepQ[k]=0 } }
  let keepF=resize(maxFilter(keepQ,QW,QH,1),QW,QH,W,H)
  var cur=[Float](repeating:0,count:N)
  for k in 0..<N { let pv=min(1,max(0,(person[k]-0.35)/0.45)); let p=pv*pv*(3-2*pv)*min(1,keepF[k])*(1-others[k]); cur[k]=max(inst[k],p) }
  if chosen.isEmpty && prevQ==nil { cur=[Float](repeating:0,count:N) }
  // ---- temporal + edge
  // motion-adaptive EMA: 0.6 cur / 0.4 prev where the matte is stable (kills edge flicker),
  // fading to pure current where it changed a lot (no ghost trails on fast hands/hair)
  if let p=prevFinal { for k in 0..<N { let d=p[k]-cur[k]; let w=0.4*max(0,1-abs(d)*2.2); cur[k]+=w*d } }
  let shaped=gauss(maxFilter(cur,W,H,2),W,H,1.5)
  prevFinal=cur
  prevQ=pool(cur,W,H,QW,QH,false).map{ $0>0.5 ? Float(1):0 }
  if let dd=debugDir, n%10==0 { savePGM(inst,W,H,"\(dd)/inst_\(n).pgm"); savePGM(person,W,H,"\(dd)/person_\(n).pgm"); savePGM(keepQ,QW,QH,"\(dd)/keep_\(n).pgm") }
  // ---- write
  var opb:CVPixelBuffer?; CVPixelBufferPoolCreatePixelBuffer(nil,ad.pixelBufferPool!,&opb)
  let ob=opb!; CVPixelBufferLockBaseAddress(ob,[])
  let base=CVPixelBufferGetBaseAddress(ob)!, bpr=CVPixelBufferGetBytesPerRow(ob)
  for y in 0..<H { let r=(base+y*bpr).assumingMemoryBound(to:UInt8.self)
    for x in 0..<W { let v=UInt8(max(0,min(255,shaped[y*W+x]*255+0.5))); r[x*4]=v; r[x*4+1]=v; r[x*4+2]=v; r[x*4+3]=255 } }
  CVPixelBufferUnlockBaseAddress(ob,[])
  while !win.isReadyForMoreMediaData { usleep(2000) }
  ad.append(ob,withPresentationTime:CMTimeSubtract(pts,firstPTS!))
  n+=1
}}
if reader.status == .failed { print("reader error",reader.error as Any) }
win.markAsFinished(); let sem=DispatchSemaphore(value:0); writer.finishWriting{sem.signal()}; sem.wait()
if writer.status != .completed { print("writer error",writer.error as Any) }
if let dd=debugDir { try? log.joined(separator:"\n").write(toFile:"\(dd)/log.txt",atomically:true,encoding:.utf8) }
print("frames",n,"fps",fps,"lost",lostFrames,"skipped(outside edit)",skipped,"gated",gatedFrames)
