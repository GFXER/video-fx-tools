import AVFoundation; import Vision; import CoreImage; import Foundation
// usage: matte in.mp4 start dur out_matte.mp4 out_face.json [preview.jpg]
// Person-segmentation matte (1080x1920, 30 fps) + face track json (coords in 2160x3840, y from top).
// Note: the source frame is rotated 90 deg CCW first (.left), for phone clips stored sideways.
let a=CommandLine.arguments
let url=URL(fileURLWithPath:a[1]); let st=Double(a[2])!, du=Double(a[3])!
let asset=AVURLAsset(url:url)
let track=asset.tracks(withMediaType:.video)[0]
let reader=try! AVAssetReader(asset:asset)
reader.timeRange=CMTimeRange(start:CMTime(seconds:st,preferredTimescale:600),duration:CMTime(seconds:du,preferredTimescale:600))
let out=AVAssetReaderTrackOutput(track:track,outputSettings:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA])
reader.add(out); reader.startReading()
let W=1080,H=1920
let wurl=URL(fileURLWithPath:a[4]); try? FileManager.default.removeItem(at:wurl)
let writer=try! AVAssetWriter(outputURL:wurl,fileType:.mp4)
let win=AVAssetWriterInput(mediaType:.video,outputSettings:[AVVideoCodecKey:AVVideoCodecType.h264,AVVideoWidthKey:W,AVVideoHeightKey:H,AVVideoCompressionPropertiesKey:[AVVideoAverageBitRateKey:8_000_000]])
let ad=AVAssetWriterInputPixelBufferAdaptor(assetWriterInput:win,sourcePixelBufferAttributes:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA,kCVPixelBufferWidthKey as String:W,kCVPixelBufferHeightKey as String:H])
writer.add(win); writer.startWriting(); writer.startSession(atSourceTime:.zero)
let ctx=CIContext()
var faces:[String]=[]; var n=0; var lastT = -1.0
let seg=VNGeneratePersonSegmentationRequest(); seg.qualityLevel = .accurate; seg.outputPixelFormat=kCVPixelFormatType_OneComponent8
while let sb=out.copyNextSampleBuffer(){
  guard let pb=CMSampleBufferGetImageBuffer(sb) else {continue}
  let t=CMSampleBufferGetPresentationTimeStamp(sb).seconds
  if t - lastT < 1.0/30 - 0.004 { continue }; lastT=t
  var img=CIImage(cvPixelBuffer:pb).oriented(.left)   // rotate CCW
  img=img.transformed(by:CGAffineTransform(translationX:-img.extent.minX,y:-img.extent.minY))
  let sx=Double(W)/img.extent.width; img=img.transformed(by:CGAffineTransform(scaleX:sx,y:sx))
  var rpb:CVPixelBuffer?; CVPixelBufferCreate(nil,W,H,kCVPixelFormatType_32BGRA,[kCVPixelBufferIOSurfacePropertiesKey as String:[:]] as CFDictionary,&rpb)
  ctx.render(img,to:rpb!)
  if n==0 && a.count>6 { let c=CIImage(cvPixelBuffer:rpb!); try? ctx.writeJPEGRepresentation(of:c,to:URL(fileURLWithPath:a[6]),colorSpace:CGColorSpaceCreateDeviceRGB()) }
  let h=VNImageRequestHandler(cvPixelBuffer:rpb!,options:[:])
  let fr=VNDetectFaceLandmarksRequest()
  try? h.perform([seg,fr])
  if let m=seg.results?.first?.pixelBuffer {
    var mi=CIImage(cvPixelBuffer:m); let k=Double(W)/mi.extent.width; mi=mi.transformed(by:CGAffineTransform(scaleX:k,y:Double(H)/mi.extent.height))
    var opb:CVPixelBuffer?; CVPixelBufferPoolCreatePixelBuffer(nil,ad.pixelBufferPool!,&opb)
    ctx.render(mi.applyingFilter("CIMaskToAlpha").composited(over:CIImage(color:.black).cropped(to:CGRect(x:0,y:0,width:W,height:H))).applyingFilter("CIColorMatrix",parameters:[:]),to:opb!)
    // ensure white mask: render grayscale mask directly
    ctx.render(mi.composited(over:CIImage(color:.black).cropped(to:CGRect(x:0,y:0,width:W,height:H))),to:opb!)
    while !win.isReadyForMoreMediaData { usleep(2000) }
    ad.append(opb!,withPresentationTime:CMTime(value:Int64(n),timescale:30))
  }
  if let f=(fr.results as? [VNFaceObservation])?.max(by:{$0.boundingBox.width<$1.boundingBox.width}) {
    let b=f.boundingBox; let s=2.0 // to 2160x3840
    let cx:Double=Double(b.midX)*Double(W)*s; let cy:Double=(1.0-Double(b.midY))*Double(H)*s; let w:Double=Double(b.width)*Double(W)*s
    var nx=cx, ny=cy
    if let nose=f.landmarks?.noseCrest?.normalizedPoints.last {
      let px:Double=Double(b.minX)+Double(nose.x)*Double(b.width)
      let py:Double=Double(b.minY)+Double(nose.y)*Double(b.height)
      nx=px*Double(W)*s; ny=(1.0-py)*Double(H)*s }
    let yaw=f.yaw?.doubleValue ?? 0
    faces.append(String(format:"{\"t\":%.4f,\"cx\":%.1f,\"cy\":%.1f,\"w\":%.1f,\"h\":%.1f,\"nx\":%.1f,\"ny\":%.1f,\"yaw\":%.3f}",t,cx,cy,w,w,nx,ny,yaw))
  }
  n+=1
}
win.markAsFinished(); let sem=DispatchSemaphore(value:0); writer.finishWriting{sem.signal()}; sem.wait()
try! ("[\n"+faces.joined(separator:",\n")+"\n]").write(toFile:a[5],atomically:true,encoding:.utf8)
print("frames",n,"faces",faces.count)
