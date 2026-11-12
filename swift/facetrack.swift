// facetrack: largest-face track every 2nd frame.
// usage: facetrack <in.mov> <out.json>
// output: [{t,cx,cy,w,h} | null] in pixel coords of the source frame, y from top.
import Foundation
import AVFoundation
import Vision
import CoreImage

let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: facetrack in.mov out.json"); exit(1) }
let url = URL(fileURLWithPath: args[1])
let asset = AVURLAsset(url: url)
guard let track = asset.tracks(withMediaType: .video).first else { print("no video"); exit(2) }
let nat = track.naturalSize.applying(track.preferredTransform)
let W = Double(abs(nat.width)), H = Double(abs(nat.height))
let reader = try! AVAssetReader(asset: asset)
let dw = W >= H ? 960 : 540, dh = W >= H ? 540 : 960
let out = AVAssetReaderTrackOutput(track: track, outputSettings: [
  kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
  kCVPixelBufferWidthKey as String: dw, kCVPixelBufferHeightKey as String: dh])
out.alwaysCopiesSampleData = false
reader.add(out)
reader.startReading()
var res: [String] = []
var idx = 0
while let sb = out.copyNextSampleBuffer() {
  defer { idx += 1 }
  if idx % 2 != 0 { continue }
  guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
  let t = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sb))
  let req = VNDetectFaceRectanglesRequest()
  let h = VNImageRequestHandler(cvPixelBuffer: pb, options: [:])
  try? h.perform([req])
  var best: VNFaceObservation? = nil
  for f in req.results ?? [] {
    if best == nil || f.boundingBox.width * f.boundingBox.height > best!.boundingBox.width * best!.boundingBox.height { best = f }
  }
  if let b = best?.boundingBox {
    let cx = (b.midX) * W, cy = (1 - b.midY) * H
    res.append(String(format: "{\"t\":%.4f,\"cx\":%.1f,\"cy\":%.1f,\"w\":%.1f,\"h\":%.1f}", t, cx, cy, b.width * W, b.height * H))
  } else {
    res.append("null")
  }
}
if reader.status == .failed { print("reader failed: \(String(describing: reader.error))"); exit(3) }
try! ("[" + res.joined(separator: ",\n") + "]\n").write(toFile: args[2], atomically: true, encoding: .utf8)
let n = res.filter { $0 != "null" }.count
print("\(url.lastPathComponent) \(res.count) samples, \(n) faces")
