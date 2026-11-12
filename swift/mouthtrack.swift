// mouthtrack: mouth openness per frame (Vision face landmarks, inner lips) for lip-sync measurement.
// usage: mouthtrack <in.mov> <out.json>   -> [{t,open}] where open = inner-lip vertical extent / face box height (0 = closed)
import Foundation
import AVFoundation
import Vision
let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: mouthtrack in.mov out.json"); exit(1) }
let asset = AVURLAsset(url: URL(fileURLWithPath: args[1]))
guard let track = asset.tracks(withMediaType: .video).first else { print("no video"); exit(2) }
let reader = try! AVAssetReader(asset: asset)
let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA, kCVPixelBufferWidthKey as String: 540, kCVPixelBufferHeightKey as String: 960])
out.alwaysCopiesSampleData = false; reader.add(out); reader.startReading()
var res: [String] = []
while let sb = out.copyNextSampleBuffer() {
  guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
  let t = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sb))
  let req = VNDetectFaceLandmarksRequest()
  try? VNImageRequestHandler(cvPixelBuffer: pb, options: [:]).perform([req])
  var best: VNFaceObservation? = nil
  for f in req.results ?? [] { if best == nil || f.boundingBox.width > best!.boundingBox.width { best = f } }
  if let f = best, let lips = f.landmarks?.innerLips {
    let pts = lips.normalizedPoints; var mn = 1.0, mx = 0.0
    for p in pts { mn = min(mn, Double(p.y)); mx = max(mx, Double(p.y)) }
    res.append(String(format: "{\"t\":%.4f,\"open\":%.4f}", t, mx - mn))
  } else { res.append(String(format: "{\"t\":%.4f,\"open\":-1}", t)) }
}
try! ("[" + res.joined(separator: ",") + "]\n").write(toFile: args[2], atomically: true, encoding: .utf8)
print("\(res.count) frames")
