# video-fx-tools

Small standalone command-line tools for video editors and motion designers: real datamoshing, measuring
eases from reference clips, colour-grade matching, beat maps, voice reverb transitions, white-background
keying, and macOS Vision tools for mattes, face tracks and mouth tracking.

## Setup

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

**ffmpeg:** tools that call ffmpeg use the `FFMPEG` environment variable if it is set, otherwise the first
`ffmpeg` on your `PATH`:

```sh
export FFMPEG=/opt/homebrew/bin/ffmpeg   # optional
```

**Swift tools (macOS only, need Xcode command line tools):**

```sh
cd swift
swiftc -O matte.swift -o matte
swiftc -O matte_subject.swift -o matte_subject
swiftc -O facetrack.swift -o facetrack
swiftc -O mouthtrack.swift -o mouthtrack
```

## Tools

### datamosh.py: real datamosh (I-frame removal + P-frame bloom)
Encodes A+B to MPEG-4 ASP with one I-frame at the cut, drops it, optionally repeats P-frames ("bloom"),
then decodes the broken stream to ProRes (`.mov`) or H.264 (`.mp4`). Needs ffmpeg only.
```sh
python3 datamosh.py --a clipA.mov --b clipB.mov --out mosh.mov --fps 30 --w 1080 --h 1920
python3 datamosh.py --a clipA.mov --b clipB.mov --out mosh.mov --q 3 --bloom 45:12,80:6
```
`--bloom F:N` repeats the P-frame at frame F (in the joined clip) N extra times. `--keep-cut-i` = no mosh (debug).

### curve_fit.py: measure a move and get the After Effects ease
Tracks a move in a reference clip and returns the value/speed graphs, the fitted 2-key AE temporal ease,
the cubic-bezier, and the closest named curve family (same families as `graph_lib.jsx` in ae-motion-toolkit).
```sh
python3 curve_fit.py scan  ref.mp4 --t0 0 --t1 5 --out moves          # global motion per frame -> csv + png
python3 curve_fit.py fit   ref.mp4 --t0 1.2 --t1 1.8 --roi 0.2,0.3,0.4,0.2 --prop scale --out pop
python3 curve_fit.py series values.csv --fps 20 --out mymove         # fit values you measured another way
python3 curve_fit.py curve zine_whip_exit --frames 12                     # print AE settings of a named family
```

### grade_match.py: measure a grade and match a render to a reference
Measures luma percentiles, channel balance, saturation and band casts, then prints per-channel Levels +
saturation fixes and a predicted residual.
```sh
python3 grade_match.py ref.mp4 my_render.mp4 --start 2 --dur 10 --ref-start 0 --ref-dur 20
python3 grade_match.py ref.mp4 frames_dir/          # a folder of png/jpg/tif frames works too
python3 grade_match.py --measure some.mp4           # just print the metrics as JSON
```
Options: `--fps 2 --max 60`, `--all-frames` (keep text cards/UI), `--remeasure`, `--json out.json`.
Optional reference library: point `GRADES_JSON` at a `grades.json` (`{"videos": {"id": {"path": ..., "metrics": ...}}}`)
and refer to references by id; `path` entries resolve relative to `GRADES_ROOT` (default: the json's folder).

### beat_map.py: tempo, phase and per-bar energy
```sh
python3 beat_map.py track.wav      # writes track.beat.json {bpm, beat, phase, sr, dur, hits, fit, bars, low}
python3 beat_map.py track.wav --grid 8th                        # + "grid": [{t, bar, beat, sub, kind}, ...]
python3 beat_map.py track.wav --grid 16th --csv cuts.csv --fps 25 # + marker list with HH:MM:SS:FF timecodes
```
`--grid beat|8th|16th` adds one entry per grid line from the first detected beat (bar 1 beat 1) to the end;
`kind` is `downbeat`, `beat`, `8th` or `16th`. `--csv` writes the same grid (beats if `--grid` is not given) as
`Marker Name, Description, In, Out, Duration, Marker Type` rows (the layout of Premiere's marker list), named
`Bar 3` / `3.2` / `3.2.3`. Timecode is non-drop at the rounded `--fps` (default 30), counted from real time.

### ramp_profile.py: per-frame speed ramp for Time Remap
Turns a ramp (`apex`, `beat` or `step`) into source time and speed for every comp frame, and says which frame
blending the slowest part needs. No dependencies. It never reads past the end of the source: the fast parts are
slowed first, and `short_source` is reported when even the slow part does not fit.
```sh
python3 ramp_profile.py '{"family":"beat","comp_in":0,"comp_out":2,"src_in":0,"src_avail":6,"src_fps":50}' 24
python3 ramp_profile.py shot.json 30 --csv ramp.csv --json ramp.json
```

### voice_fx.py: voice reverb transitions for offline mixes
`reverb_tail()` ("echo out" on the last word before a transition) and `reverse_swell()` ("pre-verb" into the
first word after one). Both return a wet-only signal to add to the dry voice. Import it, or run the demo:
```sh
python3 voice_fx.py voice.wav 3.20 3.65 tail.wav swell.wav     # word from 3.20 s to 3.65 s
```
```python
from voice_fx import reverb_tail
wet, at = reverb_tail(voice, sr, word_t0, word_t1, decay=1.6, level_db=-8)
mix[at:at + len(wet)] += wet
```

### keyalpha.py: white-background removal that keeps inner whites
Removes only white areas connected to the image border (keeps eyes/highlights inside drawings), feathers
and de-fringes the edge.
```sh
python3 keyalpha.py in.png out.png                 # ffmpeg from FFMPEG / PATH
python3 keyalpha.py /path/to/ffmpeg in.png out.png # explicit ffmpeg (original form)
```

### swift/matte: person matte + face track (macOS Vision)
```sh
swift/matte in.mp4 <start_s> <dur_s> out_matte.mp4 out_face.json [preview.jpg]
```
1080x1920 30 fps grayscale person-segmentation matte + face json (`t,cx,cy,w,h,nx,ny,yaw`, 2160x3840 coords).
Note: it rotates the source 90 deg CCW first (for phone clips stored sideways).

### swift/matte_subject: high-quality subject matte for roto pop-outs (macOS Vision)
```sh
swift/matte_subject in.mov out_matte.mp4 [cx cy] [--keep N] [--track face.json] [--reach PX] [--nocarry] [--debug dir]
```
Foreground-instance mask + person segmentation, tracked by overlap, temporally smoothed. `cx cy` = subject
anchor at frame 0 (source pixels), `--keep 2` for duets.

### swift/facetrack: largest-face track
```sh
swift/facetrack in.mov out.json     # [{t,cx,cy,w,h} | null] every 2nd frame, source pixels, y from top
```

### swift/mouthtrack: mouth openness per frame (for A/V sync checks)
```sh
swift/mouthtrack in.mov out.json    # [{t,open}], open = inner-lip height (normalised), -1 = no face
```
Cross-correlate `open` against the audio envelope to measure an audio/video offset.

## Tests

Smoke tests use synthetic inputs only (no media files needed):

```sh
pip install pytest
FFMPEG=/path/to/ffmpeg python -m pytest -q tests   # ffmpeg tests are skipped if no ffmpeg is found
```

## License

MIT, see [LICENSE](LICENSE).
