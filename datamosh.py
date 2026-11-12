#!/usr/bin/env python3
"""Real datamosh (I-frame removal + P-frame bloom) with plain ffmpeg, no Avidemux.

Based on a classic Avidemux datamosh tutorial.
Same idea as the tutorial (Avidemux 2.5 + Xvid, I-frame interval huge, B-frames 0, delete the I-frame at the cut,
copy/paste P-frames to extend motion), done on a raw MPEG-4 Part 2 elementary stream:
  1. ffmpeg: concat A + B -> mpeg4 (Xvid-like ASP), GOP = whole clip, 0 B-frames, scene-cut I-frames off, one forced I-frame exactly at the cut, raw .m4v
  2. parse VOPs (start code 00 00 01 B6, 2 bits coding type: 0=I 1=P 2=B), drop every I-VOP after the first,
     optionally repeat the P-VOP at given frames N times ("bloom": pixels keep sliding with that motion)
  3. ffmpeg decodes the broken stream (errors ignored) -> ProRes 422 HQ .mov (or mp4) for AE/Premiere

Usage (ffmpeg on PATH, or set FFMPEG=/path/to/ffmpeg):
  python3 datamosh.py --a clipA.mov --b clipB.mov --out mosh.mov [--fps 30] [--w 1080 --h 1920]
        [--q 3] [--bloom 45:12,80:6] [--keep-cut-i]
  --bloom F:N  repeat the P-frame at output frame F (counted in the concatenated clip) N extra times.
  Tips (tutorial): never delete the first I-frame; mosh works best when B has strong motion; re-encode the result.
"""
import argparse, os, shutil, subprocess, sys, tempfile

FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"   # env FFMPEG overrides PATH lookup


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:])
        raise SystemExit("ffmpeg failed")


def probe_duration(path):
    r = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "Duration:" in line:
            h, m, sec = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(sec)
    raise SystemExit("no duration for " + path)


def split_vops(data):
    """Return (header_bytes, [vop_bytes...]) for an MPEG-4 Part 2 elementary stream."""
    marker = b"\x00\x00\x01\xb6"
    idx = []
    i = data.find(marker)
    while i != -1:
        idx.append(i)
        i = data.find(marker, i + 4)
    if not idx:
        raise SystemExit("no VOPs found (not an mpeg4 ASP stream?)")
    header = data[:idx[0]]
    vops = [data[s:e] for s, e in zip(idx, idx[1:] + [len(data)])]
    return header, vops


def vop_type(v):
    return (v[4] >> 6) & 3  # 0 I, 1 P, 2 B, 3 S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--w", type=int, default=1080)
    ap.add_argument("--h", type=int, default=1920)
    ap.add_argument("--q", type=int, default=3, help="mpeg4 qscale 1..31 (higher = blockier mosh)")
    ap.add_argument("--bloom", default="", help="F:N[,F:N] repeat P-frame at frame F N times")
    ap.add_argument("--keep-cut-i", action="store_true", help="keep I-frames (debug: no mosh)")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="mosh_")
    durA = probe_duration(a.a)
    raw = os.path.join(tmp, "in.m4v")
    vf = f"scale={a.w}:{a.h}:force_original_aspect_ratio=increase,crop={a.w}:{a.h},fps={a.fps},setsar=1"
    run([FFMPEG, "-v", "error", "-y", "-i", a.a, "-i", a.b, "-filter_complex",
         f"[0:v]{vf}[x];[1:v]{vf}[y];[x][y]concat=n=2:v=1:a=0[v]", "-map", "[v]",
         "-c:v", "mpeg4", "-vtag", "xvid", "-q:v", str(a.q), "-g", "100000", "-bf", "0",
         "-sc_threshold", "1000000000", "-force_key_frames", f"{durA:.4f}", "-mpv_flags", "+strict_gop", "-f", "m4v", raw])

    header, vops = split_vops(open(raw, "rb").read())
    types = [vop_type(v) for v in vops]
    blooms = {}
    for tok in filter(None, a.bloom.split(",")):
        f, n = tok.split(":")
        blooms[int(f)] = int(n)

    out_vops, dropped = [], 0
    for k, v in enumerate(vops):
        if types[k] == 0 and k > 0 and not a.keep_cut_i:
            dropped += 1
            continue
        out_vops.append(v)
        if k in blooms and types[k] == 1:
            out_vops.extend([v] * blooms[k])
    moshed = os.path.join(tmp, "mosh.m4v")
    with open(moshed, "wb") as fh:
        fh.write(header)
        for v in out_vops:
            fh.write(v)
    print(f"VOPs {len(vops)} (I={types.count(0)} P={types.count(1)} B={types.count(2)}), dropped I={dropped}, "
          f"out frames {len(out_vops)}")

    enc = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"] if a.out.lower().endswith(".mov") \
        else ["-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p"]
    run([FFMPEG, "-v", "error", "-y", "-err_detect", "ignore_err", "-fflags", "+genpts", "-r", str(a.fps),
         "-i", moshed, *enc, "-r", str(a.fps), a.out])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
