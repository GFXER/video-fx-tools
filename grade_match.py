#!/usr/bin/env python3
"""grade_match.py: measure a video's grade and match our render to a reference numerically.

Shared measurement core for a colour-grade library (an optional grades.json) and the
matching loop. Uses only ffmpeg + numpy/scipy (+ Pillow for strips).
ffmpeg: env FFMPEG, else the first ffmpeg on PATH. Grade library: env GRADES_JSON (optional);
its "path" entries are resolved relative to env GRADES_ROOT, else the json's folder.

USAGE
  # compare our render with a reference (id from grades.json, or a path)
  python3 grade_match.py my_reference_id renders/my_edit_v2.mp4
  python3 grade_match.py 69071 my_frames_dir/            # dir of png/jpg/tif frames
  python3 grade_match.py ref.mp4 ours.mp4 --start 2 --dur 10 --ref-start 0 --ref-dur 20
  options: --remeasure (ignore stored grades.json metrics), --all-frames (keep text cards/UI),
           --json out.json (write deltas + suggestions), --fps 2 --max 60

  # measure one file only
  python3 grade_match.py --measure some.mp4

LOOP (how to match a reference numerically)
  1 measure target  2 render frames of ours  3 run this script  4 apply the per-channel Levels
  (Individual Controls) + saturation it prints  5 re-render a few frames and run again until
  every delta is inside the tolerance column. The "predicted after" block simulates step 4 on
  our frames so you can see the residual before touching AE.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np

FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"   # env FFMPEG overrides PATH lookup
GRADES_JSON = os.environ.get("GRADES_JSON", "grades.json")   # optional reference library
WS = os.environ.get("GRADES_ROOT") or os.path.dirname(os.path.abspath(GRADES_JSON))

PCTS = [1, 5, 25, 50, 75, 95, 99]
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ----------------------------------------------------------------------------- decode
def probe(path):
    out = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
    vs = [l for l in out.splitlines() if "Video:" in l]
    w = h = 0
    fps = 0.0
    if vs:
        mm = re.search(r", (\d{2,5})x(\d{2,5})", vs[0])
        if mm:
            w, h = int(mm.group(1)), int(mm.group(2))
        mf = re.search(r"([\d.]+) fps", vs[0])
        if mf:
            fps = float(mf.group(1))
    rot = re.search(r"rotation of (-?[\d.]+)", out)
    if rot and abs(abs(float(rot.group(1))) - 90) < 1:
        w, h = h, w
    return {"w": w, "h": h, "dur": dur, "fps": fps}


def read_video(path, fps=2.0, max_frames=60, longest=480, start=None, dur=None):
    info = probe(path)
    if info["w"] == 0:
        raise RuntimeError("cannot probe " + path)
    d = dur if dur else max(0.1, info["dur"] - (start or 0))
    rate = min(fps, max_frames / max(d, 0.1))
    if info["w"] >= info["h"]:
        ow, oh = longest, int(round(longest * info["h"] / info["w"] / 2)) * 2
    else:
        oh, ow = longest, int(round(longest * info["w"] / info["h"] / 2)) * 2
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-threads", "2"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", path]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-vf", "fps=%.5f,scale=%d:%d:flags=area" % (rate, ow, oh),
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (ow * oh * 3)
    arr = np.frombuffer(raw[: n * ow * oh * 3], np.uint8).reshape(n, oh, ow, 3)
    frames = [arr[i] for i in range(min(n, max_frames))]
    times = [(start or 0) + (i + 0.5) / rate for i in range(len(frames))]
    return frames, times, info


def read_images(paths, longest=480):
    from PIL import Image
    frames = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        s = longest / max(im.size)
        if s < 1:
            im = im.resize((max(2, int(im.size[0] * s)), max(2, int(im.size[1] * s))), Image.BILINEAR)
        frames.append(np.asarray(im, np.uint8))
    return frames, list(range(len(frames))), {"w": frames[0].shape[1], "h": frames[0].shape[0], "dur": 0, "fps": 0}


def load_any(path, fps=2.0, max_frames=60, start=None, dur=None):
    if os.path.isdir(path):
        ps = sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(IMG_EXT))
        if not ps:
            raise RuntimeError("no images in " + path)
        if len(ps) > max_frames:
            idx = np.linspace(0, len(ps) - 1, max_frames).astype(int)
            ps = [ps[i] for i in idx]
        return read_images(ps)
    if path.lower().endswith(IMG_EXT):
        return read_images([path])
    return read_video(path, fps, max_frames, start=start, dur=dur)


# ----------------------------------------------------------------------------- helpers
def luma(f):
    f = f.astype(np.float32)
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def hsv_sat(f):
    f = f.astype(np.float32)
    mx = f.max(-1)
    mn = f.min(-1)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def rgb_hue(r, g, b):
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 1e-6:
        return 0.0
    d = mx - mn
    if mx == r:
        h = ((g - b) / d) % 6
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return h * 60.0


def hue_name(h):
    if h >= 345 or h < 15:
        return "red"
    if h < 40:
        return "orange"
    if h < 60:
        return "amber"
    if h < 75:
        return "yellow"
    if h < 150:
        return "green"
    if h < 185:
        return "teal"
    if h < 205:
        return "cyan"
    if h < 250:
        return "blue"
    if h < 285:
        return "violet"
    return "magenta"


WARM = ("red", "orange", "amber", "yellow")


def cast_words(rgb):
    """mean RGB of a band -> (hue deg, strength 0..1, words)."""
    # strength = chroma of the band's mean colour in 8-bit code values (max-min of mean RGB):
    # <4 neutral, 4-10 slight, 10-22 clear, >22 strong
    r, g, b = rgb
    s = max(r, g, b) - min(r, g, b)
    h = rgb_hue(r, g, b)
    n = hue_name(h)
    if s < 4:
        w = "neutral"
    elif s < 10:
        w = "slight " + n
    elif s < 22:
        w = n
    else:
        w = "strong " + n
    return round(h, 1), round(s, 1), w


def detect_active(frames):
    """crop letterbox / pillarbox bars that are black in every sampled frame."""
    if not frames:
        return (0, 0, 0, 0)
    Y = np.stack([luma(f) for f in frames])  # n,h,w
    rows = Y.mean(2).max(0)  # brightest occurrence of each row mean
    cols = Y.mean(1).max(0)
    H, W = rows.shape[0], cols.shape[0]

    def edge(v, n):
        a = 0
        while a < n // 2 and v[a] < 9:
            a += 1
        b = 0
        while b < n // 2 and v[n - 1 - b] < 9:
            b += 1
        if a < 0.03 * n:
            a = 0
        if b < 0.03 * n:
            b = 0
        if a + b > 0.8 * n:
            return 0, 0
        return a, b

    t, bo = edge(rows, H)
    l, r = edge(cols, W)
    return (t, bo, l, r)


def classify_frame(f):
    Y = luma(f)
    S = hsv_sat(f)
    white = float(((Y > 212) & (S < 0.12)).mean())
    if Y.mean() < 10 and np.percentile(Y, 99) < 40:
        return "black"
    if white > 0.55:
        return "white_ui"
    if Y.std() < 7:
        return "flat_card"
    return "picture"


# ----------------------------------------------------------------------------- measure
def _pct_from_hist(h, ps):
    c = np.cumsum(h).astype(np.float64)
    tot = c[-1]
    out = []
    for p in ps:
        t = tot * p / 100.0
        i = int(np.searchsorted(c, t))
        i = min(i, 255)
        # linear interpolation inside bin i (bin i covers [i-0.5, i+0.5])
        lo = c[i - 1] if i > 0 else 0.0
        frac = min(max((t - lo) / max(h[i], 1), 0.0), 1.0)
        out.append(round(min(255.0, max(0.0, i - 0.5 + frac)), 1))
    return out


def measure_frames(frames, rng_seed=1, palette=True):
    """pooled grade metrics over a list of uint8 RGB frames (already cropped)."""
    hY = np.zeros(256)
    hC = np.zeros((3, 256))
    n = 0
    s_rgb = np.zeros(3)
    s_rgb2 = np.zeros(3)
    s_sat = 0.0
    bands = {k: [np.zeros(3), 0, 0.0, 0.0] for k in ("shadows", "mids", "highs")}  # rgbsum, count, ysum, satsum
    skin = [0, 0.0, 0.0, 0.0, 0]  # count, hue sum, sat sum, Y sum, total
    crushed = clipped = 0
    stds = []
    samples = []
    rs = np.random.RandomState(rng_seed)
    for f in frames:
        x = f.reshape(-1, 3).astype(np.float32)
        Y = 0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2]
        mx = x.max(1)
        mn = x.min(1)
        S = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        Yi = np.clip(np.round(Y), 0, 255).astype(np.int64)
        hY += np.bincount(Yi, minlength=256)
        for c in range(3):
            hC[c] += np.bincount(x[:, c].astype(np.int64), minlength=256)
        n += x.shape[0]
        s_rgb += x.sum(0)
        s_rgb2 += (x.astype(np.float64) ** 2).sum(0)
        s_sat += float(S.sum())
        crushed += int((Y < 4).sum())
        clipped += int((Y > 250).sum())
        stds.append(float(Y.std()))
        for k, m in (("shadows", Y < 64), ("mids", (Y >= 64) & (Y < 192)), ("highs", Y >= 192)):
            if m.any():
                bands[k][0] += x[m].sum(0)
                bands[k][1] += int(m.sum())
                bands[k][2] += float(Y[m].sum())
                bands[k][3] += float(S[m].sum())
        # skin: r>g>b + YCbCr box + hue/sat/luma gates
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        cr = 128 + 0.5 * r - 0.4187 * g - 0.0813 * b
        cb = 128 - 0.1687 * r - 0.3313 * g + 0.5 * b
        d = np.maximum(mx - mn, 1e-6)
        hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
        sk = (r > g) & (g > b) & (r - b > 15) & (cr > 133) & (cr < 173) & (cb > 77) & (cb < 127) \
            & (hue < 50) & (S > 0.15) & (S < 0.7) & (Y > 40) & (Y < 235)
        if sk.any():
            skin[0] += int(sk.sum())
            skin[1] += float(hue[sk].sum())
            skin[2] += float(S[sk].sum())
            skin[3] += float(Y[sk].sum())
        skin[4] += x.shape[0]
        if palette:
            k = min(1500, x.shape[0])
            samples.append(x[rs.choice(x.shape[0], k, replace=False)])
    if n == 0:
        return None
    mean = s_rgb / n
    std = np.sqrt(np.maximum(s_rgb2 / n - mean ** 2, 0))
    yp = _pct_from_hist(hY, PCTS)
    hYc = hY.copy()
    hYc[:4] = 0  # "content" = everything except pure-black design background / bars
    ypc = _pct_from_hist(hYc, PCTS) if hYc.sum() > 0 else yp
    m = {
        "Y_pct": dict(zip(["p%d" % p for p in PCTS], yp)),
        "Y_pct_content": dict(zip(["p%d" % p for p in PCTS], ypc)),
        "Y_mean": round(float((hY * np.arange(256)).sum() / n), 1),
        "crushed_pct": round(100.0 * crushed / n, 2),
        "clipped_pct": round(100.0 * clipped / n, 2),
        "rgb_mean": [round(float(v), 1) for v in mean],
        "rgb_std": [round(float(v), 1) for v in std],
        "rgb_pct": {c: dict(zip(["p%d" % p for p in PCTS], _pct_from_hist(hC[i], PCTS))) for i, c in enumerate("RGB")},
        "sat_mean": round(s_sat / n, 3),
        "warmth_R_minus_B": round(float(mean[0] - mean[2]), 1),
        "tint_G_minus_RBavg": round(float(mean[1] - (mean[0] + mean[2]) / 2), 1),
        "contrast": {
            "mid_slope": round((yp[4] - yp[2]) / 127.5, 3),  # (p75-p25)/127.5 ; 1.0 = linear ramp
            "range_p5_p95": round(yp[5] - yp[1], 1),
            "frame_std_mean": round(float(np.mean(stds)), 1),
        },
    }
    bd = {}
    for k, (rs_, cnt, ys, ss) in bands.items():
        if cnt / n < 0.005:
            bd[k] = {"frac": round(cnt / n, 3), "rgb": None, "hue": None, "strength": None, "words": "n/a", "sat": None}
            continue
        rgb = rs_ / cnt
        h, s, w = cast_words(rgb)
        bd[k] = {"frac": round(cnt / n, 3), "rgb": [round(float(v), 1) for v in rgb], "Y": round(ys / cnt, 1),
                 "sat": round(ss / cnt, 3), "hue": h, "strength": s, "words": w}
    m["bands"] = bd
    m["split_tone"] = "%s shadows, %s mids, %s highs" % (bd["shadows"]["words"], bd["mids"]["words"], bd["highs"]["words"])
    if skin[0] / max(skin[4], 1) > 0.004:
        m["skin"] = {"frac": round(skin[0] / skin[4], 3), "hue": round(skin[1] / skin[0], 1),
                     "sat": round(skin[2] / skin[0], 3), "Y": round(skin[3] / skin[0], 1)}
    else:
        m["skin"] = None
    if palette and samples:
        m["palette"] = kmeans_palette(np.concatenate(samples))
    return m


def kmeans_palette(px, k=5):
    from scipy.cluster.vq import kmeans2
    px = px.astype(np.float64)
    if px.shape[0] > 40000:
        px = px[np.random.RandomState(2).choice(px.shape[0], 40000, replace=False)]
    try:
        cent, lab = kmeans2(px, k, minit="++", seed=3)
    except TypeError:
        np.random.seed(3)
        cent, lab = kmeans2(px, k, minit="++")
    cnt = np.bincount(lab, minlength=k)
    order = np.argsort(-cnt)
    out = []
    for i in order:
        if cnt[i] == 0:
            continue
        c = np.clip(cent[i], 0, 255)
        out.append({"hex": "#%02X%02X%02X" % tuple(int(round(v)) for v in c), "share": round(float(cnt[i] / cnt.sum()), 3)})
    return out


def analyse(path, fps=2.0, max_frames=60, start=None, dur=None, all_frames=False, palette=True):
    """decode + crop + classify + measure. returns (metrics, picture_frames, info)."""
    frames, times, info = load_any(path, fps, max_frames, start, dur)
    t, b, l, r = detect_active(frames)
    if t or b or l or r:
        H, W = frames[0].shape[:2]
        frames = [f[t:H - b, l:W - r] for f in frames]
    cls = [classify_frame(f) for f in frames]
    counts = {c: cls.count(c) for c in ("picture", "white_ui", "flat_card", "black")}
    pics = [f for f, c in zip(frames, cls) if c == "picture"]
    mostly_graphic = len(pics) < max(3, 0.25 * len(frames))
    use = frames if (all_frames or mostly_graphic) else pics
    use_times = times if (all_frames or mostly_graphic) else [tt for tt, c in zip(times, cls) if c == "picture"]
    m = measure_frames(use, palette=palette)
    nonpic = {}
    for tt, c in zip(times, cls):
        if c != "picture":
            nonpic.setdefault(c, []).append(round(tt, 2))
    m["frames"] = {"sampled": len(frames), **counts, "nonpicture_times": nonpic, "measured_on": "all" if (all_frames or mostly_graphic) else "picture",
                   "mostly_graphic": mostly_graphic, "crop_tblr": [t, b, l, r]}
    if not all_frames:
        # whole-video luma summary incl. cards, for reference
        allm = measure_frames(frames, palette=False)
        m["all_frames_Y"] = {"p5": allm["Y_pct"]["p5"], "p50": allm["Y_pct"]["p50"], "p95": allm["Y_pct"]["p95"],
                             "sat_mean": allm["sat_mean"], "crushed_pct": allm["crushed_pct"]}
    return m, use, use_times, info


# ----------------------------------------------------------------------------- look words
def look_words(m):
    y = m["Y_pct"]
    p5, p50, p95 = y["p5"], y["p50"], y["p95"]
    sat = m["sat_mean"]
    warm = m["warmth_R_minus_B"]
    sl = m["contrast"]["mid_slope"]
    w = []
    if sat < 0.08:
        w.append("monochrome")
    elif sat < 0.2:
        w.append("desaturated")
    elif sat > 0.5:
        w.append("saturated")
    if p50 > 165:
        w.append("bright/high-key")
    elif p50 < 55:
        w.append("dark/low-key")
    if p5 < 2.5 or m["crushed_pct"] > 8:
        w.append("crushed blacks")
    elif p5 > 22:
        w.append("lifted/milky blacks")
    if sl > 1.05 or m["contrast"]["frame_std_mean"] > 72:
        w.append("high contrast")
    elif sl < 0.5 and m["contrast"]["frame_std_mean"] < 45:
        w.append("soft/flat contrast")
    if sat >= 0.08:
        if warm > 18:
            w.append("warm")
        elif warm < -10:
            w.append("cool")
        else:
            w.append("neutral balance")
    sh = m["bands"]["shadows"]
    hi = m["bands"]["highs"]
    shn = (sh.get("words") or "")
    hin = (hi.get("words") or "")
    if any(c in shn for c in ("teal", "cyan", "blue")) and any(c in hin for c in WARM):
        w.append("teal-orange split")
    if p50 < 70 and sat > 0.42 and any(c in (m["bands"]["mids"]["words"] + shn) for c in ("magenta", "violet", "blue", "cyan", "teal")):
        w.append("neon night")
    if m.get("frames", {}).get("mostly_graphic"):
        w.append("graphic/UI-dominated")
    if m["clipped_pct"] > 8:
        w.append("blown highlights")
    return ", ".join(w)


# ----------------------------------------------------------------------------- matching
def _levels_apply(v, ib, iw, g, ob, ow):
    x = np.clip((v - ib) / max(iw - ib, 1e-3), 0, 1)
    return ob + (ow - ob) * x ** (1.0 / g)


def fit_levels(src, dst):
    """src/dst: dict p1..p99 for one channel (0..255). returns ib, iw, gamma, ob, ow (8-bit units)."""
    from scipy.optimize import least_squares
    keys = ["p%d" % p for p in PCTS]
    s = np.array([src[k] for k in keys], float)
    d = np.array([dst[k] for k in keys], float)
    wts = np.array([0.5, 1.5, 1.0, 1.5, 1.0, 1.5, 0.5])
    # source percentiles stuck at 0 or 255 carry no information (clipped): drop them from the fit
    ok = (s > 1.0) & (s < 254.0)
    if ok.sum() >= 3:
        s, d, wts = s[ok], d[ok], wts[ok]
    lo_i, hi_i = 0, len(s) - 1
    s5, s95, t5, t95 = s[lo_i], s[hi_i], d[lo_i], d[hi_i]
    a = (t95 - t5) / max(s95 - s5, 1.0)
    b = t5 - a * s5
    ib0 = max(0.0, -b / a)
    ob0 = max(0.0, b)
    iw0 = min(255.0, (255 - b) / a)
    ow0 = min(255.0, a * iw0 + b)

    def res(p):
        ib, iw, lg, ob, ow = p
        r = (_levels_apply(s, ib, iw, math.exp(lg), ob, ow) - d) * wts
        reg = [0.15 * ob, 0.15 * (255 - ow), 0.05 * ib, 0.05 * (255 - iw)]  # prefer plain input moves
        return np.concatenate([r, reg])

    x0 = [min(ib0, 120), max(iw0, 130), 0.0, min(ob0, 120), max(ow0, 130)]
    lo = [0, 100, math.log(0.3), 0, 100]
    hi = [120, 255, math.log(3.0), 120, 255]
    x0 = [min(max(v, l + 1e-3), h - 1e-3) for v, l, h in zip(x0, lo, hi)]
    r = least_squares(res, x0, bounds=(lo, hi))
    ib, iw, lg, ob, ow = r.x
    return {"in_black": round(ib, 1), "in_white": round(iw, 1), "gamma": round(math.exp(lg), 3),
            "out_black": round(ob, 1), "out_white": round(ow, 1)}


def simulate(frames, lv, sat_scale):
    out = []
    luts = []
    for c in "RGB":
        p = lv[c]
        luts.append(np.clip(_levels_apply(np.arange(256, dtype=float), p["in_black"], p["in_white"], p["gamma"],
                                          p["out_black"], p["out_white"]), 0, 255))
    for f in frames:
        g = np.stack([luts[i][f[..., i]] for i in range(3)], -1)
        Y = (0.2126 * g[..., 0] + 0.7152 * g[..., 1] + 0.0722 * g[..., 2])[..., None]
        g = Y + sat_scale * (g - Y)
        out.append(np.clip(np.round(g), 0, 255).astype(np.uint8))
    return out


def suggest(ref, ours, our_frames=None):
    lv = {c: fit_levels(ours["rgb_pct"][c], ref["rgb_pct"][c]) for c in "RGB"}
    sat_scale = 1.0
    pred = None
    if our_frames:
        sim = simulate(our_frames, lv, 1.0)
        m1 = measure_frames(sim, palette=False)
        sat_scale = float(np.clip(ref["sat_mean"] / max(m1["sat_mean"], 1e-3), 0.2, 3.0))
        # one refinement pass: sat scaling changes the per-channel spread a little
        sim = simulate(our_frames, lv, sat_scale)
        pred = measure_frames(sim, palette=False)
        if pred["sat_mean"] > 0:
            sat_scale *= float(np.clip(ref["sat_mean"] / pred["sat_mean"], 0.8, 1.25))
            sim = simulate(our_frames, lv, sat_scale)
            pred = measure_frames(sim, palette=False)
    return lv, sat_scale, pred


TOL = {"p5": 3, "p50": 6, "p95": 8, "sat_mean": 0.03, "crushed_pct": 3, "clipped_pct": 2, "warmth_R_minus_B": 4,
       "tint_G_minus_RBavg": 3, "mid_slope": 0.06}


def flat(m):
    y = m["Y_pct"]
    d = {k: y[k] for k in y}
    d.update({"Y_mean": m["Y_mean"], "crushed_pct": m["crushed_pct"], "clipped_pct": m["clipped_pct"],
              "sat_mean": m["sat_mean"], "warmth_R_minus_B": m["warmth_R_minus_B"],
              "tint_G_minus_RBavg": m["tint_G_minus_RBavg"], "mid_slope": m["contrast"]["mid_slope"],
              "R_mean": m["rgb_mean"][0], "G_mean": m["rgb_mean"][1], "B_mean": m["rgb_mean"][2]})
    for b in ("shadows", "mids", "highs"):
        bb = m["bands"][b]
        d[b + "_frac"] = bb["frac"]
    if m.get("skin"):
        d["skin_Y"] = m["skin"]["Y"]
        d["skin_hue"] = m["skin"]["hue"]
        d["skin_sat"] = m["skin"]["sat"]
    return d


ORDER = ["p1", "p5", "p25", "p50", "p75", "p95", "p99", "Y_mean", "crushed_pct", "clipped_pct", "mid_slope",
         "R_mean", "G_mean", "B_mean", "warmth_R_minus_B", "tint_G_minus_RBavg", "sat_mean",
         "shadows_frac", "mids_frac", "highs_frac", "skin_Y", "skin_hue", "skin_sat"]


def report(ref, ours, lv, sat_scale, pred, ref_name, our_name):
    R, O = flat(ref), flat(ours)
    P = flat(pred) if pred else {}
    L = []
    L.append("REF : %s\n      look: %s\n      split: %s" % (ref_name, look_words(ref), ref["split_tone"]))
    L.append("OURS: %s\n      look: %s\n      split: %s" % (our_name, look_words(ours), ours["split_tone"]))
    L.append("")
    L.append("%-20s %8s %8s %8s %6s   %8s" % ("metric", "ref", "ours", "delta", "ok?", "pred.after"))
    for k in ORDER:
        if k not in R or k not in O:
            continue
        dlt = O[k] - R[k]
        tol = TOL.get(k)
        ok = "" if tol is None else ("ok" if abs(dlt) <= tol else "FIX")
        three = k in ("sat_mean", "mid_slope", "skin_sat") or k.endswith("_frac")
        pa = (("%8.3f" if three else "%8.1f") % P[k]) if k in P else ""
        fmt = "%-20s %8.3f %8.3f %+8.3f %6s   %s" if three else "%-20s %8.1f %8.1f %+8.1f %6s   %s"
        L.append(fmt % (k, R[k], O[k], dlt, ok, pa))
    L.append("")
    if O["crushed_pct"] - R["crushed_pct"] > 15:
        L.append("!! %.0f %% of our pixels are crushed to Y<4 (ref %.0f %%). That detail is GONE: a Levels layer on top can only"
                 % (O["crushed_pct"], R["crushed_pct"]))
        L.append("!! lift a flat black slab (see p1..p25 in pred.after). First remove/soften whatever crushes (Levels input black,")
        L.append("!! Curves toe, Exposure offset, vignette/Multiply ramp, look strength), re-render and run this again.")
        L.append("")
    L.append("SUGGESTED FIX (apply on ONE adjustment layer above our grade, then re-measure)")
    L.append("1) AE Levels (Individual Controls), 8-bpc units (divide by 255 for 32-bpc / Pro Levels 0-1):")
    L.append("   %-4s %9s %9s %7s %9s %9s" % ("ch", "inBlack", "inWhite", "gamma", "outBlack", "outWhite"))
    for c in "RGB":
        p = lv[c]
        L.append("   %-4s %9.1f %9.1f %7.3f %9.1f %9.1f" % (c, p["in_black"], p["in_white"], p["gamma"], p["out_black"], p["out_white"]))
    L.append("   (as 0-1:  " + "  ".join("%s ib %.3f iw %.3f g %.2f ob %.3f ow %.3f" % (
        c, lv[c]["in_black"] / 255, lv[c]["in_white"] / 255, lv[c]["gamma"], lv[c]["out_black"] / 255, lv[c]["out_white"] / 255)
        for c in "RGB") + ")")
    L.append("2) Saturation x%.2f after the Levels -> Hue/Saturation Master Sat %+d, or Colorista V Saturation ~%+d "
             "(both approx. linear in chroma; verify)" % (sat_scale, round((sat_scale - 1) * 100), round((sat_scale - 1) * 100)))
    # plain-language hints
    hints = []
    dw = O["warmth_R_minus_B"] - R["warmth_R_minus_B"]
    if abs(dw) > TOL["warmth_R_minus_B"]:
        hints.append("ours is %s by %.0f (R-B); Levels above already fixes it. Without Levels: Colorista Temp %s "
                     "in small steps (start %+d) and re-measure." % ("warmer" if dw > 0 else "cooler", abs(dw),
                                                                     "down" if dw > 0 else "up", -int(np.sign(dw) * min(30, round(abs(dw))))))
    dt = O["tint_G_minus_RBavg"] - R["tint_G_minus_RBavg"]
    if abs(dt) > TOL["tint_G_minus_RBavg"]:
        hints.append("ours is %s by %.0f (G vs R/B); Colorista Tint %s (start %+d)." % (
            "greener" if dt > 0 else "more magenta", abs(dt), "toward magenta" if dt > 0 else "toward green", -int(np.sign(dt) * min(20, round(abs(dt) * 2)))))
    if O["crushed_pct"] - R["crushed_pct"] > TOL["crushed_pct"]:
        hints.append("BLACKS CRUSHED: %.1f %% < Y4 vs ref %.1f %%. Remove crushing first (lower Levels input black / "
                     "Colorista Shadow Regions +, Mojo II Fade It +) before matching anything else." % (O["crushed_pct"], R["crushed_pct"]))
    if R["crushed_pct"] - O["crushed_pct"] > TOL["crushed_pct"]:
        hints.append("ref has denser blacks (%.1f %% crushed vs %.1f %%): raise input black / Colorista Shadow Regions -." % (R["crushed_pct"], O["crushed_pct"]))
    if O["clipped_pct"] - R["clipped_pct"] > TOL["clipped_pct"]:
        hints.append("highlights clip more than ref: Colorista Highlight Rolloff 20-40 or lower input white.")
    if "skin_Y" in R and "skin_Y" in O:
        ds = O["skin_Y"] - R["skin_Y"]
        if abs(ds) > 8:
            hints.append("skin luma %+.0f vs ref: fix with a face Spot Exposure / Colorista Exposure (%.2f, sensitive) on the subject, "
                         "not the global layer." % (ds, float(np.clip(-ds / 255 * 0.6, -0.08, 0.08))))
    for b in ("shadows", "mids", "highs"):
        rw, ow_ = ref["bands"][b]["words"], ours["bands"][b]["words"]
        if rw != ow_:
            hints.append("%s cast: ref '%s' vs ours '%s' (the per-channel %s handles it)." % (
                b, rw, ow_, {"shadows": "inBlack/outBlack", "mids": "gamma", "highs": "inWhite"}[b]))
    if hints:
        L.append("3) Notes:")
        for h in hints:
            L.append("   - " + h)
    if pred:
        L.append("'pred.after' = our frames with 1)+2) applied in numpy and re-measured (expected residual).")
    return "\n".join(L)


def resolve_ref(ref, remeasure=False):
    """returns (metrics or None, path, name)."""
    if os.path.exists(ref):
        return None, ref, os.path.basename(ref)
    if os.path.exists(GRADES_JSON):
        db = json.load(open(GRADES_JSON))
        vids = db.get("videos", {})
        key = None
        for k in vids:
            if k == ref or k == "Video-" + ref or k.replace("Video-", "") == ref:
                key = k
                break
        if key:
            p = os.path.join(WS, vids[key]["path"])
            return (None if remeasure else vids[key]["metrics"]), p, key
    raise SystemExit("reference not found: %s (give a path or an id from grades.json)" % ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?")
    ap.add_argument("ours", nargs="?")
    ap.add_argument("--measure", help="just measure this file and print JSON")
    ap.add_argument("--start", type=float)
    ap.add_argument("--dur", type=float)
    ap.add_argument("--ref-start", type=float)
    ap.add_argument("--ref-dur", type=float)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--all-frames", action="store_true")
    ap.add_argument("--remeasure", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.measure:
        m, _, _, info = analyse(a.measure, a.fps, a.max, a.start, a.dur, a.all_frames)
        m["look"] = look_words(m)
        print(json.dumps(m, indent=1))
        return
    if not (a.ref and a.ours):
        ap.error("need REF and OURS (or --measure FILE)")
    refm, refp, refname = resolve_ref(a.ref, a.remeasure or a.ref_start is not None or a.ref_dur is not None or a.all_frames)
    if refm is None:
        refm, _, _, _ = analyse(refp, a.fps, a.max, a.ref_start, a.ref_dur, a.all_frames)
    ours, oframes, _, _ = analyse(a.ours, a.fps, a.max, a.start, a.dur, a.all_frames)
    lv, ss, pred = suggest(refm, ours, oframes)
    print(report(refm, ours, lv, ss, pred, refname, a.ours))
    if a.json:
        json.dump({"ref": refname, "ours": a.ours, "ref_metrics": refm, "our_metrics": ours, "levels": lv,
                   "sat_scale": ss, "predicted": pred}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
