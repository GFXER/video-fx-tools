#!/usr/bin/env python3
"""ramp_profile.py: per-frame time-remap profile for a speed ramp. Pure Python, no dependencies.

A shot is a dict: family, comp_in, comp_out, src_in, src_avail, src_fps (times in seconds) plus the
family's own knobs (see DEF). ramp_profile(shot, fps) returns
    times   comp time of every frame
    src     source time to show on that frame (key these on Time Remap)
    speed   speed multiplier per frame
    blend   "none" | "pixel" | "retime": the frame blending the slowest part needs
    qc      min/max speed, first/last speeds, source used, "short_source"

Families
    apex   real speed, cosine decel to a near-freeze plateau, cosine accel out on the last frames
    beat   fast -> slow -> fast between two beats
    step   real speed, each source frame held for fps / step_fps frames (slow-shutter step print)

The profile never reads past the end of the source: first the fast parts are slowed, then, if the slow
part alone is still too long, everything is scaled and qc["short_source"] is True.

    python3 ramp_profile.py '{"family":"beat","comp_in":0,"comp_out":2,"src_in":0,"src_avail":6,"src_fps":50}' 24
    python3 ramp_profile.py shot.json 30 --csv ramp.csv
"""
import argparse
import csv
import json
import math
import os

DEF = {"fast": 4.0, "slow": 0.20, "w_in": 0.20, "w_out": 0.20,                # beat (w_* = share of the shot)
       "real": 1.0, "real_f": 6, "decel_f": 8, "plateau_speed": 0.03,         # apex (frames at 24 fps)
       "accel_f": 5, "exit_speed": 2.5,
       "step_fps": 8}                                                         # step
REQUIRED = ("family", "comp_in", "comp_out", "src_in", "src_avail", "src_fps")


def _cos(a, b, p):
    p = max(0.0, min(1.0, p))
    return a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * p))


def speed_curve(shot, fps):
    """Wanted speed per comp frame, before the source length is taken into account."""
    missing = [k for k in REQUIRED if k not in shot]
    if missing:
        raise ValueError("missing: " + ", ".join(missing))
    s = {**DEF, **shot}
    n = int(round((s["comp_out"] - s["comp_in"]) * fps))
    if n < 1:
        raise ValueError("comp_out must be at least one frame after comp_in")
    if s["family"] == "apex":
        k = fps / 24.0
        rf, df, af = (int(round(s[x] * k)) for x in ("real_f", "decel_f", "accel_f"))
        total = rf + df + af
        if total > n:                                   # short shot: shrink the three parts together
            rf, df, af = (int(x * n / total) for x in (rf, df, af))
        out = []
        for f in range(n):
            if f < rf:
                out.append(s["real"])
            elif f < rf + df:
                out.append(_cos(s["real"], s["plateau_speed"], (f - rf + 1) / df))
            elif f < n - af:
                out.append(s["plateau_speed"])
            else:
                out.append(_cos(s["plateau_speed"], s["exit_speed"], (f - (n - af) + 1) / af))
        return out
    if s["family"] == "beat":
        wi, wo = s["w_in"] * n, s["w_out"] * n
        out = []
        for f in range(n):
            if f < wi:
                out.append(_cos(s["fast"], s["slow"], f / max(1.0, wi)))
            elif f >= n - wo:
                out.append(_cos(s["slow"], s["fast"], (f - (n - wo) + 1) / max(1.0, wo)))
            else:
                out.append(s["slow"])
        return out
    if s["family"] == "step":
        return [1.0] * n
    raise ValueError("unknown family: %s (apex, beat, step)" % s["family"])


def _travel(speed, fps):
    return sum(speed[:-1]) / fps


def fit_to_source(speed, fps, room):
    """Slow the speeds until they fit `room` seconds of source. Returns (speeds, short_source)."""
    if _travel(speed, fps) <= room:
        return speed, False
    lo = min(speed)
    base = [lo] * len(speed)
    floor = _travel(base, fps)
    if floor <= room:                                   # keep the slow part, squeeze what is above it
        fac = (room - floor) / (_travel(speed, fps) - floor)
        return [lo + (v - lo) * fac for v in speed], False
    fac = room / _travel(speed, fps)
    return [v * fac for v in speed], True


def ramp_profile(shot, fps):
    speed = speed_curve(shot, fps)
    s = {**DEF, **shot}
    room = max(0.0, s["src_avail"] - 1.0 / max(1, s["src_fps"]))     # the last source frame starts here
    speed, short = fit_to_source(speed, fps, room)
    src = [s["src_in"]]
    for v in speed[:-1]:
        src.append(src[-1] + v / fps)
    if s["family"] == "step":
        hold = max(1, int(round(fps / s["step_fps"])))
        src = [src[(i // hold) * hold] for i in range(len(src))]
    slowest = min(speed) * s["src_fps"] / fps           # source frames per comp frame
    blend = "retime" if slowest < 0.2 else ("pixel" if slowest < 1.0 else "none")
    qc = {"min_speed": round(min(speed), 4), "max_speed": round(max(speed), 3),
          "first2": [round(v, 2) for v in speed[:2]], "last2": [round(v, 2) for v in speed[-2:]],
          "src_used": round(src[-1] - s["src_in"], 3), "frames": len(speed), "short_source": short}
    return {"times": [round(s["comp_in"] + i / fps, 4) for i in range(len(speed))],
            "src": [round(v, 4) for v in src], "speed": [round(v, 4) for v in speed], "blend": blend, "qc": qc}


def main():
    ap = argparse.ArgumentParser(description="per-frame time-remap profile for a speed ramp")
    ap.add_argument("shot", help="shot as JSON text, or a path to a .json file")
    ap.add_argument("fps", nargs="?", type=float, default=30.0, help="comp fps (default 30)")
    ap.add_argument("--json", help="write the whole profile here")
    ap.add_argument("--csv", help="write frame,time,src,speed rows here")
    a = ap.parse_args()
    if os.path.isfile(a.shot):
        with open(a.shot) as f:
            shot = json.load(f)
    else:
        shot = json.loads(a.shot)
    try:
        r = ramp_profile(shot, a.fps)
    except ValueError as e:
        ap.error(str(e))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time", "src", "speed"])
            w.writerows(zip(range(len(r["times"])), r["times"], r["src"], r["speed"]))
    print(json.dumps({**r["qc"], "blend": r["blend"]}))


if __name__ == "__main__":
    main()
