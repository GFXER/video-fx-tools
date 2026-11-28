#!/usr/bin/env python3
"""curve_fit.py - measure a move in a reference clip and return the After Effects ease that reproduces it.

The tool we keep using for "LEARN GRAPHS": point it at any reference region, get
  * the VALUE graph + SPEED graph (PNG)
  * frames, % of travel at 25/50/75/90 % of the time, peak-speed position, overshoot, settle, stepped/hold detection
  * the fitted AE temporal ease (2 keys): out influence/speed on key 1, in influence/speed on key 2
  * the same curve as a cubic-bezier (Flow / CSS style x1,y1,x2,y2) and as an expression
  * the named family from graph_lib.jsx (ae-motion-toolkit repo, utils/) that is closest

Needs numpy, scipy, opencv-python, matplotlib (see requirements.txt):
  python3 curve_fit.py scan  <video> [--t0 s --t1 s] --out prefix          # per-frame global motion of the whole clip -> csv + png (find the moves)
  python3 curve_fit.py fit   <video> --t0 s --t1 s [--roi x,y,w,h] [--prop auto|scale|x|y|rot] --out prefix
  python3 curve_fit.py series <values.csv|json> --fps 20 --out prefix          # fit a curve you measured another way
  python3 curve_fit.py curve <family> [--frames N]                           # print the AE settings of a named family
  fit --all: fit every property that moved (scale / rot / x / y), one graph + keys file each.
  With --out, every fit also writes prefix_keys.json (2 AE keys: times, values, influence/speed) and prefix_keys.jsx
  (run it in AE: puts those keys at the playhead on the selected layers, relative to their current value).

Measurement:
  global (no --roi): similarity transform between consecutive frames (ORB/LK + RANSAC, fallback log-polar phase
      correlation), accumulated -> scale (zoom), rotation, x/y (fractions of the frame). Use it for camera zoom-outs,
      push-ins, tilts, whole-frame slides.
  --roi x,y,w,h (fractions 0-1 of the first frame): features inside the box are tracked with LK optical flow from the
      FIRST frame of the window; similarity vs frame t0 -> the element's own scale / position / rotation (layer pops,
      strip slides, sticker moves). Bounding-box mode (--bbox) thresholds the region instead (for flat-colour pops).
AE ease maths (speeds expressed normalised, i.e. units of total change per total duration):
  key1 out: influence i0, speed s0 -> bezier P1 = (i0, s0*i0);  key2 in: influence i1, speed s1 -> P2 = (1-i1, 1-s1*i1)
  so with speed 0 on both keys the AE ease is EXACTLY css cubic-bezier(i0, 0, 1-i1, 1).
"""
import sys, os, json, argparse, math
import numpy as np

FAMILIES = {
    # name: (x1, y1, x2, y2, note) - mean curves measured 2026-11-09; keep in sync with ae_snippets/graph_lib.jsx
    "zine_whip_entry":  (0.114, 0.625, 0.218, 0.962, "after a cut, starts at peak speed: AE out 11 % speed 5.5x avg / in 78 % speed 0"),
    "zine_whip_exit":   (0.951, 0.009, 0.863, 0.374, "into a cut, peak on the last frame: AE out 95 % / in 14 % speed 4.6x avg"),
    "zine_snap_step":   (0.292, 0.809, 0.212, 1.039, "2-3 f snap zoom inside a held collage: AE out 29 % speed 2.8x / in 79 % speed -0.05x (overshoot)"),
    "through_cut_whip": (0.749, -0.092, 0.0, 1.196, "single 2-key move spanning a cut: AE out 75 / in 100, peak at 40 %"),
    "ui_tilt_s":        (0.782, 0.16, 0.146, 1.039, "lyric-app tilt: AE out 78 / in 85, S-curve"),
    "ios_scroll":       (0.384, 0.269, 0.006, 1.041, "lyric scroll: AE out 38 speed 0.7x / in 99"),
    "droste_linear":    (0.382, 0.48, 0.653, 0.305, "tunnel zoom keyed in %: AE out 38 speed 1.3x / in 35 speed 2x"),
    "expo_out":         (0.05, 0.75, 0.10, 1.00, "house expo-out: 80 % of the move in 2-3 f"),
    "flow_051":         (0.51, 0.00, 0.00, 1.00, "house Flow (0.51,0,0,1) = out 51 / in 100"),
    "house_30_90":      (0.30, 0.00, 0.10, 1.00, "house camera ease out 30 / in 90"),
    "easy_ease":        (0.333, 0.00, 0.667, 1.00, "AE F9 default 33/33"),
    "linear":           (0.00, 0.00, 1.00, 1.00, "constant speed"),
}


# ------------------------------------------------------------------------------------------------ bezier / AE ease
def bez_eval(x1, y1, x2, y2, t, n=4001):
    u = np.linspace(0, 1, n)
    x = 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u ** 2 * x2 + u ** 3
    y = 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u ** 2 * y2 + u ** 3
    x = np.maximum.accumulate(x)
    return np.interp(t, x, y)


def ae_to_bez(i0, s0, i1, s1):
    """AE influences (0-1) and normalised speeds -> cubic bezier."""
    return i0, s0 * i0, 1 - i1, 1 - s1 * i1


def bez_to_ae(x1, y1, x2, y2, dv, dur):
    """cubic bezier -> AE KeyframeEase settings (influence %, speed in value-units / s)."""
    i0 = max(0.001, x1); i1 = max(0.001, 1 - x2)
    s0 = (y1 / i0) * dv / dur if x1 > 1e-4 else (0.0 if y1 == 0 else float('inf'))
    s1 = ((1 - y2) / i1) * dv / dur if (1 - x2) > 1e-4 else 0.0
    return dict(out_influence=round(100 * max(0.001, min(1, i0)), 1), out_speed=round(s0, 3),
                in_influence=round(100 * max(0.001, min(1, i1)), 1), in_speed=round(s1, 3))


def fit_bezier(tn, pn, allow_speed=True):
    """least squares over bezier influence pairs (and optional start/end speed). tn, pn normalised 0..1."""
    from scipy.optimize import least_squares
    best = None
    def res(p):
        if allow_speed:
            i0, i1, s0, s1 = p
        else:
            i0, i1 = p; s0 = s1 = 0
        x1, y1, x2, y2 = ae_to_bez(i0, s0, i1, s1)
        return bez_eval(x1, y1, x2, y2, tn) - pn
    starts = [(a, b) for a in (0.02, 0.2, 0.5, 0.85) for b in (0.02, 0.33, 0.7, 0.95)]
    for a, b in starts:
        if allow_speed:
            for s0 in (0.0, 1.0, 4.0, 12.0):
                for s1 in (0.0, 0.5, -1.0):
                    r = least_squares(res, [a, b, s0, s1], bounds=([0.001, 0.001, -2, -8], [1, 1, 60, 8]))
                    if best is None or r.cost < best.cost: best = r
        else:
            r = least_squares(res, [a, b], bounds=([0.001, 0.001], [1, 1]))
            if best is None or r.cost < best.cost: best = r
    p = list(best.x) + ([] if allow_speed else [0, 0])
    i0, i1, s0, s1 = p
    rms = float(np.sqrt(np.mean(res(best.x) ** 2)))
    return dict(i0=i0, i1=i1, s0=s0, s1=s1, bez=ae_to_bez(i0, s0, i1, s1), rms=rms)


def closest_family(tn, pn):
    out = []
    for k, (x1, y1, x2, y2, _) in FAMILIES.items():
        e = float(np.sqrt(np.mean((bez_eval(x1, y1, x2, y2, tn) - pn) ** 2)))
        out.append((e, k))
    out.sort()
    return out


# ------------------------------------------------------------------------------------------------ video reading
def read_frames(video, t0=None, t1=None, width=480):
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = 0 if t0 is None else int(round(t0 * fps))
    f1 = n - 1 if t1 is None else min(n - 1, int(round(t1 * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = []
    for f in range(f0, f1 + 1):
        ok, im = cap.read()
        if not ok: break
        h, w = im.shape[:2]
        if w > width: im = cv2.resize(im, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
        frames.append(im)
    return frames, fps, f0


def _sim_from_pts(a, b):
    import cv2
    if a is None or len(a) < 6: return None, 0
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=4000, confidence=0.995)
    if M is None: return None, 0
    return M, int(inl.sum()) if inl is not None else 0


def pair_similarity(g0, g1, mask=None):
    """similarity transform g0 -> g1 (2x3) + inlier count. LK on corners, ORB fallback, log-polar fallback."""
    import cv2
    p0 = cv2.goodFeaturesToTrack(g0, 600, 0.01, 6, mask=mask)
    M, ninl = None, 0
    if p0 is not None and len(p0) >= 8:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=4)
        ok = st.ravel() == 1
        M, ninl = _sim_from_pts(p0[ok], p1[ok])
    if M is None or ninl < 10:
        orb = cv2.ORB_create(1500)
        k0, d0 = orb.detectAndCompute(g0, mask); k1, d1 = orb.detectAndCompute(g1, None)
        if d0 is not None and d1 is not None and len(k0) > 8 and len(k1) > 8:
            m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1)
            if len(m) > 8:
                a = np.float32([k0[x.queryIdx].pt for x in m]); b = np.float32([k1[x.trainIdx].pt for x in m])
                M2, n2 = _sim_from_pts(a, b)
                if M2 is not None and n2 > ninl: M, ninl = M2, n2
    return M, ninl


def logpolar_scale_rot(g0, g1):
    """log-polar phase correlation (global zoom/rotation only, about the centre)."""
    import cv2
    h, w = g0.shape; c = (w / 2, h / 2); R = min(c)
    def lp(g):
        F = np.abs(np.fft.fftshift(np.fft.fft2(np.float32(g) * np.outer(np.hanning(h), np.hanning(w)))))
        F = np.log1p(F)
        return cv2.warpPolar(np.float32(F), (w, h), c, R, cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR)
    a, b = lp(g0), lp(g1)
    (dx, dy), resp = cv2.phaseCorrelate(a, b)
    rot = -dy * 360.0 / h
    scale = math.exp(dx * math.log(R) / w)
    return scale, rot, resp


def M_params(M):
    s = math.hypot(M[0, 0], M[1, 0]); r = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    return s, r, M[0, 2], M[1, 2]


def to3(M):
    return np.vstack([M, [0, 0, 1]])


def scan(video, t0=None, t1=None, out=None):
    import cv2
    frames, fps, f0 = read_frames(video, t0, t1)
    h, w = frames[0].shape[:2]
    rows = []
    for k in range(1, len(frames)):
        g0 = cv2.cvtColor(frames[k - 1], cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY)
        diff = float(np.mean(np.abs(g0.astype(np.int16) - g1.astype(np.int16))))
        M, ninl = pair_similarity(g0, g1)
        if M is None: s, r, tx, ty = 1, 0, 0, 0
        else:
            # express about the frame centre
            C = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]]); Ci = np.linalg.inv(C)
            Mc = (C @ to3(M) @ Ci)[:2]
            s, r, tx, ty = M_params(Mc)
        rows.append(dict(frame=f0 + k, t=round((f0 + k) / fps, 3), dscale=round(s, 5), drot=round(r, 4),
                         dx=round(tx / w, 5), dy=round(ty / h, 5), inliers=ninl, diff=round(diff, 2),
                         cut=bool(diff > 40 or ninl < 8)))
    if out:
        import csv
        with open(out + '_scan.csv', 'w') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
        _plot_scan(rows, out + '_scan.png', os.path.basename(video))
    return rows, fps


def _plot_scan(rows, png, title):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    t = [r['t'] for r in rows]
    fig, ax = plt.subplots(4, 1, figsize=(16, 9), sharex=True)
    ax[0].plot(t, [100 * (r['dscale'] - 1) for r in rows], '.-', ms=3); ax[0].set_ylabel('zoom %/f')
    ax[1].plot(t, [r['drot'] for r in rows], '.-', ms=3); ax[1].set_ylabel('rot deg/f')
    ax[2].plot(t, [100 * r['dx'] for r in rows], '.-', ms=3, label='x'); ax[2].plot(t, [100 * r['dy'] for r in rows], '.-', ms=3, label='y')
    ax[2].set_ylabel('pos %frame/f'); ax[2].legend()
    ax[3].plot(t, [r['diff'] for r in rows], '.-', ms=3); ax[3].set_ylabel('mean |diff|')
    for r in rows:
        if r['cut']:
            for a in ax: a.axvline(r['t'], color='r', lw=0.4, alpha=0.5)
    for a in ax: a.grid(alpha=0.3)
    ax[0].set_title(title + '  (red = cut / unmatched frame)'); ax[3].set_xlabel('s'); ax[3].set_xticks(np.arange(0, max(t) + 0.5, 0.5))
    fig.tight_layout(); fig.savefig(png, dpi=80); plt.close(fig)


def track(video, t0, t1, roi=None, bbox=False, thr=None):
    """per-frame absolute transform relative to frame t0: scale, rot (deg), x, y (fraction of frame)."""
    import cv2
    frames, fps, f0 = read_frames(video, t0, t1)
    h, w = frames[0].shape[:2]
    G = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    S, R, X, Y, Q = [1.0], [0.0], [0.0], [0.0], [999]
    if bbox and roi:
        # threshold the roi area: element = pixels differing from the median of the roi in the first frame
        x, y, ww, hh = [int(v) for v in (roi[0] * w, roi[1] * h, roi[2] * w, roi[3] * h)]
        base = None
        S, X, Y, R = [], [], [], []
        for g in G:
            sub = g[y:y + hh, x:x + ww].astype(np.float32)
            med = np.median(sub) if thr is None else thr
            m = np.abs(sub - med) > 30
            ys, xs = np.nonzero(m)
            if len(xs) < 20: S.append(np.nan); X.append(np.nan); Y.append(np.nan); R.append(np.nan); continue
            S.append(math.sqrt(len(xs))); X.append((xs.mean() + x) / w); Y.append((ys.mean() + y) / h)
            c = np.cov(np.vstack([xs, ys])); ev, evec = np.linalg.eigh(c); R.append(math.degrees(math.atan2(evec[1, 1], evec[0, 1])))
        S = np.array(S); S = S / S[np.isfinite(S)][0]
        X = np.array(X) - X[0]; Y = np.array(Y) - Y[0]
        return dict(fps=fps, f0=f0, scale=S, rot=np.array(R) - R[0], x=X, y=Y, q=None, w=w, h=h)
    mask = None
    if roi:
        mask = np.zeros_like(G[0]); x, y, ww, hh = [int(v) for v in (roi[0] * w, roi[1] * h, roi[2] * w, roi[3] * h)]
        mask[y:y + hh, x:x + ww] = 255
    C = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]]); Ci = np.linalg.inv(C)
    if roi:
        cx, cy = (roi[0] + roi[2] / 2) * w, (roi[1] + roi[3] / 2) * h
        C = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]]); Ci = np.linalg.inv(C)
        p0 = cv2.goodFeaturesToTrack(G[0], 400, 0.01, 4, mask=mask)
        pts = p0; Acc = np.eye(3)
        for k in range(1, len(G)):
            p1, st, _ = cv2.calcOpticalFlowPyrLK(G[k - 1], G[k], pts, None, winSize=(21, 21), maxLevel=4)
            ok = st.ravel() == 1
            M, n = _sim_from_pts(p0[ok], p1[ok])
            if M is None: M = np.array([[1, 0, 0], [0, 1, 0]], float); n = 0
            p0, pts = p0[ok], p1[ok]
            Mc = (C @ to3(M) @ Ci)[:2]; s, r, tx, ty = M_params(Mc)
            S.append(s); R.append(r); X.append(tx / w); Y.append(ty / h); Q.append(n)
            if len(pts) < 12: break
    else:
        Acc = np.eye(3)
        for k in range(1, len(G)):
            M, n = pair_similarity(G[k - 1], G[k])
            if M is None:
                sc, rt, _ = logpolar_scale_rot(G[k - 1], G[k]); M = cv2.getRotationMatrix2D((w / 2, h / 2), -rt, sc)
            Acc = to3(M) @ Acc
            Mc = (C @ Acc @ Ci)[:2]; s, r, tx, ty = M_params(Mc)
            S.append(s); R.append(r); X.append(tx / w); Y.append(ty / h); Q.append(n)
    return dict(fps=fps, f0=f0, scale=np.array(S), rot=np.array(R), x=np.array(X), y=np.array(Y), q=np.array(Q), w=w, h=h)


# ------------------------------------------------------------------------------------------------ analysis
def analyse(v, fps, name='value', units='', stepped_tol=None, end='auto'):
    v = np.asarray(v, float)
    good = np.isfinite(v); v = np.interp(np.arange(len(v)), np.nonzero(good)[0], v[good])
    n = len(v); T = (n - 1) / fps
    d = np.diff(v)
    # end value: median of the last 3 samples (the settled value)
    v0, v1 = v[0], np.median(v[-3:]) if n > 4 else v[-1]
    if end == 'last' or (end == 'auto' and n > 4 and abs(d[-1]) >= 0.95 * np.max(np.abs(d))):
        v1 = v[-1]  # exit move (accelerating into a cut): the end key sits ON the cut with full speed
    dv = v1 - v0
    if abs(dv) < 1e-9: dv = 1e-9
    p = (v - v0) / dv
    # overshoot = max progress beyond 1
    over = float(max(0.0, p.max() - 1.0))
    i_peak_val = int(np.argmax(p))
    # settle: first frame after which |p-1| < 2 %
    settle = None
    for k in range(n):
        if np.all(np.abs(p[k:] - 1) < 0.02): settle = k; break
    # stepped detection: share of near-zero deltas inside the moving part
    mv = np.abs(d) / abs(dv)
    moving = np.nonzero(mv > 0.01)[0]
    held = 0; step = None
    if len(moving) > 1:
        seg = mv[moving[0]:moving[-1] + 1]
        held = float(np.mean(seg < 0.002))
        z = (seg < 0.002).astype(int)
        if held > 0.25:
            runs = []; c = 0
            for zz in z:
                if zz: c += 1
                elif c: runs.append(c); c = 0
            step = 1 + (int(np.median(runs)) if runs else 0)
    tn = np.arange(n) / (n - 1)
    at = {}
    for q in (0.25, 0.5, 0.75, 0.9):
        at[f'{int(q * 100)}%'] = round(float(np.interp(q, tn, p)) * 100, 1)
    spd = np.abs(np.gradient(p, tn))
    pk = int(np.argmax(spd))
    fit = fit_bezier(tn, p)  # full curve; overshoot is reproduced by a NEGATIVE incoming speed (AE-native, 2 keys)
    fit0 = fit_bezier(tn, p, allow_speed=False)
    fam = closest_family(tn, np.clip(p, -0.2, 1.2))
    x1, y1, x2, y2 = fit['bez']
    ae = bez_to_ae(x1, y1, x2, y2, dv, T)
    ae0 = bez_to_ae(*fit0['bez'], dv, T)
    expr = (f"// {name}: fitted ease, apply between key 1 and key 2 (t0/t1, v0/v1)\n"
            f"function bz(t){{var x1={x1:.3f},y1={y1:.3f},x2={x2:.3f},y2={y2:.3f};var lo=0,hi=1,u=t;for(var k=0;k<30;k++){{u=(lo+hi)/2;var x=3*(1-u)*(1-u)*u*x1+3*(1-u)*u*u*x2+u*u*u;if(x<t)lo=u;else hi=u;}}"
            f"return 3*(1-u)*(1-u)*u*y1+3*(1-u)*u*u*y2+u*u*u;}}\n"
            f"var a=key(1),b=key(2);var t=clamp((time-a.time)/(b.time-a.time),0,1);a.value+(b.value-a.value)*bz(t);")
    return dict(name=name, frames=n, dur_s=round(T, 3), fps=fps, v0=round(float(v0), 4), v1=round(float(v1), 4), units=units,
                travel_at_time=at, peak_speed_at=round(float(tn[pk]) * 100, 1), peak_speed_norm=round(float(spd[pk]), 2),
                overshoot_pct=round(over * 100, 1), overshoot_frame=i_peak_val if over > 0.02 else None,
                settle_frame=settle, held_share=round(held, 2), step_frames=step, stepped=bool(held > 0.25),
                fit=dict(bezier=[round(float(c), 3) for c in fit['bez']], rms_pct=round(100 * fit['rms'], 2), ae=ae),
                fit_speed0=dict(bezier=[round(float(c), 3) for c in fit0['bez']], rms_pct=round(100 * fit0['rms'], 2), ae=ae0),
                family=[dict(name=k, rms_pct=round(100 * e, 2)) for e, k in fam[:3]], expression=expr, _p=p, _tn=tn)


def plot_fit(res, png, title=''):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    tn, p = res['_tn'], res['_p']; n = res['frames']; fps = res['fps']
    fr = np.arange(n)
    x1, y1, x2, y2 = res['fit']['bezier']
    tt = np.linspace(0, 1, 400); fitp = bez_eval(x1, y1, x2, y2, tt)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(fr, p * 100, 'o', ms=5, label='measured (ref)')
    ax[0].plot(tt * (n - 1), fitp * 100, '-', lw=2, label=f"fit bez({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}) rms {res['fit']['rms_pct']}%")
    fam = res['family'][0]['name']; f = FAMILIES[fam]
    ax[0].plot(tt * (n - 1), bez_eval(*f[:4], tt) * 100, '--', lw=1, label=f'family {fam}')
    ax[0].set_title(f"VALUE  {title}  ({res['v0']}->{res['v1']} {res['units']}, {n} f @{fps:g})"); ax[0].set_xlabel('frame'); ax[0].set_ylabel('% of travel')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    sp = np.gradient(p, tn)
    ax[1].bar(fr, sp, width=0.6, alpha=.5, label='measured speed')
    ax[1].plot(tt * (n - 1), np.gradient(fitp, tt), '-', lw=2, label='fit speed')
    ax[1].set_title(f"SPEED  peak at {res['peak_speed_at']}% of time  ae: out {res['fit']['ae']['out_influence']}% / in {res['fit']['ae']['in_influence']}%")
    ax[1].set_xlabel('frame'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(png, dpi=80); plt.close(fig)


def clean(res):
    return {k: v for k, v in res.items() if not k.startswith('_')}


# ------------------------------------------------------------------------------------------------ AE keyframe export
# prop -> (transform match name, how the measured values map onto the layer's own value)
#   scale: measured % relative to 100 at t0 -> multiplies the layer's scale (a 100->150 pop on an 80 % layer = 80->120)
#   rot:   degrees, added to the layer's rotation
#   x / y: % of the frame width / height, converted to comp pixels when applied, added to the (separated) position
AE_PROPS = {
    'scale': ('ADBE Scale', 'multiply'),
    'rot':   ('ADBE Rotate Z', 'add'),
    'x':     ('ADBE Position_0', 'add_frac_w'),
    'y':     ('ADBE Position_1', 'add_frac_h'),
}
AE_UNITS = dict(scale='%', rot='deg', x='% frame W', y='% frame H')
SPEED_CAP = 1e6  # AE rejects non-finite speeds; a near-zero influence can give a huge one


def _finite(x, cap=SPEED_CAP):
    x = float(x)
    if not math.isfinite(x): return math.copysign(cap, x) if not math.isnan(x) else 0.0
    return max(-cap, min(cap, x))


def ae_keyframes(res, prop=None):
    """the fitted 2-key move as AE keyframe data: times relative to the start key, values in the measured units.

    prop = scale | rot | x | y names the layer transform property it was measured on (None = any 1D property,
    values added to it). Speeds are AE KeyframeEase speeds (value units / s, signed), influences in %."""
    ae = res['fit']['ae']
    match, mode = AE_PROPS.get(prop, (None, 'add'))
    keys = [dict(t=0.0, value=res['v0'], out=dict(influence=ae['out_influence'], speed=_finite(ae['out_speed']))),
            dict(t=res['dur_s'], value=res['v1'], **{'in': dict(influence=ae['in_influence'], speed=_finite(ae['in_speed']))})]
    return dict(format='curve_fit.ae_keys/1', name=res['name'], prop=prop, match_name=match, mode=mode,
                units=res['units'], fps=res['fps'], frames=res['frames'], keys=keys,
                bezier=res['fit']['bezier'], rms_pct=res['fit']['rms_pct'],
                family=res['family'][0]['name'] if res.get('family') else None)


AE_JSX = r"""// curve_fit.py keys: %(name)s  (%(frames)d f @ %(fps)g fps, fit rms %(rms)s %%)
// Run in AE (File > Scripts > Run Script File). Puts the 2 keys at the playhead:
//   with a transform property (scale/rot/x/y): on every selected layer, relative to the layer's current value;
//   without one: on every selected 1D property.
(function () {
    var K = %(json)s;
    var comp = app.project.activeItem;
    if (!(comp instanceof CompItem)) { alert("curve_fit: open a comp first"); return; }
    var mul = K.mode === "multiply" && K.keys[0].value !== 0;  // a series that starts at 0 can only be added
    function mapped(v, base, d) {
        if (mul) return base * v / K.keys[0].value;
        if (K.mode === "add_frac_w") return base + (v - K.keys[0].value) / 100 * comp.width;
        if (K.mode === "add_frac_h") return base + (v - K.keys[0].value) / 100 * comp.height;
        return base + (v - K.keys[0].value);
    }
    function speedScale(base) {
        if (mul) return base / K.keys[0].value;
        if (K.mode === "add_frac_w") return comp.width / 100;
        if (K.mode === "add_frac_h") return comp.height / 100;
        return 1;
    }
    function eases(e, bases) {
        var a = [];
        for (var d = 0; d < bases.length; d++) a.push(new KeyframeEase(e.speed * speedScale(bases[d]), Math.max(0.1, Math.min(100, e.influence))));
        return a;
    }
    function apply(p) {
        var t0 = comp.time, cur = p.valueAtTime(t0, false);
        var arr = (typeof cur !== "number"), bases = arr ? cur : [cur], k, d, v, idx = [];
        for (k = 0; k < K.keys.length; k++) {
            if (arr) { v = []; for (d = 0; d < bases.length; d++) v.push(mapped(K.keys[k].value, bases[d], d)); }
            else v = mapped(K.keys[k].value, cur, 0);
            p.setValueAtTime(t0 + K.keys[k].t, v);
        }
        for (k = 0; k < K.keys.length; k++) idx.push(p.nearestKeyIndex(t0 + K.keys[k].t));
        for (k = 0; k < idx.length; k++) p.setInterpolationTypeAtKey(idx[k], KeyframeInterpolationType.BEZIER, KeyframeInterpolationType.BEZIER);
        p.setTemporalEaseAtKey(idx[0], p.keyInTemporalEase(idx[0]), eases(K.keys[0].out, bases));
        p.setTemporalEaseAtKey(idx[1], eases(K.keys[1]["in"], bases), p.keyOutTemporalEase(idx[1]));
    }
    app.beginUndoGroup("curve_fit " + K.name);
    var n = 0, i;
    if (K.match_name) {
        var L = comp.selectedLayers;
        for (i = 0; i < L.length; i++) {
            var tr = L[i].property("ADBE Transform Group");
            if (K.match_name.indexOf("ADBE Position_") === 0) {
                var pos = tr.property("ADBE Position");
                if (!pos.dimensionsSeparated) pos.dimensionsSeparated = true;
            }
            var p = tr.property(K.match_name);
            if (p) { apply(p); n++; }
        }
    } else {
        var P = comp.selectedProperties;
        for (i = 0; i < P.length; i++) {
            if (P[i].propertyType === PropertyType.PROPERTY && P[i].canVaryOverTime && P[i].propertyValueType === PropertyValueType.OneD) { apply(P[i]); n++; }
        }
    }
    app.endUndoGroup();
    if (!n) alert(K.match_name ? "curve_fit: select a layer" : "curve_fit: select a 1D property");
})();
"""


def ae_jsx(keys):
    """an ExtendScript file that applies ae_keyframes() data (values embedded, so it needs no JSON parser in AE)."""
    return AE_JSX % dict(name=' '.join(str(keys['name']).split()), frames=keys['frames'], fps=keys['fps'], rms=keys['rms_pct'],
                         json=json.dumps(keys, allow_nan=False))


def write_outputs(res, prefix, prop=None):
    """prefix.png (value + speed graph), prefix.json (analysis), prefix_keys.json + prefix_keys.jsx (AE keys)."""
    plot_fit(res, prefix + '.png', res['name'])
    with open(prefix + '.json', 'w') as f: json.dump(clean(res), f, indent=1)
    keys = ae_keyframes(res, prop)
    with open(prefix + '_keys.json', 'w') as f: json.dump(keys, f, indent=1, allow_nan=False)
    with open(prefix + '_keys.jsx', 'w') as f: f.write(ae_jsx(keys))
    return keys


# a property counts as "moving" (for --all) when its range passes this, in its own units
MOVE_MIN = dict(scale=0.5, rot=0.5, x=0.5, y=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('src', nargs='?')
    ap.add_argument('--t0', type=float); ap.add_argument('--t1', type=float)
    ap.add_argument('--roi'); ap.add_argument('--bbox', action='store_true')
    ap.add_argument('--prop', default='auto'); ap.add_argument('--fps', type=float, default=20)
    ap.add_argument('--frames', type=int, default=20); ap.add_argument('--out'); ap.add_argument('--name', default='move'); ap.add_argument('--end', default='auto')
    ap.add_argument('--all', action='store_true', help='fit: fit every property that moves, one png + keys file each')
    a = ap.parse_args()
    if a.prop not in ('auto', 'scale', 'rot', 'x', 'y'):
        ap.error('--prop must be auto, scale, rot, x or y')
    if a.cmd == 'curve':
        x1, y1, x2, y2, note = FAMILIES[a.src]
        print(json.dumps(dict(family=a.src, bezier=[x1, y1, x2, y2], ae=bez_to_ae(x1, y1, x2, y2, 1.0, a.frames / a.fps), note=note), indent=1)); return
    if a.cmd == 'scan':
        rows, fps = scan(a.src, a.t0, a.t1, a.out or 'scan'); print(f'{len(rows)} frame pairs -> {a.out}_scan.csv/.png'); return
    if a.cmd == 'series':
        v = json.load(open(a.src)) if a.src.endswith('.json') else np.loadtxt(a.src, delimiter=',')
        prop = None if a.prop == 'auto' else a.prop  # series values: say which transform property they are, if any
        res = analyse(v, a.fps, a.name, AE_UNITS.get(prop, ''), end=a.end)
    elif a.cmd == 'fit':
        roi = [float(x) for x in a.roi.split(',')] if a.roi else None
        tr = track(a.src, a.t0, a.t1, roi, a.bbox)
        cand = dict(scale=(tr['scale'] * 100, AE_UNITS['scale']), rot=(tr['rot'], AE_UNITS['rot']),
                    x=(tr['x'] * 100, AE_UNITS['x']), y=(tr['y'] * 100, AE_UNITS['y']))
        mag = {k: (float(np.nanmax(c[0]) - np.nanmin(c[0])) if np.isfinite(c[0]).any() else 0.0) for k, c in cand.items()}
        all_props = {k: [round(float(x), 3) for x in np.nan_to_num(c[0])] for k, c in cand.items()}
        if a.all:
            moving = [k for k in cand if mag[k] >= MOVE_MIN[k]]
            if not moving:
                print(json.dumps(dict(moving=[], range=mag), indent=1)); return
            out = {}
            for k in moving:
                rk = analyse(cand[k][0], tr['fps'], f'{a.name}.{k}', cand[k][1], end=a.end)
                if a.out: write_outputs(rk, f'{a.out}_{k}', k)
                r = clean(rk); r.pop('expression', None); out[k] = r
            print(json.dumps(dict(moving=moving, fits=out), indent=1)); return
        if a.prop == 'auto':
            weighted = dict(mag, rot=mag['rot'] * 0.6)  # a degree of rotation reads smaller than 1 % of travel
            prop = max(weighted, key=weighted.get)
        else:
            prop = a.prop
        res = analyse(cand[prop][0], tr['fps'], f'{a.name}.{prop}', cand[prop][1], end=a.end)
        res['all_props'] = all_props
    else:
        ap.error(f'unknown command {a.cmd!r} (scan, fit, series, curve)')
    if a.out:
        write_outputs(res, a.out, prop)
    r = clean(res); r.pop('all_props', None); r.pop('expression', None)
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
