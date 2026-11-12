"""beat_map.py track.wav -> robust tempo + phase from energy hits (kick/snare transients), hit-fit %, and per-bar energy (sections).
Writes <track>.beat.json next to it: {bpm, beat, phase, sr, dur, bars:[db...], low:[db...]}"""
import sys, json, numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt
f = sys.argv[1]; y, sr = sf.read(f, always_2d=True); m = y.mean(1)
if sr != 48000: print('NOTE sr', sr)
w = int(0.005 * sr); fr = 200
e = 20 * np.log10(np.sqrt(np.convolve(m ** 2, np.ones(w) / w, 'same'))[::w] + 1e-6)
lo = sosfilt(butter(4, 140, 'lp', fs=sr, output='sos'), m); el = 20 * np.log10(np.sqrt(np.convolve(lo ** 2, np.ones(w) / w, 'same'))[::w] + 1e-6)
def hits(x, thr):
    return np.array([i / fr for i in range(60, len(x) - 10) if np.max(x[i:i + 3]) - np.mean(x[i - 50:i - 6]) > thr and x[i] - x[i - 1] > 2.5])
ts = hits(el, 9)
if len(ts) < 40: ts = np.concatenate([ts, hits(e, 9)]); ts.sort()
best = None
for bpm in np.arange(60, 181, 0.25):
    T = 60 / bpm
    for ph in np.arange(0, T, 0.01):
        r = np.abs(((ts - ph + T / 2) % T) - T / 2); s = (r < 0.035).mean()
        if best is None or s > best[0] + 1e-9: best = (s, T, ph)
s, T, a = best
for it in range(4):
    k = np.round((ts - a) / T); g = np.abs(ts - a - k * T) < 0.045
    if g.sum() > 8:
        A = np.vstack([np.ones(g.sum()), k[g]]).T; a, T = np.linalg.lstsq(A, ts[g], rcond=None)[0]
a = a % T; k = np.round((ts - a) / T); fit = (np.abs(ts - a - k * T) < 0.045).mean()
bar = 4 * T; nb = int((len(m) / sr - a) / bar)
bars = [round(float(20 * np.log10(np.sqrt(np.mean(m[int((a + b * bar) * sr):int((a + (b + 1) * bar) * sr)] ** 2)) + 1e-9)), 1) for b in range(nb)]
lows = [round(float(20 * np.log10(np.sqrt(np.mean(lo[int((a + b * bar) * sr):int((a + (b + 1) * bar) * sr)] ** 2)) + 1e-9)), 1) for b in range(nb)]
out = {'bpm': round(60 / T, 3), 'beat': round(T, 5), 'phase': round(a, 4), 'sr': sr, 'dur': round(len(m) / sr, 2), 'hits': len(ts), 'fit': round(float(fit), 2), 'bars': bars, 'low': lows}
json.dump(out, open(f.rsplit('.', 1)[0] + '.beat.json', 'w'))
print('%s: %.2f BPM (beat %.4f s) phase %.3f | %d hits, %.0f %% on grid | %.1f s, %d bars' % (f.split('/')[-1], 60 / T, T, a, len(ts), fit * 100, len(m) / sr, nb))
print('   bars full/low:', ' '.join('%d:%.0f/%.0f' % (b, bars[b], lows[b]) for b in range(nb)))
