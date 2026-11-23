"""beat_map.py track.wav -> robust tempo + phase from energy hits (kick/snare transients), hit-fit %, and per-bar energy (sections).
Writes <track>.beat.json next to it: {bpm, beat, phase, sr, dur, hits, fit, bars:[db...], low:[db...]}

Optional:
  --grid beat|8th|16th   add a "grid" list to the json: one entry per grid line {t, bar, beat, sub, kind}
  --csv markers.csv      write the grid (default: beats) as a marker list with timecodes, columns
                         Marker Name, Description, In, Out, Duration, Marker Type (Premiere's marker-list layout)
  --fps 30               timecode rate for --csv (fractional rates like 29.97 use non-drop timecode at the rounded rate)
"""
import argparse, csv, json, sys
import numpy as np

SUBDIV = {'beat': 1, '8th': 2, '16th': 4}


def make_grid(beat, phase, dur, sub=1, beats_per_bar=4):
    """Grid lines from `phase` to `dur` every beat/sub seconds. bar/beat/sub are 1-based; kind is
    downbeat | beat | 8th | 16th. Lines before `phase` are skipped (bar 1 beat 1 = the first detected beat)."""
    step = beat / sub
    out, i = [], 0
    while True:
        t = phase + i * step
        if t >= dur - 1e-9:
            break
        b, s = divmod(i, sub)
        bar, bt = divmod(b, beats_per_bar)
        if s == 0:
            kind = 'downbeat' if bt == 0 else 'beat'
        else:
            kind = '8th' if sub == 2 or s == 2 else '16th'
        out.append({'t': round(t, 4), 'bar': bar + 1, 'beat': bt + 1, 'sub': s + 1, 'kind': kind})
        i += 1
    return out


def timecode(t, fps):
    """Seconds -> HH:MM:SS:FF (non-drop) at round(fps) frames per second, frames counted from real time."""
    nf = int(round(fps))
    f = int(round(t * fps))
    hh, rem = divmod(f, 3600 * nf)
    mm, rem = divmod(rem, 60 * nf)
    ss, ff = divmod(rem, nf)
    return '%02d:%02d:%02d:%02d' % (hh, mm, ss, ff)


def write_markers_csv(path, grid, fps):
    zero = timecode(0, fps)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['Marker Name', 'Description', 'In', 'Out', 'Duration', 'Marker Type'])
        for g in grid:
            name = 'Bar %d' % g['bar'] if g['kind'] == 'downbeat' else (
                '%d.%d' % (g['bar'], g['beat']) if g['kind'] == 'beat' else '%d.%d.%d' % (g['bar'], g['beat'], g['sub']))
            tc = timecode(g['t'], fps)
            w.writerow([name, g['kind'], tc, tc, zero, 'Comment'])


def analyse(path):
    import soundfile as sf
    from scipy.signal import butter, sosfilt
    y, sr = sf.read(path, always_2d=True); m = y.mean(1)
    if sr != 48000: print('NOTE sr', sr)
    w = int(0.005 * sr); fr = 200
    e = 20 * np.log10(np.sqrt(np.convolve(m ** 2, np.ones(w) / w, 'same'))[::w] + 1e-6)
    lo = sosfilt(butter(4, 140, 'lp', fs=sr, output='sos'), m); el = 20 * np.log10(np.sqrt(np.convolve(lo ** 2, np.ones(w) / w, 'same'))[::w] + 1e-6)

    def hits(x, thr, gap=0.09, rise=0.5):
        # one hit per transient: the peak must also clear the last ~50 ms by `rise` dB (a decaying low kick ripples in the
        # 5 ms envelope and used to re-trigger every ~25 ms), and re-triggers within `gap` s are ignored
        out, last = [], -1e9
        for i in range(60, len(x) - 10):
            pk = np.max(x[i:i + 3])
            if pk - np.mean(x[i - 50:i - 6]) > thr and x[i] - x[i - 1] > 2.5 and pk - np.max(x[i - 10:i - 1]) > rise and i / fr - last >= gap:
                out.append(i / fr); last = i / fr
        return np.array(out)
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
    return {'bpm': round(60 / T, 3), 'beat': round(float(T), 5), 'phase': round(float(a), 4), 'sr': sr, 'dur': round(len(m) / sr, 2),
            'hits': len(ts), 'fit': round(float(fit), 2), 'bars': bars, 'low': lows}


def main(argv=None):
    ap = argparse.ArgumentParser(description='Tempo, phase and per-bar energy from a music file.')
    ap.add_argument('track')
    ap.add_argument('--grid', choices=sorted(SUBDIV), help='add a beat/8th/16th grid to the json')
    ap.add_argument('--csv', metavar='OUT', help='write grid markers (HH:MM:SS:FF) to a CSV file')
    ap.add_argument('--fps', type=float, default=30.0, help='timecode rate for --csv (default 30)')
    o = ap.parse_args(argv)
    if o.fps <= 0:
        ap.error('--fps must be > 0')
    f = o.track
    out = analyse(f)
    T, a, nb = out['beat'], out['phase'], len(out['bars'])
    grid = None
    if o.grid or o.csv:
        grid = make_grid(out['beat'], out['phase'], out['dur'], SUBDIV[o.grid or 'beat'])
    if o.grid:
        out['grid'] = grid
    json.dump(out, open(f.rsplit('.', 1)[0] + '.beat.json', 'w'))
    if o.csv:
        write_markers_csv(o.csv, grid, o.fps)
    print('%s: %.2f BPM (beat %.4f s) phase %.3f | %d hits, %.0f %% on grid | %.1f s, %d bars' % (f.split('/')[-1], 60 / T, T, a, out['hits'], out['fit'] * 100, out['dur'], nb))
    print('   bars full/low:', ' '.join('%d:%.0f/%.0f' % (b, out['bars'][b], out['low'][b]) for b in range(nb)))
    if o.csv:
        print('   %d %s markers -> %s @ %g fps' % (len(grid), o.grid or 'beat', o.csv, o.fps))


if __name__ == '__main__':
    sys.exit(main())
