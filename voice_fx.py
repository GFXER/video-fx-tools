"""Voice reverb transitions for offline mixes (numpy/scipy/soundfile).

Two classic editor moves:
  reverb_tail(...)    "echo out": the LAST word before a transition rings out into a reverb
                      tail (wet only after the word), so the cut/whoosh-less transition breathes.
  reverse_swell(...)  "pre-verb": a reversed reverb swells UP into the FIRST word after a
                      transition (sucked-in ghost of the word), landing exactly on its onset.
Both return a wet-only signal to ADD to the dry voice track at the returned start sample, so the
dry speech itself is untouched (no timing change, no smearing of other words).

Typical use (times in seconds on the reel timeline, voice = mono/stereo float array):
    wet, at = reverb_tail(voice, sr, word_t0, word_t1, decay=1.6, level_db=-8)
    mix[at:at+len(wet)] += wet
Suggested taste: dark and short (LP 5 kHz, decay 1.2-2.0 s, wet -6..-10 dB under the dry word),
max 1-2 per video, on a section change / before the CTA / into an insert. Duck any music bed
under the tail. Never on every cut.

usage (demo): python -I voice_fx.py in.wav t0 t1 out_tail.wav out_swell.wav
"""
import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt


def _ir(sr, decay, lp_hz=5000, predelay=0.012, seed=7):
    """Synthetic plate-like impulse response: exponentially decaying noise, darkened, stereo-ish."""
    n = int(sr * decay)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    env = np.exp(-6.9 * t / decay)                       # -60 dB at `decay`
    ir = rng.standard_normal((n, 2)) * env[:, None]
    ir = sosfilt(butter(2, lp_hz, "low", fs=sr, output="sos"), ir, axis=0)
    ir = sosfilt(butter(1, 180, "high", fs=sr, output="sos"), ir, axis=0)  # no mud
    ir = np.vstack([np.zeros((int(sr * predelay), 2)), ir])
    return ir / np.sqrt((ir ** 2).sum(axis=0, keepdims=True))


def _st(x):
    return np.repeat(x[:, None], 2, axis=1) if x.ndim == 1 else x[:, :2]


def _db(d):
    return 10 ** (d / 20)


def reverb_tail(voice, sr, t0, t1, decay=1.6, level_db=-8.0, lp_hz=5000):
    """Wet tail of the word [t0,t1]; returned wet starts at t0 and only blooms after t1."""
    a, b = int(t0 * sr), int(t1 * sr)
    word = _st(voice)[a:b]
    wet = fftconvolve(word, _ir(sr, decay, lp_hz), axes=0)
    # keep the wet quiet under the word itself, let it open as the word ends
    g = np.ones(len(wet))
    k = b - a
    g[:k] = np.linspace(0.15, 1.0, k) ** 2
    wet *= g[:, None]
    wet *= _db(level_db) * np.max(np.abs(word)) / (np.max(np.abs(wet)) + 1e-9)
    return wet, a


def reverse_swell(voice, sr, t0, t1, decay=1.2, level_db=-9.0, lp_hz=4500):
    """Reverse-reverb swell that ENDS at the onset t0 of the word [t0,t1]; returns (wet, start)."""
    a, b = int(t0 * sr), int(t1 * sr)
    word = _st(voice)[a:b][::-1]                          # reverse the word
    wet = fftconvolve(word, _ir(sr, decay, lp_hz), axes=0)[::-1]  # reverb, reverse back
    wet = wet[: len(wet) - (b - a)]                       # keep only the part BEFORE the onset
    fade = int(0.006 * sr)                                # tiny fade so it lands cleanly on the word
    wet[-fade:] *= np.linspace(1, 0, fade)[:, None]
    wet *= _db(level_db) * np.max(np.abs(word)) / (np.max(np.abs(wet)) + 1e-9)
    return wet, a - len(wet)


if __name__ == "__main__":
    src, t0, t1, out_tail, out_swell = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
    v, sr = sf.read(src, always_2d=False)
    dry = _st(v).astype(np.float64)
    for fn, out in ((reverb_tail, out_tail), (reverse_swell, out_swell)):
        wet, at = fn(dry, sr, t0, t1)
        pad = max(0, -at)
        mix = np.vstack([np.zeros((pad, 2)), dry, np.zeros((len(wet), 2))])
        at += pad
        mix[at:at + len(wet)] += wet
        if fn is reverb_tail:      # demo: silence the dry speech after the word so the tail is audible
            mix[int(t1 * sr) + pad:at + len(wet)] = wet[int(t1 * sr) - (at - pad):][: at + len(wet) - int(t1 * sr) - pad]
        sf.write(out, mix / max(1.0, np.max(np.abs(mix)) / 0.89), sr)
        print(out, "wet starts", round((at - pad) / sr, 3), "s, len", round(len(wet) / sr, 2), "s")
