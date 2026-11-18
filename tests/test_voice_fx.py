import numpy as np
import pytest
import soundfile as sf

import voice_fx as vf
from conftest import run_tool

SR = 24000


def _voice(dur=2.0, t0=0.8, t1=1.2):
    """Silence with one 'word' (a 220 Hz burst) between t0 and t1."""
    t = np.arange(int(dur * SR)) / SR
    x = np.zeros_like(t)
    m = (t >= t0) & (t < t1)
    x[m] = 0.5 * np.sin(2 * np.pi * 220 * t[m])
    return x


def test_ir_is_stereo_and_unit_energy():
    ir = vf._ir(SR, decay=0.8)
    assert ir.ndim == 2 and ir.shape[1] == 2
    assert np.allclose((ir ** 2).sum(axis=0), 1.0, atol=1e-6)
    assert np.all(ir[: int(0.012 * SR) - 1] == 0)          # predelay is silent


def test_reverb_tail_starts_at_word_and_outlasts_it():
    x = _voice()
    wet, at = vf.reverb_tail(x, SR, 0.8, 1.2, decay=1.0, level_db=-8)
    assert at == int(0.8 * SR)
    assert wet.shape[1] == 2
    assert len(wet) > int(0.4 * SR) + int(0.9 * SR)          # word + most of the decay
    peak_word = np.max(np.abs(x))
    assert np.max(np.abs(wet)) == pytest.approx(peak_word * 10 ** (-8 / 20), rel=1e-3)
    assert np.all(np.isfinite(wet))


def test_reverse_swell_ends_exactly_on_onset():
    x = _voice()
    wet, start = vf.reverse_swell(x, SR, 0.8, 1.2, decay=0.6, level_db=-9)
    assert start + len(wet) == int(0.8 * SR)                # lands on the word, not after it
    assert np.max(np.abs(wet[-3:])) < 0.05 * np.max(np.abs(wet))   # tiny fade into the word
    # it swells: the last quarter is louder than the first quarter
    q = len(wet) // 4
    assert np.sqrt(np.mean(wet[-q:] ** 2)) > np.sqrt(np.mean(wet[:q] ** 2))


def test_demo_cli_writes_both_files(tmp_path):
    src = tmp_path / "v.wav"
    sf.write(src, _voice(), SR)
    tail, swell = tmp_path / "tail.wav", tmp_path / "swell.wav"
    r = run_tool("voice_fx.py", src, 0.8, 1.2, tail, swell)
    assert r.returncode == 0, r.stderr
    for p in (tail, swell):
        y, sr = sf.read(p)
        assert sr == SR and y.ndim == 2 and np.max(np.abs(y)) <= 0.9
