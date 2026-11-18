import json

import numpy as np
import pytest
import soundfile as sf

from conftest import run_tool

SR = 48000


def _kick_track(bpm=120.0, phase=0.25, dur=24.0):
    """Synthetic four-on-the-floor: a decaying 60 Hz kick on every beat + quiet noise floor."""
    n = int(dur * SR)
    y = 0.002 * np.random.default_rng(3).standard_normal(n)
    k = np.arange(int(0.18 * SR)) / SR
    kick = 0.8 * np.sin(2 * np.pi * 60 * k) * np.exp(-k / 0.05)
    beat = 60.0 / bpm
    t = phase
    while t + 0.2 < dur:
        a = int(t * SR)
        y[a:a + len(kick)] += kick
        t += beat
    return y


@pytest.mark.parametrize("bpm,phase", [(120.0, 0.25), (95.0, 0.40)])
def test_beat_map_finds_tempo_and_phase(tmp_path, bpm, phase):
    wav = tmp_path / "track.wav"
    sf.write(wav, _kick_track(bpm, phase), SR)
    r = run_tool("beat_map.py", wav)
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "track.beat.json").read_text())
    assert out["bpm"] == pytest.approx(bpm, abs=0.5)
    beat = 60.0 / bpm
    d = (out["phase"] - phase) % beat
    assert min(d, beat - d) < 0.02                          # phase within 20 ms
    # fit = share of detected hits on the grid; the synthetic noise floor still adds a few stray hits
    assert out["fit"] > 0.75
    assert out["sr"] == SR
    assert len(out["bars"]) == len(out["low"]) > 5
