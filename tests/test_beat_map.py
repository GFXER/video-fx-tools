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


def test_make_grid_labels_and_spacing():
    from beat_map import make_grid
    g = make_grid(0.5, 0.1, 4.2, sub=2)                   # 120 BPM, 8ths, 2 bars + a bit
    assert [x["kind"] for x in g[:4]] == ["downbeat", "8th", "beat", "8th"]
    assert g[0]["t"] == pytest.approx(0.1) and g[1]["t"] == pytest.approx(0.35)
    assert (g[8]["bar"], g[8]["beat"], g[8]["sub"]) == (2, 1, 1)
    assert g[-1]["t"] < 4.2
    k16 = [x["kind"] for x in make_grid(0.5, 0.0, 0.5, sub=4)]
    assert k16 == ["downbeat", "16th", "8th", "16th"]


@pytest.mark.parametrize("t,fps,tc", [(0, 30, "00:00:00:00"), (1.5, 30, "00:00:01:15"),
                                      (61.04, 25, "00:01:01:01"), (3600.5, 50, "01:00:00:25"),
                                      (1.0, 29.97, "00:00:01:00")])
def test_timecode(t, fps, tc):
    from beat_map import timecode
    assert timecode(t, fps) == tc


def test_beat_map_grid_and_csv(tmp_path):
    import csv
    wav = tmp_path / "track.wav"
    sf.write(wav, _kick_track(120.0, 0.25, 12.0), SR)
    mk = tmp_path / "markers.csv"
    r = run_tool("beat_map.py", wav, "--grid", "16th", "--csv", mk, "--fps", "25")
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "track.beat.json").read_text())
    g = out["grid"]
    assert g[0]["kind"] == "downbeat" and g[0]["t"] == pytest.approx(out["phase"])
    assert len(g) == pytest.approx((out["dur"] - out["phase"]) / (out["beat"] / 4), abs=1.01)
    rows = list(csv.reader(mk.open()))
    assert rows[0] == ["Marker Name", "Description", "In", "Out", "Duration", "Marker Type"]
    assert len(rows) - 1 == len(g)
    assert rows[1][0] == "Bar 1" and rows[2][1] == "16th"
    assert all(len(x[2].split(":")) == 4 for x in rows[1:])


def test_beat_map_without_flags_keeps_old_json(tmp_path):
    wav = tmp_path / "track.wav"
    sf.write(wav, _kick_track(120.0, 0.25, 12.0), SR)
    assert run_tool("beat_map.py", wav).returncode == 0
    assert "grid" not in json.loads((tmp_path / "track.beat.json").read_text())
