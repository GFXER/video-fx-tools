import csv
import json

import pytest

import ramp_profile as rp
from conftest import run_tool

BASE = {"comp_in": 1.0, "comp_out": 3.0, "src_in": 0.5, "src_avail": 20.0, "src_fps": 50}


def shot(family, **kw):
    return {**BASE, "family": family, **kw}


def last_allowed(s):
    return s["src_in"] + s["src_avail"] - 1 / s["src_fps"]


@pytest.mark.parametrize("family", ["apex", "beat", "step"])
@pytest.mark.parametrize("fps", [24, 30, 60])
def test_one_value_per_frame_and_source_never_runs_backwards(family, fps):
    r = rp.ramp_profile(shot(family), fps)
    assert len(r["times"]) == len(r["src"]) == len(r["speed"]) == 2 * fps
    assert r["times"][0] == 1.0 and r["src"][0] == 0.5
    assert all(b >= a for a, b in zip(r["src"], r["src"][1:]))
    assert r["qc"]["short_source"] is False


@pytest.mark.parametrize("family", ["apex", "beat", "step"])
@pytest.mark.parametrize("avail", [0.02, 0.1, 0.3, 0.8, 1.5])
def test_never_reads_past_the_source(family, avail):
    s = shot(family, src_avail=avail)
    assert max(rp.ramp_profile(s, 30)["src"]) <= last_allowed(s) + 1e-3


def test_squeezed_ramp_keeps_the_slow_part():
    r = rp.ramp_profile(shot("beat", src_avail=1.0), 30)
    assert r["qc"]["min_speed"] == pytest.approx(0.2)
    assert r["qc"]["max_speed"] < 4.0
    assert r["qc"]["short_source"] is False


def test_short_source_is_flagged():
    r = rp.ramp_profile(shot("beat", src_avail=0.1), 30)       # the 0.2x part alone needs about 0.4 s
    assert r["qc"]["short_source"] is True
    assert r["qc"]["min_speed"] < 0.2


def test_apex_shape():
    sp = rp.ramp_profile(shot("apex"), 24)["speed"]
    assert sp[:6] == [1.0] * 6
    assert sp[20] == pytest.approx(0.03)
    assert sp[-1] == pytest.approx(2.5)


def test_apex_shot_shorter_than_its_parts():
    sp = rp.ramp_profile(shot("apex", comp_out=1.5), 24)["speed"]      # 12 frames for 6 + 8 + 5
    assert len(sp) == 12
    assert sp[0] == 1.0 and sp[-1] == pytest.approx(2.5)


def test_step_holds_source_frames():
    src = rp.ramp_profile(shot("step", step_fps=8), 24)["src"]
    assert src[0] == src[1] == src[2] != src[3]


def test_blend_follows_the_slowest_part():
    assert rp.ramp_profile(shot("step", src_fps=60), 30)["blend"] == "none"
    assert rp.ramp_profile(shot("beat", src_fps=50), 24)["blend"] == "pixel"
    assert rp.ramp_profile(shot("apex"), 24)["blend"] == "retime"


def test_bad_input():
    with pytest.raises(ValueError):
        rp.ramp_profile(shot("nope"), 30)
    with pytest.raises(ValueError):
        rp.ramp_profile({"family": "beat"}, 30)
    with pytest.raises(ValueError):
        rp.ramp_profile(shot("beat", comp_out=1.0), 30)


def test_cli_csv_and_json(tmp_path):
    out_csv, out_json = tmp_path / "r.csv", tmp_path / "r.json"
    p = run_tool("ramp_profile.py", json.dumps(shot("beat")), 24, "--csv", out_csv, "--json", out_json)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["frames"] == 48
    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == 48 and rows[0]["src"] == "0.5"
    assert len(json.loads(out_json.read_text())["src"]) == 48
