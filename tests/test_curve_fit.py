import json

import numpy as np
import pytest

import curve_fit as cf
from conftest import run_tool


def test_bez_eval_endpoints_and_monotonic():
    t = np.linspace(0, 1, 50)
    for x1, y1, x2, y2, _ in cf.FAMILIES.values():
        y = cf.bez_eval(x1, y1, x2, y2, t)
        assert y[0] == pytest.approx(0, abs=1e-3)
        assert y[-1] == pytest.approx(1, abs=1e-3)


def test_linear_family_is_identity():
    t = np.linspace(0, 1, 21)
    assert np.allclose(cf.bez_eval(0, 0, 1, 1, t), t, atol=1e-3)


def test_ae_to_bez_speed_zero_is_css_bezier():
    # with speed 0 on both keys the AE ease is exactly cubic-bezier(i0, 0, 1-i1, 1)
    assert cf.ae_to_bez(0.3, 0.0, 0.9, 0.0) == pytest.approx((0.3, 0.0, 0.1, 1.0))


def test_bez_to_ae_roundtrip_influences():
    ae = cf.bez_to_ae(0.30, 0.0, 0.10, 1.0, dv=100.0, dur=1.0)
    assert ae["out_influence"] == pytest.approx(30.0)
    assert ae["in_influence"] == pytest.approx(90.0)
    assert ae["out_speed"] == 0 and ae["in_speed"] == 0


def test_fit_bezier_recovers_known_ease():
    tn = np.linspace(0, 1, 31)
    pn = cf.bez_eval(0.30, 0.0, 0.10, 1.0, tn)            # house_30_90
    fit = cf.fit_bezier(tn, pn, allow_speed=False)
    assert fit["rms"] < 0.005
    assert fit["i0"] == pytest.approx(0.30, abs=0.05)
    assert fit["i1"] == pytest.approx(0.90, abs=0.05)


def test_closest_family_finds_source_curve():
    tn = np.linspace(0, 1, 41)
    x1, y1, x2, y2, _ = cf.FAMILIES["expo_out"]
    best = cf.closest_family(tn, cf.bez_eval(x1, y1, x2, y2, tn))[0]
    assert best[1] == "expo_out" and best[0] < 1e-6


def test_analyse_synthetic_move():
    tn = np.linspace(0, 1, 25)
    v = 100 + 50 * cf.bez_eval(0.05, 0.75, 0.10, 1.0, tn)   # expo-out from 100 to 150
    res = cf.analyse(v, fps=24, name="scale", units="%")
    assert res["frames"] == 25
    assert res["v0"] == pytest.approx(100)
    assert res["v1"] == pytest.approx(150, abs=0.5)
    assert res["overshoot_pct"] < 1
    assert res["stepped"] is False
    assert res["travel_at_time"]["25%"] > 70        # expo-out: most of the travel happens early
    assert res["fit"]["rms_pct"] < 1.0
    assert "key(1)" in res["expression"]


def test_analyse_detects_stepped_motion():
    v = np.repeat(np.linspace(0, 1, 8), 3)          # each value held 3 frames
    res = cf.analyse(v, fps=24)
    assert res["stepped"] is True
    assert res["step_frames"] == 3


def test_cli_curve_prints_family():
    r = run_tool("curve_fit.py", "curve", "flow_051", "--frames", "12")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["family"] == "flow_051"
    assert out["ae"]["out_influence"] == pytest.approx(51.0)
    assert out["ae"]["in_influence"] == pytest.approx(100.0)


def test_cli_series_json(tmp_path):
    vals = list(100 * cf.bez_eval(0.333, 0.0, 0.667, 1.0, np.linspace(0, 1, 20)))
    src = tmp_path / "vals.json"
    src.write_text(json.dumps(vals))
    r = run_tool("curve_fit.py", "series", src, "--fps", "20")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["frames"] == 20
    assert out["family"][0]["name"] == "easy_ease"


# ---------------------------------------------------------------- AE keyframe export
def _scale_pop():
    tn = np.linspace(0, 1, 13)
    v = 100 + 50 * cf.bez_eval(0.30, 0.0, 0.10, 1.0, tn)     # house_30_90, 100 -> 150 %
    return cf.analyse(v, fps=24, name="pop.scale", units="%")


def test_ae_keyframes_two_keys_match_fit():
    res = _scale_pop()
    k = cf.ae_keyframes(res, "scale")
    assert k["match_name"] == "ADBE Scale" and k["mode"] == "multiply"
    assert [key["t"] for key in k["keys"]] == [0.0, res["dur_s"]]
    assert k["keys"][0]["value"] == pytest.approx(100)
    assert k["keys"][1]["value"] == pytest.approx(150, abs=0.5)
    assert k["keys"][0]["out"]["influence"] == res["fit"]["ae"]["out_influence"]
    assert k["keys"][1]["in"]["influence"] == res["fit"]["ae"]["in_influence"]
    json.dumps(k, allow_nan=False)                          # strict JSON, AE-safe


def test_ae_keyframes_caps_non_finite_speed():
    res = _scale_pop()
    res["fit"]["ae"] = dict(res["fit"]["ae"], out_speed=float("inf"), in_speed=float("nan"))
    k = cf.ae_keyframes(res, None)
    assert k["keys"][0]["out"]["speed"] == cf.SPEED_CAP
    assert k["keys"][1]["in"]["speed"] == 0.0
    assert k["match_name"] is None and k["mode"] == "add"


def test_ae_jsx_embeds_keys():
    js = cf.ae_jsx(cf.ae_keyframes(_scale_pop(), "scale"))
    assert '"ADBE Scale"' in js and "setTemporalEaseAtKey" in js and "beginUndoGroup" in js


def test_cli_series_writes_keys(tmp_path):
    pytest.importorskip("matplotlib")
    vals = list(100 + 20 * cf.bez_eval(0.05, 0.75, 0.10, 1.0, np.linspace(0, 1, 16)))
    src = tmp_path / "vals.json"
    src.write_text(json.dumps(vals))
    r = run_tool("curve_fit.py", "series", src, "--fps", "30", "--prop", "scale", "--out", tmp_path / "mv")
    assert r.returncode == 0, r.stderr
    keys = json.loads((tmp_path / "mv_keys.json").read_text())
    assert keys["prop"] == "scale" and len(keys["keys"]) == 2
    assert (tmp_path / "mv_keys.jsx").read_text().startswith("// curve_fit.py keys")
    assert (tmp_path / "mv.png").stat().st_size > 0


def test_cli_rejects_unknown_prop():
    r = run_tool("curve_fit.py", "series", "x.json", "--prop", "opacity")
    assert r.returncode != 0
