import numpy as np
import pytest

import grade_match as gm


def _gradient_frames(n=3, h=72, w=128, tint=(0, 0, 0)):
    """Horizontal grey ramps with an optional per-channel offset, as uint8 RGB frames."""
    ramp = np.linspace(0, 255, w)[None, :, None].repeat(h, 0).repeat(3, 2)
    f = np.clip(ramp + np.array(tint, float), 0, 255).astype(np.uint8)
    return [f.copy() for _ in range(n)]


def test_luma_rec709_weights():
    px = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]], np.uint8)
    y = gm.luma(px)[0]
    assert y == pytest.approx([0.2126 * 255, 0.7152 * 255, 0.0722 * 255, 255], rel=1e-4)


def test_hsv_sat_grey_is_zero_and_pure_is_one():
    px = np.array([[[128, 128, 128], [255, 0, 0], [0, 0, 0]]], np.uint8)
    assert gm.hsv_sat(px)[0] == pytest.approx([0, 1, 0])


def test_levels_apply_identity():
    v = np.arange(256, dtype=float)
    assert np.allclose(gm._levels_apply(v, 0, 255, 1.0, 0, 255), v)


def test_fit_levels_identity():
    pct = {"p%d" % p: float(v) for p, v in zip(gm.PCTS, (5, 20, 70, 128, 180, 230, 250))}
    lv = gm.fit_levels(pct, pct)
    assert lv["gamma"] == pytest.approx(1.0, abs=0.05)
    assert lv["in_black"] < 6 and lv["in_white"] > 249
    assert lv["out_black"] < 6 and lv["out_white"] > 249


def test_fit_levels_recovers_lift():
    src = {"p%d" % p: float(v) for p, v in zip(gm.PCTS, (5, 20, 70, 128, 180, 230, 250))}
    dst = {k: 20 + v * (235 / 255) for k, v in src.items()}        # lifted blacks, softer whites
    lv = gm.fit_levels(src, dst)
    pred = gm._levels_apply(np.array(list(src.values())), lv["in_black"], lv["in_white"], lv["gamma"],
                            lv["out_black"], lv["out_white"])
    assert np.max(np.abs(pred - np.array(list(dst.values())))) < 4


def test_measure_frames_basic_metrics():
    m = gm.measure_frames(_gradient_frames(), palette=False)
    assert set(m["rgb_pct"]) == {"R", "G", "B"}
    assert m["sat_mean"] == pytest.approx(0, abs=0.01)            # pure grey ramp
    assert 100 < m["Y_pct"]["p50"] < 155
    assert m["Y_pct"]["p1"] < m["Y_pct"]["p50"] < m["Y_pct"]["p99"]


def test_measure_frames_sees_warm_cast():
    warm = gm.measure_frames(_gradient_frames(tint=(25, 0, -25)), palette=False)
    neutral = gm.measure_frames(_gradient_frames(), palette=False)
    assert warm["warmth_R_minus_B"] > neutral["warmth_R_minus_B"] + 20


def test_simulate_identity_levels_keeps_frames():
    frames = _gradient_frames(n=1)
    ident = {c: {"in_black": 0, "in_white": 255, "gamma": 1.0, "out_black": 0, "out_white": 255} for c in "RGB"}
    out = gm.simulate(frames, ident, 1.0)
    assert np.array_equal(out[0], frames[0])


def test_suggest_moves_render_towards_reference():
    ref_frames = _gradient_frames(tint=(20, 0, -20))
    our_frames = _gradient_frames()
    ref = gm.measure_frames(ref_frames, palette=False)
    ours = gm.measure_frames(our_frames, palette=False)
    lv, sat_scale, pred = gm.suggest(ref, ours, our_frames)
    assert pred is not None
    before = abs(ours["warmth_R_minus_B"] - ref["warmth_R_minus_B"])
    after = abs(pred["warmth_R_minus_B"] - ref["warmth_R_minus_B"])
    assert after < before / 2
