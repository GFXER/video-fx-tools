import numpy as np
import pytest

from conftest import run_tool

Image = pytest.importorskip("PIL.Image")


def test_keeps_inner_whites_removes_outer_white(tmp_path, ffmpeg):
    # white canvas, black ring, white "eye" inside the ring
    h = w = 96
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    img = np.full((h, w, 3), 255, np.uint8)
    img[(r > 18) & (r < 30)] = 0
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    Image.fromarray(img).save(src)

    res = run_tool("keyalpha.py", src, dst, env={"FFMPEG": ffmpeg})
    assert res.returncode == 0, res.stderr
    out = np.array(Image.open(dst).convert("RGBA"))
    assert out.shape == (h, w, 4)
    a = out[..., 3]
    assert a[2, 2] == 0 and a[-3, -3] == 0                  # border-connected white is keyed out
    assert a[h // 2, w // 2] == 255                         # the enclosed white stays opaque
    assert a[h // 2, w // 2 + 24] == 255                    # the ring itself stays opaque
    assert tuple(out[h // 2, w // 2, :3]) == (255, 255, 255)
