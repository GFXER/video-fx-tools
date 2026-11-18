import re
import subprocess

import pytest

import datamosh as dm
from conftest import run_tool

M = b"\x00\x00\x01\xb6"


def _vop(kind, payload=b"xx"):
    # VOP header byte: top 2 bits = coding type (0 I, 1 P, 2 B, 3 S)
    return M + bytes([kind << 6]) + payload


def test_split_vops_and_types():
    header = b"\x00\x00\x01\xb0HEADER"
    vops = [_vop(0), _vop(1, b"abc"), _vop(1), _vop(0, b"z"), _vop(2)]
    h, out = dm.split_vops(header + b"".join(vops))
    assert h == header
    assert out == vops
    assert [dm.vop_type(v) for v in out] == [0, 1, 1, 0, 2]


def test_split_vops_rejects_non_mpeg4():
    with pytest.raises(SystemExit):
        dm.split_vops(b"not a video stream")


def _testclip(ff, path, src):
    subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", f"{src}=size=160x284:rate=12:duration=1",
                    "-pix_fmt", "yuv420p", path], check=True)


def _frames(ff, path):
    r = subprocess.run([ff, "-v", "error", "-i", path, "-map", "0:v:0", "-f", "null", "-",
                        "-progress", "pipe:1"], capture_output=True, text=True)
    vals = [l.split("=")[1] for l in r.stdout.splitlines() if l.startswith("frame=")]
    return int(vals[-1]) if vals else 0


def test_end_to_end_mosh(tmp_path, ffmpeg):
    a, b, out = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "mosh.mp4"
    _testclip(ffmpeg, a, "testsrc2")
    _testclip(ffmpeg, b, "smptebars")
    r = run_tool("datamosh.py", "--a", a, "--b", b, "--out", out, "--fps", 12, "--w", 160, "--h", 284,
                 "--bloom", "14:3", env={"FFMPEG": ffmpeg})
    assert r.returncode == 0, r.stderr
    assert "dropped I=1" in r.stdout                         # the cut I-frame is gone
    assert out.exists() and out.stat().st_size > 0
    # the tool reports VOPs - dropped I + bloom repeats; the final encode must keep every one of them
    m = re.search(r"VOPs (\d+) .*dropped I=(\d+), out frames (\d+)", r.stdout)
    assert m, r.stdout
    vops, dropped, n_out = map(int, m.groups())
    assert n_out == vops - dropped + 3                        # --bloom 14:3 repeats one P-frame 3 times
    assert abs(_frames(ffmpeg, str(out)) - n_out) <= 1
