"""Shared fixtures. Every test builds its own synthetic input (no media files in the repo)."""
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _find_ffmpeg():
    ff = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not ff:
        return None
    try:
        subprocess.run([ff, "-version"], capture_output=True, check=True, timeout=20)
    except Exception:
        return None
    return ff


@pytest.fixture(scope="session")
def ffmpeg():
    """Path to a working ffmpeg, or skip the test (tools read FFMPEG / PATH the same way)."""
    ff = _find_ffmpeg()
    if not ff:
        pytest.skip("ffmpeg not found (set FFMPEG or put ffmpeg on PATH)")
    return ff


def run_tool(script, *args, env=None, timeout=300):
    """Run one of the repo's CLI scripts with the current interpreter; returns CompletedProcess."""
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run([sys.executable, os.path.join(ROOT, script), *map(str, args)],
                          capture_output=True, text=True, env=e, timeout=timeout)
