import json

import numpy as np
from PIL import Image

from conftest import run_tool
import dupe_find


def _photo(path, seed, size=(640, 480), rotate_exif=False):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (12, 16, 3), dtype=np.uint8)          # coarse structure, upscaled = photo-like gradients
    im = Image.fromarray(base).resize(size, Image.BICUBIC)
    if rotate_exif:                                                    # stored sideways + EXIF says rotate back
        ex = Image.Exif(); ex[0x0112] = 6
        im.rotate(90, expand=True).save(path, exif=ex, quality=92)
    else:
        im.save(path, quality=92)
    return im


def test_resized_and_reencoded_copy_matches(tmp_path):
    a = _photo(tmp_path / 'a.jpg', 1)
    a.resize((320, 240)).save(tmp_path / 'a_small.jpg', quality=60)
    _photo(tmp_path / 'b.jpg', 2)
    ha, hs, hb = (dupe_find.dhash(tmp_path / f) for f in ('a.jpg', 'a_small.jpg', 'b.jpg'))
    assert dupe_find.ham(ha, hs) <= 6
    assert dupe_find.ham(ha, hb) > 16


def test_exif_rotation_is_applied(tmp_path):
    _photo(tmp_path / 'up.jpg', 3)
    _photo(tmp_path / 'side.jpg', 3, rotate_exif=True)
    assert dupe_find.ham(dupe_find.dhash(tmp_path / 'up.jpg'), dupe_find.dhash(tmp_path / 'side.jpg')) <= 6


def test_cli_unused_and_groups(tmp_path):
    arch, used = tmp_path / 'arch', tmp_path / 'used'
    arch.mkdir(); used.mkdir()
    for s in range(4):
        _photo(arch / f'p{s}.jpg', 10 + s)
    Image.open(arch / 'p0.jpg').resize((400, 300)).save(used / 'placed.jpg')         # p0 is already used
    Image.open(arch / 'p1.jpg').save(arch / 'p1_copy.png')                           # p1 twice in the archive
    r = run_tool('dupe_find.py', 'unused', arch, '--used', used, '--json', tmp_path / 'u.json')
    assert r.returncode == 0, r.stderr
    names = sorted(p.split('/')[-1] for p in json.load(open(tmp_path / 'u.json')))
    assert names == ['p1.jpg', 'p2.jpg', 'p3.jpg']                                   # p0 used, p1 copy collapsed
    g = run_tool('dupe_find.py', 'groups', arch)
    assert 'p1.jpg' in g.stdout and 'p1_copy.png' in g.stdout and '1 duplicate groups' in g.stderr
