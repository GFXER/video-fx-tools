#!/usr/bin/env python3
"""dupe_find.py: find near-duplicate photos (re-exports, resized copies, light edits) with a 64-bit difference hash.

Typical use: you have an archive folder and a list of photos already placed in a layout, and you want the
archive photos that are NOT already used (and not used twice under different names).

  python3 dupe_find.py groups ARCHIVE_DIR                     # print groups of near-duplicates
  python3 dupe_find.py unused ARCHIVE_DIR --used USED_DIR     # archive photos that match nothing in USED_DIR
  python3 dupe_find.py unused ARCHIVE_DIR --used USED_DIR --json out.json

--dist N   Hamming distance (0-64) that still counts as the same photo (default 8 for 'groups', 10 against --used).
EXIF rotation is applied before hashing, so a rotated phone shot still matches its upright export.
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageOps

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.bmp')


def dhash(path, size=8):
    """64-bit difference hash: grayscale, (size+1) x size, compare each pixel with its right neighbour."""
    im = Image.open(path)
    im.draft('L', (size * 8, size * 8))          # fast JPEG decode at reduced size
    im = ImageOps.exif_transpose(im).convert('L').resize((size + 1, size), Image.LANCZOS)
    px = list(im.tobytes())
    h = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            h = (h << 1) | (1 if row[c] > row[c + 1] else 0)
    return h


def ham(a, b):
    return bin(a ^ b).count('1')


def scan(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(EXTS) and not f.startswith('.'):
                p = os.path.join(root, f)
                try:
                    out.append((p, dhash(p)))
                except Exception as e:                      # unreadable / truncated file
                    print(f'skip {p}: {e}', file=sys.stderr)
    return out


def groups(items, dist):
    """single-link groups of photos within `dist` of each other; returns lists with 2+ members"""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if ham(items[i][1], items[j][1]) <= dist:
                parent[find(i)] = find(j)
    g = {}
    for i, (p, _) in enumerate(items):
        g.setdefault(find(i), []).append(p)
    return [v for v in g.values() if len(v) > 1]


def unused(items, used, dist, self_dist=6):
    """archive items that match nothing in `used`, keeping one photo per near-duplicate group"""
    keep, seen = [], []
    for p, h in items:
        if any(ham(h, u) <= dist for _, u in used):
            continue
        if any(ham(h, s) <= self_dist for s in seen):
            continue
        seen.append(h)
        keep.append(p)
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['groups', 'unused'])
    ap.add_argument('archive')
    ap.add_argument('--used', help='folder of photos already used (mode unused)')
    ap.add_argument('--dist', type=int, default=None)
    ap.add_argument('--json', help='write the result as JSON')
    a = ap.parse_args(argv)
    items = scan(a.archive)
    if a.mode == 'groups':
        res = groups(items, 8 if a.dist is None else a.dist)
        for k, g in enumerate(res, 1):
            print(f'group {k}:'); [print('  ' + p) for p in g]
        print(f'{len(items)} photos, {len(res)} duplicate groups', file=sys.stderr)
    else:
        if not a.used:
            ap.error('mode unused needs --used')
        res = unused(items, scan(a.used), 10 if a.dist is None else a.dist)
        print('\n'.join(res))
        print(f'{len(items)} photos, {len(res)} unused', file=sys.stderr)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
