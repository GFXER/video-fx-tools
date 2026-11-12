"""Border-connected white background removal (keeps whites inside drawings, e.g. eyes).
usage: keyalpha.py in.png out.png          (ffmpeg from env FFMPEG or PATH)
       keyalpha.py FFMPEG in.png out.png   (explicit ffmpeg binary, original form)"""
import os, shutil, sys, subprocess, numpy as np
from scipy import ndimage
if len(sys.argv) >= 4:
    FF, src, dst = sys.argv[1:4]
elif len(sys.argv) == 3:
    FF = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
    src, dst = sys.argv[1:3]
else:
    raise SystemExit(__doc__)
probe = subprocess.run([FF, '-i', src], capture_output=True, text=True).stderr
import re; w, h = map(int, re.search(r', (\d{2,5})x(\d{2,5})', probe).groups())
raw = subprocess.run([FF, '-v', 'error', '-i', src, '-f', 'rawvideo', '-pix_fmt', 'rgba', '-'], capture_output=True).stdout
im = np.frombuffer(raw, np.uint8).reshape(h, w, 4).astype(np.float32)
rgb = im[..., :3]; mn = rgb.min(-1); mx = rgb.max(-1)
whiteish = (mn > 228) & (mx - mn < 26)
lab, n = ndimage.label(whiteish)
border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
bgm = np.isin(lab, list(border))
# grow 1px into the anti-aliased fringe, then feather
bgm = ndimage.binary_dilation(bgm, iterations=1) & (mn > 200)
alpha = 1.0 - ndimage.gaussian_filter(bgm.astype(np.float32), 1.0)
alpha = np.clip(alpha * 1.15, 0, 1)
# decontaminate white fringe: un-premultiply against white
a = np.maximum(alpha, 1e-3)[..., None]
rgb2 = np.clip((rgb - 255.0 * (1 - a)) / a, 0, 255)
rgb2 = np.where(alpha[..., None] > 0.98, rgb, rgb2)
out = np.dstack([rgb2, alpha * 255]).astype(np.uint8)
subprocess.run([FF, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{w}x{h}', '-i', '-', dst], input=out.tobytes())
print('ok', dst, 'bg%', round(bgm.mean() * 100, 1))
