"""
Training-data trajectories: the noise -> data paths the model is trained on, z_t = t x + (1 - t) s e (t = data fraction,
0 = pure noise, as train_jit.py; s = noise_scale, JiT's img_size / 256 by default), for a few real training images per
class with one fixed noise draw each. No model involved.

Writes to {out}/:
  grid.gif              every sample at once (rows = classes, columns = samples), t sweeping 0 -> 1, with the t of each frame
  {class}_{i}.gif       one GIF per sample
  strip.png             each sample at strip_columns noise levels (columns), labelled with t and the share of training draws
                        at that t or noisier under JiT's logit-normal t sampling (P_mean, P_std)

Both the GIF frames and the strip columns are evenly spaced under t_spacing: 'uniform' = even steps in t; 'training' =
even steps in the training distribution's CDF (quantiles), so every frame / column stands for the same share of training
draws and the visualization lingers where training spends its samples. Either way the first and last are t = 0 (noise)
and t = 1 (data); the levels between are the midpoint quantiles (i + 0.5) / (n - 2).

$ python viz_trajectories.py --data_dir=/data/in1k_128
$ python viz_trajectories.py --data_dir=/data/in1k_128 --classes=1,207,388,980 --per_class=3 --t_spacing=training
"""

import os
import math

import numpy as np
from PIL import Image, ImageDraw

from data_jit import MemmapImages

# -----------------------------------------------------------------------------
data_dir = '' # prepared dataset (prepare_jit_data.py)
split = 'train' # 'train' or 'val'
classes = '' # comma-separated class indices; '' = 4 evenly spaced classes
per_class = 2 # images per class
noise_scale = 0.0 # 0 -> img_size / 256 (JiT's rule; 0.5 at 128px)
P_mean = -0.8 # training t distribution (logit-normal), for the strip's annotation and t_spacing='training'
P_std = 0.8
frames = 48 # GIF frames from t = 0 to t = 1
t_spacing = 'uniform' # 'uniform' or 'training'
strip_columns = 11 # noise levels in strip.png (evenly spaced under t_spacing, including t = 0 and 1)
fps = 12
hold = 1.0 # seconds to hold the first (noise) and last (data) frames
upscale = 2 # nearest-neighbour upscaling of the GIFs for viewing
seed = 0
out = 'trajectories'
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

assert data_dir, 'pass --data_dir'
assert t_spacing in ('uniform', 'training'), t_spacing
data = MemmapImages(data_dir if split == 'train' else os.path.join(data_dir, 'val'))
R = data.img_size
s = noise_scale or R / 256
cls = [int(c) for c in classes.split(',') if c.strip()] if classes else \
    np.linspace(0, data.num_classes - 1, 4).round().astype(int).tolist()
try:
    names = data.class_names()
except Exception: # (class names need torchvision's ImageNet list)
    names = [str(i) for i in range(data.num_classes)]

# a few images per class, spread over the class (deterministic), one fixed noise draw each
rng = np.random.default_rng(seed)
samples = [] # (class, name, x [R, R, 3] in [-1, 1], e)
for c in cls:
    idx = np.flatnonzero(data.labels == c)
    assert len(idx), f'class {c} has no images in {split}'
    for j in np.linspace(0, len(idx) - 1, per_class).round().astype(int):
        x = data.images[idx[j]].astype(np.float32) / 127.5 - 1
        samples.append((c, names[c], x, rng.standard_normal(x.shape).astype(np.float32)))

def normal_cdf(v):
    return 0.5 * (1 + math.erf(v / math.sqrt(2)))

def train_cdf(t):
    # share of training draws with data fraction <= t under logit-normal t sampling
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    return normal_cdf((math.log(t / (1 - t)) - P_mean) / P_std)

def train_quantile(q):
    # inverse of train_cdf (bisection on logit t)
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if normal_cdf((mid - P_mean) / P_std) < q:
            lo = mid
        else:
            hi = mid
    return 1 / (1 + math.exp(-(lo + hi) / 2))

def frame(x, e, t):
    z = t * x + (1 - t) * s * e
    return (np.clip((z + 1) / 2, 0, 1) * 255).round().astype(np.uint8)

def big(a):
    return np.kron(a, np.ones((upscale, upscale, 1), dtype=np.uint8)) if upscale > 1 else a

def spaced_ts(n):
    # n noise levels evenly spaced under t_spacing, from t = 0 to t = 1
    if t_spacing == 'uniform':
        return np.linspace(0, 1, n).tolist()
    return [0.0] + [train_quantile((i + 0.5) / (n - 2)) for i in range(n - 2)] + [1.0] # midpoint quantiles + the exact ends

ts = spaced_ts(frames)
durations = [int(1000 / fps)] * len(ts)
durations[0] = durations[-1] = int(1000 * hold)

os.makedirs(out, exist_ok=True)
pad, label_h = 4, 16
def save_gif(path, imgs):
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=durations, loop=0)

# grid GIF: rows = classes, columns = samples per class, a header line with t
rows, cols = len(cls), per_class
cell = R * upscale
W, H = cols * (cell + pad) + pad, label_h + rows * (cell + label_h + pad) + pad
grid_frames = []
for t in ts:
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 2), f't = {t:.3f}  (data fraction; training draws below: {100 * train_cdf(t):.0f}%)  noise scale {s:g}', fill=(0, 0, 0))
    for k, (c, name, x, e) in enumerate(samples):
        r, q = divmod(k, cols)
        x0, y0 = pad + q * (cell + pad), label_h + r * (cell + label_h + pad)
        if q == 0:
            d.text((x0, y0 + 1), f'{c}: {name}'[:48], fill=(0, 0, 0))
        canvas.paste(Image.fromarray(big(frame(x, e, t))), (x0, y0 + label_h))
    grid_frames.append(canvas)
save_gif(os.path.join(out, 'grid.gif'), grid_frames)

# one GIF per sample
for k, (c, name, x, e) in enumerate(samples):
    i = k % per_class
    safe = ''.join(ch if ch.isalnum() else '-' for ch in name)[:32]
    save_gif(os.path.join(out, f'{c:04d}-{safe}_{i}.gif'), [Image.fromarray(big(frame(x, e, t))) for t in ts])

# strip: samples x fixed t, annotated with the training CDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
st = spaced_ts(strip_columns)
fig, axes = plt.subplots(len(samples), len(st), figsize=(1.3 * len(st), 1.3 * len(samples) + 0.6), squeeze=False)
for k, (c, name, x, e) in enumerate(samples):
    for j, t in enumerate(st):
        ax = axes[k, j]
        ax.imshow(frame(x, e, t), interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if k == 0:
            ax.set_title(f't={t:.3f}\n{100 * train_cdf(t):.1f}% below', fontsize=7)
        if j == 0:
            ax.set_ylabel(name[:18], fontsize=7)
fig.suptitle(f'training trajectories z_t = t x + (1 - t) {s:g} e, columns evenly spaced in ' +
             ('t' if t_spacing == 'uniform' else f'training quantile') +
             f'  (t = data fraction; "% below": share of training draws at that t or noisier, logit-normal({P_mean}, {P_std}))', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, 'strip.png'), dpi=150)
print(f'wrote {out}/grid.gif ({len(samples)} samples, {len(ts)} frames, t spacing {t_spacing}), {len(samples)} per-sample GIFs '
      f'and {out}/strip.png (noise scale {s:g}, classes {cls})')
