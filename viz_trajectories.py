"""
Training-data trajectories: the noise -> data paths the model is trained on, z_t = t x + (1 - t) s e (t = data fraction,
0 = pure noise, as train_jit.py; s = noise_scale, JiT's img_size / 256 by default), for a few real training images per
class with one fixed noise draw each. No model involved.

Writes to {out}/ (default trajectories/), file names carrying the settings, e.g. grid_logitnorm-0.8_0.8_noise0.5_c4x2_f48.gif:
  grid_{settings}.gif   every sample at once (rows = classes, columns = samples), t sweeping 0 -> 1, with the t of each frame
  strip_{settings}.png  each sample at strip_columns noise levels (columns), labelled with t
settings: t_spacing, noise scale, classes (explicit list, else count) x images per class, split / seed when not the
default, and the frame count (GIF) or column count (strip).

t_spacing is a distribution over t in [0, 1]; the GIF frames and strip columns sit at its quantiles i / (n - 1), i.e.
evenly spaced in its CDF (the first is t = 0, pure noise, the last t = 1, data). Supported:
  uniform              even steps in t
  logitnorm(m, s)      t = sigmoid(N(m, s)); JiT trains with logitnorm(-0.8, 0.8)
  beta(a, b)           Beta(a, b)
  splitlogitnorm(m, sl, sr)  two-piece logit-normal: std sl on the noisy side of the mode, sr on the clean side
  spectral             even steps in how much of the image is visible, from the dataset's own power spectrum: frequency f
                       (cycles / image) emerges from the noise when its SNR t^2 P(f) / ((1 - t)^2 s^2) reaches 1, at
                       t = s / (s + sqrt(P(f))); the levels put that cutoff at log-spaced frequencies from 1 cycle per image
                       to Nyquist (one octave of detail per step on a power-law spectrum). P(f): radially averaged
                       |FFT|^2 (orthonormal, per channel) of spectrum_images training images in [-1, 1]
Quantiles come from the density's CDF integrated numerically on a fine logit-spaced grid, so other densities are one
line in DENSITIES.

$ python viz_trajectories.py --data_dir=/data/in1k_128
$ python viz_trajectories.py --data_dir=/data/in1k_128 --classes=1,207,388,980 --per_class=3 --t_spacing='logitnorm(-0.8, 0.8)'
  -> trajectories/grid_logitnorm-0.8_0.8_noise0.5_c1-207-388-980x3_f48.gif, trajectories/strip_logitnorm-0.8_0.8_noise0.5_c1-207-388-980x3_n11.png
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
frames = 48 # GIF frames from t = 0 to t = 1
t_spacing = 'uniform' # distribution whose quantiles place the frames / columns: 'uniform', 'logitnorm(m, s)', 'beta(a, b)'
strip_columns = 11 # noise levels in strip.png (quantiles of t_spacing, including t = 0 and 1)
spectrum_images = 2000 # t_spacing='spectral': images for the power spectrum
fps = 12
hold = 1.0 # seconds to hold the first (noise) and last (data) frames
upscale = 2 # nearest-neighbour upscaling of the GIFs for viewing
seed = 0
out = 'trajectories' # output directory
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

assert data_dir, 'pass --data_dir'
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

# densities over t in (0, 1), by name; parameters parsed from 'name(p1, p2, ...)'
DENSITIES = {
    'uniform': lambda t: np.ones_like(t),
    'logitnorm': lambda t, m, sd: np.exp(-0.5 * ((np.log(t / (1 - t)) - m) / sd) ** 2) / (sd * math.sqrt(2 * math.pi) * t * (1 - t)),
    'beta': lambda t, a, b: t ** (a - 1) * (1 - t) ** (b - 1),
    # two-piece logit-normal: std sl left of the mode m (noisier), sr right of it (cleaner); train_jit.py --P_std / --P_std_right
    'splitlogitnorm': lambda t, m, sl, sr: np.exp(-0.5 * ((np.log(t / (1 - t)) - m) / np.where(np.log(t / (1 - t)) <= m, sl, sr)) ** 2) / (t * (1 - t)),
}

def parse_spacing(spec):
    spec = spec.replace(' ', '')
    name, _, rest = spec.partition('(')
    assert name in DENSITIES or name == 'spectral', f"t_spacing: unknown distribution {name!r} (supported: uniform, logitnorm(m, s), splitlogitnorm(m, sl, sr), beta(a, b), spectral)"
    params = [float(v) for v in rest.rstrip(')').split(',') if v] if rest else []
    return name, params

def quantiles(n):
    # t at the CDF levels i / (n - 1), from the density integrated on a logit-spaced grid (dense near both ends)
    name, params = parse_spacing(t_spacing)
    if name == 'uniform':
        return np.linspace(0, 1, n).tolist()
    if name == 'spectral':
        r, P = power_spectrum()
        t_r = np.maximum.accumulate(s / (s + np.sqrt(P))) # emergence level per radius (monotone: power falls with frequency)
        f = np.exp(np.linspace(0, np.log(r[-1]), n - 2)) # cutoffs log-spaced from 1 cycle / image to Nyquist
        return [0.0] + np.interp(np.log(f), np.log(r), t_r).tolist() + [1.0]
    t = 1 / (1 + np.exp(-np.linspace(-30, 30, 400001)))
    t = t[(t > 0) & (t < 1)]
    pdf = DENSITIES[name](t, *params)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(t))])
    cdf /= cdf[-1]
    q = np.linspace(0, 1, n)
    out = np.interp(q, cdf, t)
    out[0], out[-1] = 0.0, 1.0 # the exact ends: pure noise and data
    return out.tolist()

_spectrum = None
def power_spectrum():
    # radially averaged power per frequency radius 1 .. R/2 (cycles / image), orthonormal FFT per channel, images in [-1, 1]
    global _spectrum
    if _spectrum is None:
        idx = np.linspace(0, len(data) - 1, min(spectrum_images, len(data))).round().astype(int)
        fy, fx = np.meshgrid(np.fft.fftfreq(R) * R, np.fft.fftfreq(R) * R, indexing='ij')
        rad = np.sqrt(fx ** 2 + fy ** 2).round().astype(int)
        acc = np.zeros((R, R))
        for i in range(0, len(idx), 100):
            x = data.images[np.sort(idx[i:i + 100])].astype(np.float32) / 127.5 - 1 # [B, R, R, 3]
            acc += (np.abs(np.fft.fft2(x, axes=(1, 2), norm='ortho')) ** 2).mean(-1).sum(0)
        acc /= len(idx)
        r = np.arange(1, R // 2 + 1)
        P = np.array([acc[rad == k].mean() for k in r])
        _spectrum = (r, P)
        print(f'power spectrum of {len(idx)} images: P(1) {P[0]:.3g}, P({R // 2}) {P[-1]:.3g}, '
              f'slope {np.polyfit(np.log(r), np.log(P), 1)[0]:.2f} (natural images ~ -2)')
    return _spectrum

def frame(x, e, t):
    z = t * x + (1 - t) * s * e
    return (np.clip((z + 1) / 2, 0, 1) * 255).round().astype(np.uint8)

def big(a):
    return np.kron(a, np.ones((upscale, upscale, 1), dtype=np.uint8)) if upscale > 1 else a

ts = quantiles(frames)
durations = [int(1000 / fps)] * len(ts)
durations[0] = durations[-1] = int(1000 * hold)

os.makedirs(out, exist_ok=True)
# file-name settings: t_spacing ('logitnorm(-0.8, 0.8)' -> 'logitnorm-0.8_0.8'), noise scale, classes x per class, split / seed if changed
_sp = t_spacing.replace(' ', '')
_name, _, _rest = _sp.partition('(')
tag = '_'.join([_name + ''.join(_rest.rstrip(')').split(',')[:1]) + ''.join('_' + v for v in _rest.rstrip(')').split(',')[1:]) if _rest else _name,
                f'noise{s:g}', (f"c{'-'.join(map(str, cls))}" if classes else f'c{len(cls)}') + f'x{per_class}']
               + ([split] if split != 'train' else []) + ([f'seed{seed}'] if seed else []))
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
    d.text((pad, 2), f't = {t:.3f}  (frames at quantiles of {t_spacing})  noise scale {s:g}', fill=(0, 0, 0))
    for k, (c, name, x, e) in enumerate(samples):
        r, q = divmod(k, cols)
        x0, y0 = pad + q * (cell + pad), label_h + r * (cell + label_h + pad)
        if q == 0:
            d.text((x0, y0 + 1), f'{c}: {name}'[:48], fill=(0, 0, 0))
        canvas.paste(Image.fromarray(big(frame(x, e, t))), (x0, y0 + label_h))
    grid_frames.append(canvas)
grid_path = os.path.join(out, f'grid_{tag}_f{len(ts)}.gif')
save_gif(grid_path, grid_frames)

# strip: samples x the quantiles of t_spacing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
st = quantiles(strip_columns)
fig, axes = plt.subplots(len(samples), len(st), figsize=(1.3 * len(st), 1.3 * len(samples) + 0.6), squeeze=False)
for k, (c, name, x, e) in enumerate(samples):
    for j, t in enumerate(st):
        ax = axes[k, j]
        ax.imshow(frame(x, e, t), interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if k == 0:
            ax.set_title(f't={t:.3f}', fontsize=7)
        if j == 0:
            ax.set_ylabel(name[:18], fontsize=7)
fig.suptitle(f'training trajectories z_t = t x + (1 - t) {s:g} e  (t = data fraction, 1 = clean), columns at quantiles of {t_spacing}', fontsize=8)
fig.tight_layout()
strip_path = os.path.join(out, f'strip_{tag}_n{len(st)}.png')
fig.savefig(strip_path, dpi=150)
print(f'wrote {grid_path} ({len(samples)} samples, {len(ts)} frames) and {strip_path} ({len(st)} columns); '
      f't spacing {t_spacing}, noise scale {s:g}, classes {cls}')
