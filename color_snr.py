"""
When does each colour component and spatial frequency emerge from the noise, in different data spaces? No model: data
statistics only (CPU, seconds to minutes).

Noising z_t = t x + (1 - t) s e (t = data fraction, s = noise scale, e white). A component of the data with power P
(per frequency, orthonormal FFT) has SNR t^2 P / ((1 - t)^2 s^2), which reaches 1 at t* = s / (s + sqrt(P)): before t*
the noise hides it, after it the noisy image shows it. Computed for the principal colour axes of the data (PCA of pixel
colours: typically luminance and two opponent-colour axes with much less variance) at every spatial frequency, in each
data space:
  pm1             pixels in [-1, 1] (JiT), noise scale s (JiT's rule: img_size / 256 = 0.5 at 128 px)
  imagenet        per-channel (x - mean) / std (ImageNet stats), same s: ~2.2x the data scale, so everything emerges earlier
  imagenet_match  the same with s scaled to keep pm1's average SNR (s x 1 / (2 mean std) = 1.106 at 128 px)
  whiten          each principal colour axis scaled to unit variance, s matched to pm1's luminance: equalizes colour and
                  luminance (what a decorrelating colour transform with per-axis scaling would do)
Prints each axis's variance share and the emergence t of its total power and of a few frequency bands; plots t*(f) per
axis and space ({out}.png, {out}.json).

$ python color_snr.py --data_dir=/data/in1k_128
"""

import os
import json

import numpy as np

from data_jit import MemmapImages, IMAGENET_MEAN, IMAGENET_STD

# -----------------------------------------------------------------------------
data_dir = ''
num_images = 5000
noise_scale = 0.0 # 0 -> img_size / 256
seed = 0
out = 'color_snr'
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

assert data_dir, 'pass --data_dir'
data = MemmapImages(data_dir)
R = data.img_size
s0 = noise_scale or R / 256
idx = np.sort(np.random.default_rng(seed).choice(len(data), min(num_images, len(data)), replace=False))
x = data.images[idx].astype(np.float32) / 127.5 - 1 # [N, R, R, 3] in [-1, 1]

# principal colour axes of pm1 pixels
pix = x.reshape(-1, 3)
mean_pm1 = pix.mean(0)
cov = np.cov((pix - mean_pm1).T)
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order] # columns: axes, largest variance first
names = ['axis1 (luminance-like)', 'axis2', 'axis3']
mean_std = float(np.mean(np.array(IMAGENET_STD)))

def space(name):
    # (images in that space [N, R, R, 3], noise scale, axis basis to project on) for each data space
    if name == 'pm1':
        return x, s0, evecs
    if name in ('imagenet', 'imagenet_match'):
        xi = ((x + 1) / 2 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        # the same physical colour axes, mapped into this space (per-channel scaling skews them slightly)
        basis = np.diag(0.5 / np.array(IMAGENET_STD)) @ evecs
        basis = basis / np.linalg.norm(basis, axis=0, keepdims=True)
        return xi, (s0 if name == 'imagenet' else s0 / (2 * mean_std)), basis
    if name == 'whiten':
        # project on the axes and scale each to unit variance; noise matched to pm1's luminance SNR
        xw = (x - mean_pm1) @ evecs / np.sqrt(evals)
        return xw, s0 / np.sqrt(evals[0]), np.eye(3)
    raise ValueError(name)

fy, fx = np.meshgrid(np.fft.fftfreq(R) * R, np.fft.fftfreq(R) * R, indexing='ij')
rad = np.sqrt(fx ** 2 + fy ** 2).round().astype(int)
radii = np.arange(1, R // 2 + 1)
res = {'variance_share': (evals / evals.sum()).tolist(), 'axes_rgb': evecs.T.tolist(), 'spaces': {}}
print(f'{R}px, {len(idx)} images; principal colour axes of [-1, 1] pixels (RGB weights), variance share:')
for k in range(3):
    print(f'  {names[k]:<24} rgb {np.round(evecs[:, k], 3)}  {100 * evals[k] / evals.sum():5.1f}%')

for sp in ('pm1', 'imagenet', 'imagenet_match', 'whiten'):
    xs, s, basis = space(sp)
    comp = (xs - xs.reshape(-1, 3).mean(0)) @ basis # [N, R, R, 3] components along the axes
    P = np.zeros((3, len(radii)))
    total = np.zeros(3)
    for k in range(3):
        acc = np.zeros((R, R))
        for i in range(0, len(comp), 500):
            acc += (np.abs(np.fft.fft2(comp[i:i + 500, :, :, k], norm='ortho')) ** 2).sum(0)
        acc /= len(comp)
        P[k] = [acc[rad == r].mean() for r in radii]
        total[k] = comp[..., k].var()
    t_star = s / (s + np.sqrt(P)) # [3, radii] emergence t per axis and frequency
    t_total = s / (s + np.sqrt(total)) # whole-axis variance (per pixel)
    res['spaces'][sp] = {'noise_scale': s, 't_star': t_star.tolist(), 't_total': t_total.tolist(), 'radii': radii.tolist()}
    bands = [1, 4, 16, R // 2]
    print(f'\n{sp} (noise scale {s:.3f}): emergence t (SNR = 1) per axis: whole-axis variance; frequency 1 / 4 / 16 / {R // 2} cycles per image')
    for k in range(3):
        print(f'  {names[k]:<24} {t_total[k]:.3f};   ' + ' / '.join(f'{t_star[k, b - 1]:.3f}' for b in bands))

with open(out + '.json', 'w') as f:
    json.dump(res, f, indent=1)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
cols = ['k', '#2e86c1', '#c0392b']
for a, sp in zip(axes, res['spaces']):
    d = res['spaces'][sp]
    for k in range(3):
        a.semilogx(radii, d['t_star'][k], color=cols[k], lw=1.5, label=f"{names[k]} ({100 * res['variance_share'][k]:.1f}% var)")
    a.set_title(f"{sp} (s = {d['noise_scale']:.3f})", fontsize=9)
    a.set_xlabel('spatial frequency (cycles / image)'); a.grid(alpha=0.3, which='both')
axes[0].set_ylabel('t where SNR = 1 (later = hidden longer)')
axes[0].legend(fontsize=7)
fig.suptitle(f'when each colour axis and frequency emerges from the noise ({R}px)', fontsize=10)
fig.tight_layout()
fig.savefig(out + '.png', dpi=150)
print(f'\nwrote {out}.json and .png')
