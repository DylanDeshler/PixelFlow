"""
The share of each frequency a trained denoiser recovers, per noise level: the nonlinear counterpart of the Wiener gain, for
train_jit.py --loss_freq_weight=table. For a linear (Gaussian) denoiser, 1 - E|X - X_hat|^2 / E|X|^2 per frequency is the
Wiener gain SNR / (1 + SNR); measured on a trained network it includes everything the model predicts from context (edges,
textures, class), which on ImageNet is 5-20x the linear gain for fine detail at mid noise. A trained model's recovery is a
lower bound on what's recoverable, so use the best checkpoint available (EMA weights).

For each noise level on a logit-spaced grid, held-out images are noised as in training (true labels) and predicted; the
error and image power are averaged over images and channels per integer-radius frequency ring:
    gain[t, ring] = clip(1 - E|FFT(x_pred - x)|^2 / E|FFT(x)|^2, 0, 1)
Writes JSON {logit_t, gain [n_t][n_rings], img_size, ckpt, iter, ...} to --out (default
{run}/diagnostics/freq_table_{iter}.json) and prints a band summary.

$ python build_freq_table.py --ckpt=runs/in1k_128/jit-b16_icl0/<run> --data_dir=/data/in1k_128
"""

import os
import json
import math
from contextlib import nullcontext

import numpy as np
import torch

from data_jit import MemmapImages, to_model_input
from jit_common import load_denoiser

# -----------------------------------------------------------------------------
ckpt = '' # run directory (its ckpt.pt) or checkpoint file
data_dir = '' # the prepared dataset (its val/ split is used)
out = '' # '' -> {run}/diagnostics/freq_table_{iter}.json
ema = 1 # 1 or 2 = that EMA (the sampled weights; the best estimate of what's recoverable), 0 = raw
n_images = 1024
t_points = 24
logit_min, logit_max = -5.0, 5.0 # the grid in logit(t); the weighting holds the end values beyond it
batch_size = 128
seed = 0
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

assert ckpt and data_dir, 'pass --ckpt and --data_dir'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' or dtype == 'float32') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

path = os.path.join(ckpt, 'ckpt.pt') if os.path.isdir(ckpt) else ckpt
denoiser, meta = load_denoiser(path, ema, device)
net, ns = denoiser.net, denoiser.noise_scale
val = MemmapImages(os.path.join(data_dir, 'val'))
H = val.img_size
assert denoiser.img_size == H, f'{path} and {data_dir} disagree'
idx = np.linspace(0, len(val) - 1, min(n_images, len(val))).round().astype(np.int64)
xs_all, ys_all = val.get(idx)

f = torch.fft.fftfreq(H, device=device) * H
ring = torch.sqrt(f.view(-1, 1) ** 2 + f.view(1, -1) ** 2).round().long() # [H, H] integer radius
n_rings = int(ring.max()) + 1
count = torch.bincount(ring.flatten(), minlength=n_rings).float()
def per_ring(p): # [H, H] -> [n_rings] mean over each ring
    return torch.zeros(n_rings, device=device).scatter_add_(0, ring.flatten(), p.flatten()) / count

logits = np.linspace(logit_min, logit_max, t_points)
gain = []
g = torch.Generator(device=device).manual_seed(seed)
with torch.no_grad():
    for lt in logits:
        t_val = 1 / (1 + math.exp(-lt))
        err_p = torch.zeros(H, H, device=device); img_p = torch.zeros(H, H, device=device)
        for i in range(0, len(xs_all), batch_size):
            x = to_model_input(xs_all[i:i + batch_size], flip=False, device=device)
            y = torch.from_numpy(ys_all[i:i + batch_size]).to(device)
            t = torch.full((len(x), 1, 1, 1), t_val, device=device)
            z = t * x + (1 - t) * torch.randn(x.shape, generator=g, device=device) * ns
            with ctx:
                x_pred = net(z, t.flatten(), y)
            err_p += torch.fft.fft2(x_pred.float() - x).abs().pow(2).sum(dim=(0, 1))
            img_p += torch.fft.fft2(x).abs().pow(2).sum(dim=(0, 1))
        gain.append((1 - per_ring(err_p) / per_ring(img_p).clamp_min(1e-12)).clamp(0, 1).tolist())

run_dir = os.path.dirname(os.path.abspath(path))
it = meta.get('iter_num')
out = out or os.path.join(run_dir, 'diagnostics', f'freq_table_{it}.json')
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, 'w') as fh:
    json.dump({'logit_t': logits.tolist(), 'gain': gain, 'img_size': H, 'n_rings': n_rings, 'ckpt': path, 'iter': it,
               'weights': f'ema{ema}' if ema else 'raw', 'config': config}, fh)
print(f'{os.path.basename(run_dir)} (iter {it}): recovered share by frequency (8 bands of rings, band0 coarsest) and noise level')
edges = np.linspace(0, n_rings, 9).round().astype(int)
for lt, row in zip(logits[::3], gain[::3]):
    bands = [np.mean(row[a:b]) for a, b in zip(edges[:-1], edges[1:])]
    print(f'  t={1 / (1 + math.exp(-lt)):.3f}: ' + ' '.join(f'{v:.2f}' for v in bands))
print(f'written to {out}')
