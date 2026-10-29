"""
Gradient signal vs noise of the training loss, per noise level and frequency band, on finished checkpoints (no training).
Tests whether frequency-weighting the loss could help: if the loss components the noise has made unpredictable (fine
detail at high noise) mostly add gradient noise, down-weighting them (e.g. by their Wiener gain) should make training
more efficient without changing what the model converges to (a weighted L2 in an orthonormal basis has the same minimizer).

For each checkpoint and noise level t, on K independent mini-batches of m held-out images (fixed t, the run's noise):
  loss_b      the training loss (x-prediction, v-loss) restricted to radial frequency band b of the residual, via the FFT
              (Parseval: the bands sum to the plain loss)
  g_b         its gradient w.r.t. every trainable parameter, accumulated as running sums over the K batches
From those, per band (and for the plain and the Wiener-weighted loss):
  signal      |E g|^2, unbiased: (|sum g|^2 - sum |g|^2) / (K (K - 1))
  noise       E |g - E g|^2 for a batch of m (unbiased sample variance)
  B_noise     m * noise / signal: the batch size at which gradient noise equals signal (McCandlish et al. 2018,
              "An Empirical Model of Large-Batch Training"); inf when no signal is detected
  frac@B      signal / (signal + noise * m / B): the share of the expected squared gradient that is signal at the run's
              training batch B (1 = clean, 0 = pure noise)
  loss share  the band's share of the loss at that t
The Wiener weight of frequency f at noise level t: SNR / (1 + SNR), SNR = t^2 P_x(f) / ((1 - t)^2 noise_scale^2), P_x the
images' power spectrum (radially averaged) -- the share of that frequency the noisy input still carries. Also reported:
the cosine between the plain and the weighted loss's expected gradients (how much the weighting turns the update).

Use the raw weights (--ema=0, ckpt.pt): those are what the optimizer steps from. Writes
{run}/diagnostics/grad_snr_{iter}_{weights}.json and prints tables.

$ python analyze_grad_snr.py --ckpts=runs/in1k_128/jit-b16_p32_icl0/muon,runs/in1k_128/jit-b16_icl0/muon,runs/in1k_128/jit-b16_p8_icl0/muon \\
      --data_dir=/data/in1k_128
"""

import os
import json
import math
from contextlib import nullcontext

import numpy as np
import torch

from data_jit import MemmapImages, to_model_input
from jit_common import load_denoiser
import diagnostics as diag_lib

# -----------------------------------------------------------------------------
ckpts = '' # comma list of run directories (their ckpt.pt) or checkpoint files
data_dir = '' # the prepared dataset the runs trained on (its val/ split is used)
ema = 0 # 0 = raw weights (what the optimizer steps from; ckpt.pt only), 1 or 2 = that EMA
ts = '0.1,0.2,0.3,0.45,0.6,0.75,0.9' # noise levels (t = 1 is clean)
freq_bands = 8
n_batches = 16 # K independent mini-batches per noise level
batch_size = 64 # m images per mini-batch (K * m distinct held-out images, the same set at every t)
seed = 0
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

assert ckpts and data_dir, 'pass --ckpts and --data_dir'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' or dtype == 'float32') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# JiT's blocks and final layer are @torch.compile'd; their compiled backwards donate buffers, which can't be combined with
# retain_graph (one forward, a backward per band). Donation only saves memory, so turn it off
import torch._functorch.config
torch._functorch.config.donated_buffer = False

val = MemmapImages(os.path.join(data_dir, 'val'))
n_img = n_batches * batch_size
assert n_img <= len(val), f'need {n_img} held-out images, the val split has {len(val)}'
idx = np.linspace(0, len(val) - 1, n_img).round().astype(np.int64)
xs_all, ys_all = val.get(idx)
H = val.img_size
masks = diag_lib.radial_band_masks(H, freq_bands, device) # [bands, H, H], a partition of the frequencies
t_list = [float(v) for v in ts.split(',') if v]
weights = f'ema{ema}' if ema else 'raw'


def power_spectrum(x):
    # images' power per frequency, E |FFT(x)|^2 / (H W) (white noise of variance s^2 has s^2), radially averaged [H, H]
    p = torch.fft.fft2(x.float()).abs().pow(2).mean(dim=(0, 1)) / (H * H)
    f = torch.fft.fftfreq(H, device=x.device) * H
    r = torch.sqrt(f.view(-1, 1) ** 2 + f.view(1, -1) ** 2).round().long()
    radial = torch.zeros(int(r.max()) + 1, device=x.device).scatter_add_(0, r.flatten(), p.flatten())
    count = torch.zeros_like(radial).scatter_add_(0, r.flatten(), torch.ones_like(p).flatten())
    return (radial / count.clamp_min(1))[r]


def band_losses(x_pred, x, z, t, t_eps, wiener):
    # the training loss of the batch restricted to each frequency band [bands], the plain loss, and the Wiener-weighted
    # loss. residual r = v_pred - v = (x_pred - x) / (1 - t); mean_p r^2 = sum_f |R(f)|^2 / (H W)^2 per channel
    r = (x_pred.float() - x) / (1 - t).clamp_min(t_eps)
    pw = torch.fft.fft2(r).abs().pow(2).mean(dim=(0, 1)) / (H * H) ** 2 # [H, H], averaged over images and channels
    per_band = (masks * pw).sum(dim=(1, 2))
    return per_band, per_band.sum(), (wiener * pw).sum()


results = []
for spec in ckpts.split(','):
    path = os.path.join(spec, 'ckpt.pt') if os.path.isdir(spec) else spec
    if not os.path.isfile(path):
        raise FileNotFoundError(f'checkpoint not found: {path}')
    denoiser, meta = load_denoiser(path, ema, device)
    assert denoiser.img_size == H and denoiser.num_classes == val.num_classes, f'{path} and {data_dir} disagree'
    net, ns, t_eps = denoiser.net, denoiser.noise_scale, denoiser.t_eps
    cfg = meta.get('config', {})
    train_batch = cfg.get('batch_size', 1024) * cfg.get('gradient_accumulation_steps', 1)
    params = [p for p in net.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in params)
    run = os.path.basename(os.path.dirname(os.path.abspath(path)))
    it = meta.get('iter_num')
    print(f'\n{run} (iter {it}, {weights} weights, {n_par / 1e6:.1f}M parameters, training batch {train_batch}, noise scale {ns})')
    xs = to_model_input(xs_all, flip=False, device=device)
    ys = torch.from_numpy(ys_all).to(device)
    P_x = power_spectrum(xs)
    names = [f'band{b}' for b in range(freq_bands)] + ['plain', 'wiener']
    r_out = {'run': run, 'ckpt': path, 'iter': it, 'weights': weights, 'train_batch': train_batch, 't': t_list, 'names': names,
             'B_noise': [], 'frac_at_B': [], 'loss_share': [], 'cos_plain_wiener': [], 'wiener_band_mean': []}
    for t_val in t_list:
        snr = t_val ** 2 * P_x / ((1 - t_val) ** 2 * ns ** 2)
        wiener = snr / (1 + snr)
        S = {nm: torch.zeros(n_par, device=device) for nm in names} # sums of gradients
        Q = {nm: 0.0 for nm in names} # sums of squared gradient norms
        cross = 0.0 # sum over batches of <g_plain, g_wiener>
        loss_sum = torch.zeros(freq_bands, device=device)
        g = torch.Generator(device=device).manual_seed(seed)
        for k in range(n_batches):
            x, y = xs[k * batch_size:(k + 1) * batch_size], ys[k * batch_size:(k + 1) * batch_size]
            e = torch.randn(x.shape, generator=g, device=device) * ns
            t = torch.full((len(x), 1, 1, 1), t_val, device=device)
            z = t * x + (1 - t) * e
            with ctx:
                x_pred = net(z, t.flatten(), y)
            per_band, plain, weighted = band_losses(x_pred, x, z, t, t_eps, wiener)
            loss_sum += per_band.detach()
            losses = list(per_band) + [plain, weighted]
            flat = {}
            for i, (nm, l) in enumerate(zip(names, losses)):
                grads = torch.autograd.grad(l, params, retain_graph=i < len(losses) - 1, allow_unused=True)
                v = torch.cat([(gr if gr is not None else torch.zeros_like(p)).reshape(-1).float() for gr, p in zip(grads, params)])
                S[nm] += v
                Q[nm] += v.square().sum().item()
                if nm in ('plain', 'wiener'):
                    flat[nm] = v
                del grads
            cross += (flat['plain'] * flat['wiener']).sum().item()
            del flat
        K, m = n_batches, batch_size
        row_B, row_f = [], []
        sig = {}
        for nm in names:
            s2 = S[nm].square().sum().item()
            signal = (s2 - Q[nm]) / (K * (K - 1))
            noise = (Q[nm] - s2 / K) / (K - 1)
            sig[nm] = signal
            row_B.append(m * noise / signal if signal > 0 else float('inf'))
            row_f.append(signal / (signal + noise * m / train_batch) if signal > 0 else 0.0)
        dot = ((S['plain'] * S['wiener']).sum().item() - cross) / (K * (K - 1))
        cos = dot / math.sqrt(sig['plain'] * sig['wiener']) if sig['plain'] > 0 and sig['wiener'] > 0 else float('nan')
        r_out['B_noise'].append(row_B); r_out['frac_at_B'].append(row_f)
        r_out['loss_share'].append((loss_sum / loss_sum.sum()).tolist()); r_out['cos_plain_wiener'].append(cos)
        r_out['wiener_band_mean'].append(((masks * wiener).sum(dim=(1, 2)) / masks.sum(dim=(1, 2))).tolist())
        del S
        print(f'  t={t_val:.2f} done', flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(path)), 'diagnostics')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f'grad_snr_{it}_{weights}.json'), 'w') as f:
        json.dump(dict(r_out, config=config), f)
    results.append(r_out)
    del denoiser, net, params
    if device_type == 'cuda':
        torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# tables, one block per checkpoint: rows noise levels, columns bands (band0 = coarsest), then the plain and weighted loss

fmt_B = lambda v: 'inf' if v == float('inf') else (f'{v:.0f}' if v >= 10 else f'{v:.1f}')
for r in results:
    print(f"\n=== {r['run']} (iter {r['iter']}, {r['weights']} weights; training batch {r['train_batch']})")
    head = ''.join(f'{nm:>9}' for nm in r['names'])
    for title, key, fmt in (('gradient noise scale B_noise (batch where noise = signal; >> training batch: noise-dominated)', 'B_noise', fmt_B),
                            (f"signal fraction at the training batch (1 = clean gradient, 0 = pure noise)", 'frac_at_B', lambda v: f'{v:.2f}')):
        print(f'\n{title}\n{"t":>6}{head}')
        for t_val, row in zip(r['t'], r[key]):
            print(f'{t_val:>6.2f}' + ''.join(f'{fmt(v):>9}' for v in row))
    print(f'\nloss share by band, mean Wiener weight by band, and cos(plain, Wiener-weighted expected gradient)')
    print(f'{"t":>6}' + ''.join(f'{"b" + str(b):>7}' for b in range(freq_bands)) + ' |' + ''.join(f'{"w" + str(b):>6}' for b in range(freq_bands)) + '    cos')
    for i, t_val in enumerate(r['t']):
        print(f'{t_val:>6.2f}' + ''.join(f'{v:>7.1%}' for v in r['loss_share'][i]) + ' |'
              + ''.join(f'{v:>6.2f}' for v in r['wiener_band_mean'][i]) + f"{r['cos_plain_wiener'][i]:>7.2f}")
