"""
Head diagnostics on finished checkpoints, no training. For each run, on the same held-out images and noise (the first
n_images of the diag_images evenly spaced val images train_jit.py's diagnostics use):
    one-step    held-out loss per noise level and relative error per frequency band (diagnostics.noise_level_pass), for
                the sampled prediction and, in skip pathway runs, the trunk-only one
    sensitivity how strongly the prediction follows its noisy input, per noise level and band: copy, gain and the Tweedie
                ratio (1 = an MMSE denoiser with the same error; diagnostics.input_sensitivity), sampled and trunk-only
    trajectory  held-out images noised to t0 and sampled from there with the run's sampler: final error vs the one-step
                error at t0, per band, and the final image's spectrum vs the clean image's (diagnostics.trajectory_pass)
Prints side-by-side tables across the runs and writes {run}/diagnostics/ckpt_diag_{iter}_{weights}.json.

$ python diagnose_ckpt.py --ckpts=runs/in100_128/jit-b16_icl0/<run1>,runs/in100_128/jit-b16_icl0/<run2> --data_dir=/data/in100_128
--ckpts takes run directories (their ckpt.pt) or checkpoint files; milestone ckpt_{iter}.pt files compare runs at matched
iterations. EMA1 weights by default (what's sampled); --ema=0 for the raw weights the training diagnostics use.
"""

import os
import json
from contextlib import nullcontext

import numpy as np
import torch

from data_jit import MemmapImages, to_model_input
from jit_common import load_denoiser
import diagnostics as diag_lib

# -----------------------------------------------------------------------------
ckpts = '' # comma list of run directories or checkpoint files
data_dir = '' # the prepared dataset the runs trained on (its val/ split is used)
ema = 1 # 1 or 2 = that EMA, 0 = raw weights
diag_images = 1024 # the held-out image set: diag_images evenly spaced val images, as in train_jit.py
n_images = 256 # images used from it
t_bins = 10
freq_bands = 8
trajectory_t0 = '0.3,0.5,0.7,0.85'
sampling_method = '' # trajectory sampler: '' = the checkpoint's; 'heun', 'euler' or 'ancestral'
num_sampling_steps = 0 # 0 = the checkpoint's
sampling_eta = 1.0
time_schedule = 'uniform'
batch_size = 128
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

val = MemmapImages(os.path.join(data_dir, 'val'))
idx = np.linspace(0, len(val) - 1, min(diag_images, len(val))).round().astype(np.int64)[:n_images]
xs, ys = val.get(idx)
xs, ys = to_model_input(xs, flip=False, device=device), torch.from_numpy(ys).to(device)
ts = diag_lib.t_grid(t_bins)
t0s = [float(v) for v in trajectory_t0.split(',') if v]
masks = diag_lib.radial_band_masks(val.img_size, freq_bands, device)
weights = f'ema{ema}' if ema else 'raw'

results = []
for spec in ckpts.split(','):
    path = os.path.join(spec, 'ckpt.pt') if os.path.isdir(spec) else spec
    if not os.path.isfile(path):
        raise FileNotFoundError(f'checkpoint not found: {path}')
    denoiser, meta = load_denoiser(path, ema, device)
    assert denoiser.img_size == val.img_size and denoiser.num_classes == val.num_classes, f'{path} and {data_dir} disagree'
    if sampling_method:
        denoiser.method = sampling_method
    if num_sampling_steps:
        denoiser.steps = num_sampling_steps
    denoiser.eta, denoiser.time_schedule = sampling_eta, time_schedule
    net, ns, has_skip = denoiser.net, denoiser.noise_scale, getattr(denoiser.net, 'skip_pathway', False)
    run = os.path.basename(os.path.dirname(os.path.abspath(path)))
    it = meta.get('iter_num')
    print(f"{run} (iter {it}, {weights} weights, trajectory sampler {denoiser.method} {denoiser.steps} steps"
          f"{f', eta {sampling_eta:g}' if denoiser.method == 'ancestral' else ''})")
    r = {'run': run, 'ckpt': path, 'iter': it, 'weights': weights, 'has_skip': has_skip, 't': ts.tolist(), 't0': t0s,
         'sampler': {'method': denoiser.method, 'steps': denoiser.steps, 'eta': sampling_eta, 'time_schedule': time_schedule}}
    heads = [('sampled', True)] + ([('trunk', False)] if has_skip else [])
    for head, skip_on in heads:
        p = diag_lib.noise_level_pass(net, xs, ys, ts, ns, denoiser.t_eps, ctx, masks, chunk=batch_size, skip_on=skip_on)
        r[f'onestep_{head}'] = {k: p[k].tolist() for k in ('v_loss', 'x_mse', 'freq_rel_err')}
        s = diag_lib.input_sensitivity(net, xs, ys, ts, ns, ctx, n_images, chunk=min(batch_size, 64), skip_on=skip_on, masks=masks)
        r[f'sensitivity_{head}'] = {k: v.tolist() for k, v in s.items()}
    tr = diag_lib.trajectory_pass(denoiser, xs, ys, t0s, ns, ctx, masks, n_images, chunk=batch_size)
    r['trajectory'] = {k: v.tolist() for k, v in tr.items()}
    out = os.path.join(os.path.dirname(os.path.abspath(path)), 'diagnostics')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f'ckpt_diag_{it}_{weights}.json'), 'w') as f:
        json.dump(dict(r, config=config), f)
    results.append(r)
    del denoiser, net
    if device_type == 'cuda':
        torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# side-by-side tables: runs as columns A, B, ...

letters = [chr(ord('A') + i) for i in range(len(results))]
print('\nruns:')
for l, r in zip(letters, results):
    print(f"  {l}  {r['run']} (iter {r['iter']})")
tl = [f'{t:.3f}' for t in ts.tolist()]

def table(title, rows, cols, cell, width=9):
    print(f'\n{title}')
    print(' ' * 10 + ''.join(f'{c:>{width}}' for c in cols))
    for i, row in enumerate(rows):
        print(f'{row:>10}' + ''.join(f'{cell(i, j):>{width}}' for j in range(len(cols))))

def trunk_or_dash(r, key, i, fmt):
    return fmt(r[f'sensitivity_trunk'][key][i]) if r['has_skip'] else '-'

table('held-out v-loss by noise level: sampled (trunk-only)', [f't={t}' for t in tl], letters,
      lambda i, j: f"{results[j]['onestep_sampled']['v_loss'][i]:.4f}" + (f"({results[j]['onestep_trunk']['v_loss'][i]:.4f})" if results[j]['has_skip'] else ''), width=19)
for key, what in (('copy', 'copy: 1 = passes the input through'), ('gain', 'gain: squared amplification of a change to the input (copy = 1)'),
                  ('tweedie', 'Tweedie ratio: 1 = an MMSE denoiser with the same error, > 1 = follows its input more')):
    table(f'{what}; sampled (trunk-only)', [f't={t}' for t in tl], letters,
          lambda i, j: f"{results[j]['sensitivity_sampled'][key][i]:.3f}" + (f"({trunk_or_dash(results[j], key, i, lambda v: f'{v:.3f}')})" if results[j]['has_skip'] else ''), width=15)
bands = [f'band{k}' for k in range(freq_bands)]
for t_pick in (0.583, 0.841, 0.953):
    i = int(np.argmin([abs(t - t_pick) for t in ts.tolist()]))
    table(f'Tweedie ratio by frequency band at t={tl[i]} (sampled prediction; band0 = coarsest)', letters, bands,
          lambda a, k: f"{results[a]['sensitivity_sampled']['tweedie_band'][i][k]:.2f}")
for i, t0 in enumerate(t0s):
    table(f'trajectory from t0={t0:g}: final error / one-step error, by band (1 = the sampler adds nothing)', letters, bands,
          lambda a, k: f"{results[a]['trajectory']['traj_err'][i][k] / results[a]['trajectory']['onestep_err'][i][k]:.2f}")
    table(f'trajectory from t0={t0:g}: final image spectrum, log10 power / clean image (> 0 = excess, e.g. grain)', letters, bands,
          lambda a, k: f"{results[a]['trajectory']['traj_spectrum'][i][k]:+.2f}")
