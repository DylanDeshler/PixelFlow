"""
Attention-sink diagnostic for trained JiT checkpoints (no training): do the models park attention on a few tokens, and do
tokens grow norm spikes, the symptoms register tokens and gated attention are meant to remove?

For each run, on held-out images at a few noise levels (the same images, noise and labels for every run), the instrumented
forward (diagnostics.instrumented_forward: the model's own math with explicit attention weights) gives per block:
  sink mean / max   per head, the largest share of attention any single key token receives (averaged over queries), then
                    the mean / max over heads. Uniform attention gives 1 / keys; a sink head puts a large share on one key
  sink x uniform    the max-head sink relative to uniform (1 = no concentration; >> 1 = a sink)
  prefix mass       attention from patches to in-context / kv-prefix tokens (where a model with them puts its sinks)
  norm ratio        largest / median token RMS after the block (sink tokens tend to have large norms)
  gate              mean attention-gate value (gated-attention models; 1 = gate open, the init value)
Prints a table per run (blocks x those columns, averaged over noise levels), then one summary line per run, and writes
{out}.json with everything per noise level.

$ python sink_diag.py --runs='best=runs/in1k_128/jit-b16_icl0/<best run>;in-context=runs/in1k_128/jit-b16/<run>' --data_dir=/data/in1k_128
$ python sink_diag.py --runs=... --data_dir=... --ts=0.1,0.5,0.9 --num_images=256 --ema=1 --device=cuda
"""

import os
import json

import numpy as np
import torch

from jit_common import load_denoiser
from data_jit import MemmapImages, to_model_input
import diagnostics as diag_lib

# -----------------------------------------------------------------------------
runs = '' # ';'-separated 'label=run_dir' (or a checkpoint .pt path) entries
data_dir = '' # prepared dataset (its val/ split is used)
num_images = 256
batch = 64
ts = '0.1,0.3,0.5,0.7,0.9' # noise levels (t = data fraction: 0 = pure noise)
ema = 1 # weights: 1 / 2 = that EMA, 0 = raw
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 0
out = 'sink_diag'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

entries = []
for e in (e.strip() for e in runs.split(';') if e.strip()):
    label, path = e.split('=', 1) if '=' in e else (os.path.basename(e.rstrip('/')), e)
    ckpt = path if path.endswith('.pt') else os.path.join(path, 'ckpt.pt')
    assert os.path.exists(ckpt), f'no checkpoint at {ckpt}'
    entries.append((label, ckpt))
assert entries and data_dir, 'pass --runs and --data_dir'

val = MemmapImages(os.path.join(data_dir, 'val'))
idx = np.linspace(0, len(val) - 1, num_images).astype(int)
xs, ys = val.get(idx)
t_list = [float(v) for v in ts.split(',') if v]
keys = ('attn_sink_mass', 'attn_sink_max', 'attn_sink_uniform', 'attn_prefix_mass', 'token_norm_ratio', 'attn_gate_mean')
ctx = torch.autocast('cuda', dtype=torch.bfloat16) if 'cuda' in device else torch.autocast('cpu', enabled=False)

results = {}
for label, ckpt in entries:
    denoiser, meta = load_denoiser(ckpt, ema=ema, device=device)
    net = denoiser.net
    noise_scale = meta['denoiser_args'].get('noise_scale', 1.0)
    per_t = {}
    for t_val in t_list:
        g = torch.Generator(device='cpu').manual_seed(seed)
        acc = {k: [] for k in keys}
        for s in range(0, num_images, batch):
            x = to_model_input(xs[s:s + batch], flip=False, device=device)
            y = torch.from_numpy(ys[s:s + batch]).to(device)
            e = torch.randn(x.shape, generator=g).to(device) * noise_scale
            t = torch.full((len(x),), t_val, device=device)
            z = t.view(-1, 1, 1, 1) * x + (1 - t.view(-1, 1, 1, 1)) * e
            with torch.no_grad(), ctx:
                _, stats, _ = diag_lib.instrumented_forward(net, z, t, y)
            for k in keys:
                acc[k].append(stats[k].float().cpu() * len(x))
        per_t[t_val] = {k: (torch.stack(v).sum(0) / num_images).tolist() for k, v in acc.items()} # [blocks]
    results[label] = {'ckpt': ckpt, 'iter': meta.get('iter_num'), 'per_t': per_t}
    # table: blocks x columns, averaged over noise levels
    nb = len(next(iter(per_t.values()))['attn_sink_mass'])
    avg = {k: [sum(per_t[t][k][b] for t in t_list) / len(t_list) for b in range(nb)] for k in keys}
    print(f"\n{label} ({ckpt}, iter {meta.get('iter_num')}, EMA {ema}), averaged over t = {t_list}")
    print(f"  {'block':>5} {'sink mean':>10} {'sink max':>9} {'uniform':>8} {'max x unif':>10} {'prefix mass':>12} {'norm ratio':>11} {'gate':>6}")
    for b in range(nb):
        rel = avg['attn_sink_max'][b] / max(avg['attn_sink_uniform'][b], 1e-12)
        print(f"  {b + 1:>5} {avg['attn_sink_mass'][b]:>10.3f} {avg['attn_sink_max'][b]:>9.3f} {avg['attn_sink_uniform'][b]:>8.3f} {rel:>10.1f} "
              f"{avg['attn_prefix_mass'][b]:>12.3f} {avg['token_norm_ratio'][b]:>11.2f} {avg['attn_gate_mean'][b]:>6.2f}")
    results[label]['summary'] = {
        'max_sink_x_uniform': max(avg['attn_sink_max'][b] / max(avg['attn_sink_uniform'][b], 1e-12) for b in range(nb)),
        'mean_sink_x_uniform': sum(avg['attn_sink_mass'][b] / max(avg['attn_sink_uniform'][b], 1e-12) for b in range(nb)) / nb,
        'max_norm_ratio': max(avg['token_norm_ratio']),
    }
    del denoiser, net
    if 'cuda' in device:
        torch.cuda.empty_cache()

print('\nsummary (strongest block; 1 = uniform attention):')
for label, r in results.items():
    s_ = r['summary']
    print(f"  {label:<24} max-head sink {s_['max_sink_x_uniform']:6.1f}x uniform, mean-head sink {s_['mean_sink_x_uniform']:5.1f}x, "
          f"max token-norm ratio {s_['max_norm_ratio']:.1f}")
with open(out + '.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'wrote {out}.json')
