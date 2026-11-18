"""
Where along t did training improve the samples? Checkpoint splicing over a run's milestone checkpoints (train_jit.py
--keep_ckpt_every_epochs, ckpt_{iter}.pt): for each pair of consecutive checkpoints A (earlier) -> B (later), sample with
A everywhere except inside one t window, where B makes the network calls. The quality gained by swapping in B inside
window W, FID(A) - FID(A with B in W), is the part of A -> B's improvement that comes through W; summed over windows it
should roughly match FID(A) - FID(B) (interactions between windows make it inexact; the gap is reported). Repeating this
for every pair shows how the windows' shares shift over training (e.g. whether low-noise windows matter more as the
semantics saturate). FID-10k and FD-DINOv2 (more texture-sensitive than Inception) for every pass.

Every pass uses the same labels and initial noise (seeded), so the comparisons are paired: differences are much less noisy
than independent FIDs. The model's sampler (Heun, --steps, --time_schedule, CFG --cfg) is unchanged; a network call
belongs to window W when its t falls in [lo, hi) (the last window includes t = 1).

Writes {out}/splice_{run}.json and .png (per pair: each window's gain as a share of the pair's total, FID and FD-DINOv2;
and the shares across pairs).

$ python t_splice.py --run=runs/in1k_128/jit-b16_icl0/<run with ckpt_*.pt> --data_dir=/data/in1k_128
$ python t_splice.py --run=... --data_dir=... --iters=12510,25020 --windows=0,0.2,0.4,0.6,0.8,1   # one pair, 5 windows
$ python t_splice.py --run=... --data_dir=... --pairs=31275-37530,93825-100080                     # chosen pairs only
"""

import os
import glob
import json
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from jit_common import load_denoiser, generate_and_score
from data_jit import MemmapImages
from fid import InceptionMetrics, load_ref_stats
from dino_fd import DinoFD

# -----------------------------------------------------------------------------
run = '' # run directory with ckpt_{iter}.pt milestones
data_dir = ''
iters = '' # comma-separated milestone iterations to use (in order; consecutive ones are paired); '' = all found
pairs = '' # explicit pairs instead, comma-separated 'A-B' (e.g. '31275-37530,93825-100080'); overrides the consecutive pairing
windows = '0,0.1,0.2,0.3,0.4,0.5,0.65,0.8,1' # t window edges
num_images = 10000
cfg = 1.0
interval_min = 0.0
interval_max = 1.0
steps = 50
time_schedule = 'uniform'
dino = True # also FD-DINOv2
dino_encoder = 'dinov2_vitb14'
batch = 250
seed = 0
compile = True
device = 'cuda'
out = 'tsplice'
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

assert run and data_dir, 'pass --run and --data_dir'
found = sorted(glob.glob(os.path.join(run, 'ckpt_*.pt')), key=lambda p: int(p.rsplit('_', 1)[1][:-3]))
found = {int(p.rsplit('_', 1)[1][:-3]): p for p in found}
if pairs:
    pair_list = [tuple(int(v) for v in p_.split('-')) for p_ in pairs.split(',') if p_.strip()]
    its = sorted({i for p_ in pair_list for i in p_})
else:
    its = [int(v) for v in iters.split(',') if v.strip()] if iters else sorted(found)
    pair_list = list(zip(its[:-1], its[1:]))
assert len(its) >= 2 and all(i in found for i in its), f'need >= 2 milestones among {its}; found {sorted(found)}'
edges = [float(v) for v in windows.split(',')]
W = list(zip(edges[:-1], edges[1:]))
run_name = os.path.basename(run.rstrip('/'))
ctx = torch.autocast('cuda', dtype=torch.bfloat16) if 'cuda' in device else nullcontext()

data = MemmapImages(data_dir)
ref_path = os.path.join(data_dir, 'fid_stats_jit.npz')
ref = load_ref_stats(ref_path if os.path.exists(ref_path) else os.path.join(data_dir, 'fid_stats.npz'))
metrics = InceptionMetrics(device)
fd = DinoFD(device, encoder=dino_encoder) if dino else None
dref = fd.reference(data, n_per_class=50, cache=os.path.join(out, 'dino_ref_train50k.npz')) if dino else None

nets = {}
def net_at(i):
    if i not in nets:
        d, meta = load_denoiser(found[i], ema=1, device=device)
        if compile:
            d.net.compile()
        nets[i] = (d, meta)
    return nets[i]

class Spliced(nn.Module):
    # B's network for calls with t in [lo, hi), A's otherwise
    def __init__(self, a, b, lo, hi):
        super().__init__()
        self.a, self.b, self.lo, self.hi = a, b, lo, hi
    def forward(self, z, t, y, **kw):
        m = (t >= self.lo) & ((t < self.hi) | (self.hi >= 1.0))
        if bool(m.all()):
            return self.b(z, t, y, **kw)
        if not bool(m.any()):
            return self.a(z, t, y, **kw)
        return torch.where(m.view(-1, 1, 1, 1), self.b(z, t, y, **kw), self.a(z, t, y, **kw))

def score(denoiser, desc):
    feats = []
    hook = (lambda x, labels: feats.append(fd.features(x))) if dino else None
    r = generate_and_score(denoiser, metrics, ref, num_images, data.num_classes, device, ctx, cfg=cfg,
                           interval=(interval_min, interval_max), steps=steps, method='heun', batch_size=batch, seed=seed,
                           on_batch=hook, desc=desc)
    out_ = {'fid': r['fid']}
    if dino:
        out_['fd_dino'] = fd.distance(torch.cat(feats), dref)
    return out_

os.makedirs(out, exist_ok=True)
res_path = os.path.join(out, f'splice_{run_name}.json')
res = json.load(open(res_path)) if os.path.exists(res_path) else {}
res.setdefault('settings', {k: globals()[k] for k in config_keys})
res.setdefault('full', {})
res.setdefault('pairs', {})

def save():
    with open(res_path, 'w') as f:
        json.dump(res, f, indent=1)

for i in its: # each checkpoint on its own
    if str(i) not in res['full']:
        d, meta = net_at(i)
        d.time_schedule = time_schedule
        res['full'][str(i)] = dict(score(d, f'ckpt {i} (epoch {meta.get("epoch", 0):.0f})'), epoch=meta.get('epoch'))
        save()
        print(f"ckpt {i}: {res['full'][str(i)]}")

for a, b in pair_list:
    key = f'{a}->{b}'
    pr = res['pairs'].setdefault(key, {})
    da, _ = net_at(a)
    db, _ = net_at(b)
    net_a = da.net
    for lo, hi in W:
        wk = f'{lo:g}-{hi:g}'
        if wk in pr:
            continue
        da.net = Spliced(net_a, db.net, lo, hi)
        da.time_schedule = time_schedule
        try:
            pr[wk] = score(da, f'{key}: B in t [{lo:g}, {hi:g})')
        finally:
            da.net = net_a
        save()
        print(f'{key} window {wk}: {pr[wk]}')
    for k in [k for k in nets if k != b]: # keep only what the next pair needs
        del nets[k]
    torch.cuda.empty_cache()

# summary: each window's gain and share of the pair's total, per metric
metrics_ = ['fid'] + (['fd_dino'] if dino else [])
summary = {}
print(f'\n{run_name}: gain from swapping the later checkpoint into each window (FID points; share of the pair\'s total)')
for a, b in pair_list:
    key = f'{a}->{b}'
    summary[key] = {}
    for m in metrics_:
        A, B = res['full'][str(a)][m], res['full'][str(b)][m]
        gains = {f'{lo:g}-{hi:g}': A - res['pairs'][key][f'{lo:g}-{hi:g}'][m] for lo, hi in W}
        total = A - B
        summary[key][m] = {'A': A, 'B': B, 'total': total, 'gains': gains, 'sum_of_gains': sum(gains.values()),
                           'shares': {k: (g / total if abs(total) > 1e-9 else float('nan')) for k, g in gains.items()}}
        print(f"  {key} {m}: {A:.2f} -> {B:.2f} (total {total:+.2f}, windows sum {sum(gains.values()):+.2f}): "
              + ', '.join(f'{k} {g:+.2f}' for k, g in gains.items()))
res['summary'] = summary
save()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
pairs = list(summary)
fig, axes = plt.subplots(len(metrics_), 1, figsize=(10, 3.6 * len(metrics_)), squeeze=False)
centers = [(lo + hi) / 2 for lo, hi in W]
widths = [hi - lo for lo, hi in W]
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(pairs)))
for r, m in enumerate(metrics_):
    ax = axes[r, 0]
    for j, key in enumerate(pairs):
        s = summary[key][m]
        share = [s['shares'][f'{lo:g}-{hi:g}'] / (hi - lo) for lo, hi in W] # share per unit t, so window widths compare
        ax.step(edges, share + [share[-1]], where='post', color=cmap[j], lw=1.6,
                label=f"{key} ({m} {s['A']:.1f} -> {s['B']:.1f}, windows sum {s['sum_of_gains']:+.2f} of {s['total']:+.2f})")
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlim(0, 1); ax.grid(alpha=0.3)
    ax.set_xlabel('t window'); ax.set_ylabel(f'share of the {m} gain per unit t')
    ax.legend(fontsize=7)
axes[0, 0].set_title(f'{run_name}: where each 5-epoch improvement came from (later checkpoint swapped into one window)', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(out, f'splice_{run_name}.png'), dpi=150)
print(f'wrote {res_path} and .png')
