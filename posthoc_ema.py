"""
Post-hoc EMA (EDM2): rebuild power-function EMAs of any length from a run's snapshots (train_jit.py --phema_stds) and score
each with FID, to choose the EMA after training instead of guessing it before. See phema.py for the method.

For each target sigma_rel (the EMA window as a fraction of training so far), at one snapshot step: the snapshots up to that
step are combined with least-squares coefficients into the target EMA, which is sampled with the same settings as the
in-loop eval (same seed, labels and noise for every target, so the FID differences are not sampling noise). Prints a table
next to the window JiT's fixed EMAs amount to at that step, writes {run}/phema/posthoc_{iter}.json and a plot of FID vs
sigma_rel, and saves the best EMA as a checkpoint for eval_jit.py (the reconstructed weights in the EMA1 slot):
{run}/phema/posthoc_{iter}_std{sigma}.pt.

$ python posthoc_ema.py --run=runs/in1k_128/jit-b16_icl0/<run> --data_dir=/data/in1k_128
$ python posthoc_ema.py --run=... --data_dir=... --at_iter=125100 --stds=0.05,0.1,0.15      # an earlier snapshot step
$ python eval_jit.py --ckpt=runs/.../phema/posthoc_125100_std0.100.pt --data_dir=... --cfgs=1.0:4.0:0.25   # then a CFG sweep
"""

import os
import re
import json
from contextlib import nullcontext

import numpy as np
import torch

from jit_common import build_denoiser, generate_and_score
from fid import InceptionMetrics, load_ref_stats
from phema import solve_coefficients, fixed_ema_std
from util.progress import bar
import plot_style as ps

# -----------------------------------------------------------------------------
run = '' # run directory (with phema/ snapshots)
data_dir = '' # prepared dataset dir, for the FID reference stats
stds = '0.03,0.05,0.07,0.1,0.15,0.2,0.3' # target sigma_rel values; well inside the tracked pair's range reconstructs best
at_iter = -1 # snapshot step to build the EMAs at; -1 = the latest
fid_num_images = 10000
cfg = 1.0
interval_min = 0.1
interval_max = 1.0
sampling_method = 'heun'
sampling_eta = 1.0
num_sampling_steps = 50
gen_batch_size = 256
save_best = True # write the best EMA as a checkpoint for eval_jit.py
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

assert run and data_dir, 'pass --run and --data_dir'
snap_dir = os.path.join(run, 'phema')
pattern = re.compile(r'phema-(\d+)-([\d.]+)\.pt$')
snaps = sorted((int(m.group(1)), float(m.group(2)), os.path.join(snap_dir, f))
               for f in (os.listdir(snap_dir) if os.path.isdir(snap_dir) else []) if (m := pattern.match(f)))
assert snaps, f'no power-EMA snapshots in {snap_dir} (train with --phema_stds)'
steps = sorted({t for t, _, _ in snaps})
at = steps[-1] if at_iter < 0 else at_iter
assert at in steps, f'no snapshots at step {at}; snapshot steps: {steps}'
use = [s for s in snaps if s[0] <= at]
targets = [float(v) for v in stds.split(',') if v]
coef = solve_coefficients(np.array([s[0] for s in use]), np.array([s[1] for s in use]), at, targets) # [len(use), len(targets)]
tracked = sorted({s for _, s, _ in use})
print(f"{run}: {len(use)} snapshots up to step {at} (power EMAs {tracked}); building sigma_rel {targets}")
for k, s in enumerate(targets):
    if s < min(tracked) * 0.6 or s > max(tracked) * 3:
        print(f"  sigma_rel {s}: far outside the tracked {tracked} (sum |coef| {np.abs(coef[:, k]).sum():.1f}): reconstruction is approximate")

# one pass over the snapshots, accumulating every target (float32, on CPU)
acc, meta = None, None
for i, (t, s, path) in enumerate(bar(use, f'reading {len(use)} snapshots')):
    snap = torch.load(path, map_location='cpu', weights_only=False)
    sd = snap['model']
    params = set(snap['params'])
    if acc is None:
        meta = snap
        acc = [{n: torch.zeros_like(v, dtype=torch.float32) for n, v in sd.items() if n in params} for _ in targets]
    for k in range(len(targets)): # parameters: the weighted combination of the snapshots' EMAs
        c = float(coef[i, k])
        for n in params:
            acc[k][n].add_(sd[n].float(), alpha=c)
    if t == at: # buffers (not averaged in training, e.g. the token allocation's statistics): the live model's at the target step
        for k in range(len(targets)):
            acc[k].update({n: v.clone() for n, v in sd.items() if n not in params})
    del snap, sd

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if not compile:
    torch._dynamo.config.disable = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' or dtype == 'float32') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
denoiser = build_denoiser(**meta['denoiser_args']).to(device).eval()
if compile:
    denoiser.net.compile()
denoiser.time_schedule = 'uniform'
ref_path = os.path.join(data_dir, 'fid_stats_jit.npz')
ref_stats = load_ref_stats(ref_path if os.path.exists(ref_path) else os.path.join(data_dir, 'fid_stats.npz'))
num_classes = meta['denoiser_args']['num_classes']
metrics = InceptionMetrics(device)

rows = []
for k, s in enumerate(targets):
    denoiser.load_state_dict(acc[k])
    out = generate_and_score(denoiser, metrics, ref_stats, fid_num_images, num_classes, device, ctx, cfg=cfg,
                             interval=(interval_min, interval_max), steps=num_sampling_steps, method=sampling_method,
                             batch_size=gen_batch_size, eta=sampling_eta, desc=f'sigma_rel {s} ({k + 1}/{len(targets)}), FID-{fid_num_images / 1000:g}k')
    rows.append({'std': s, 'fid': out['fid'], 'is': out['is'], 'coef_abs_sum': float(np.abs(coef[:, k]).sum())})

best = min(rows, key=lambda r: r['fid'])
ema1, ema2 = meta['denoiser_args'].get('ema_decay1', 0.9999), meta['denoiser_args'].get('ema_decay2', 0.9996)
print(f"\nstep {at} (epoch {meta['epoch'] * at / meta['t']:.1f}), FID-{fid_num_images / 1000:g}k at cfg {cfg}:")
print(f"  {'sigma_rel':>9}  {'FID':>8}  {'IS':>6}")
for r in rows:
    print(f"  {r['std']:>9g}  {r['fid']:8.2f}  {r['is']:6.1f}" + ('   <- best' if r is best else ''))
fmt = lambda d: f"{fixed_ema_std(d, at):.3f}" if fixed_ema_std(d, at) < 1 else "> 1 (a window longer than the run so far)"
print(f"for reference, JiT's fixed EMAs at this step: {ema1} ~ sigma_rel {fmt(ema1)}, {ema2} ~ {fmt(ema2)}")

res = {'run': run, 'at_iter': at, 'snapshots': [(t, s) for t, s, _ in use], 'rows': rows, 'best_std': best['std'],
       'fixed_ema_std': {str(ema1): float(fixed_ema_std(ema1, at)), str(ema2): float(fixed_ema_std(ema2, at))}, 'config': config}
with open(os.path.join(snap_dir, f'posthoc_{at}.json'), 'w') as f:
    json.dump(res, f, indent=2)
plt = ps.setup()
fig, ax = plt.subplots(figsize=(6, 4), facecolor=ps.SURFACE)
ax.plot([r['std'] for r in rows], [r['fid'] for r in rows], color=ps.BLUE, marker='o')
for d, lab in ((ema1, 'EMA1'), (ema2, 'EMA2')):
    ax.axvline(fixed_ema_std(d, at), color=ps.INK_2, lw=0.8, ls='--')
    ax.annotate(f'{lab} {d}', (fixed_ema_std(d, at), ax.get_ylim()[1]), fontsize=7, color=ps.INK_2, rotation=90, va='top', ha='right')
ax.set_xscale('log')
ps.style_axis(ax, 'sigma_rel (EMA window / training so far)', f'FID-{fid_num_images / 1000:g}k', f'post-hoc EMA at step {at}')
fig.tight_layout()
fig.savefig(os.path.join(snap_dir, f'posthoc_{at}.png'), dpi=120, facecolor=ps.SURFACE)
if save_best:
    path = os.path.join(snap_dir, f"posthoc_{at}_std{best['std']:.3f}.pt")
    torch.save({'denoiser_args': meta['denoiser_args'], 'model_ema1': {n: v for n, v in acc[targets.index(best['std'])].items()},
                'iter_num': at, 'posthoc_std': best['std']}, path)
    print(f"best EMA saved for eval_jit.py: {path}")
print(f"results: {os.path.join(snap_dir, f'posthoc_{at}.json')}")
