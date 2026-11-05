"""
FID vs images seen, vs training compute and vs parameters across runs: hardware-independent comparisons of architectures
(wall-clock is printed for reference but depends on the GPU and the implementation).

For each run: the in-loop FID-10k at every eval (diagnostics/diag_<iter>.json), images seen (iter x effective batch), and
the model's trainable parameters and forward GFLOPs per image: from config.json (recorded by train_jit.py since
2026-11-05) or, for older runs, measured by rebuilding the model from its saved architecture (jit_common.model_cost, one
image on CPU). Training compute = images x 3 x forward FLOPs (backward ~2x forward; implementation-only recompute such as
--tail_checkpoint is not counted, it changes nothing about the model).
  images  FID vs images seen (every eval, one line per run): sample efficiency
  flops   FID vs training PFLOPs (every eval): compute efficiency, the fair axis for designs that cost more per image
  params  FID at --at_epoch vs parameters (one point per run)
(--plots picks which, in that order.) Patterns that match nothing are skipped with a note, so one command can list runs
that don't exist yet.
Writes {out}.png and {out}.json, and prints a table (params, forward GFLOPs/img, FID, img/s).

$ python plot_scaling.py --runs='best=runs/.../<best>;tail 2=runs/.../muon_tail2*' --at_epoch=20 --out=runs/plots/tail_scaling
"""

import os
import glob
import json

import torch

from jit_common import build_denoiser, model_cost
import plot_style as ps

# -----------------------------------------------------------------------------
runs = '' # ';'-separated entries, each 'label=pattern' or just a pattern (run directory, globs allowed; run names can contain
          # commas, e.g. tok-16,8). Unlabelled runs are named by what differs between the run names
at_epoch = 20.0 # the parameters plot uses each run's FID at this epoch (nearest eval within half an epoch)
img_size = 128 # for runs whose config.json predates img_size / num_classes being recorded
num_classes = 1000
out = 'scaling' # output path prefix (.png, .json)
plots = 'images,flops,params' # panels, left to right
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

dirs, given = [], {}
for entry in (e.strip() for e in runs.split(';') if e.strip()):
    label, pat = entry.split('=', 1) if '=' in entry else (None, entry)
    hits = sorted(d for d in glob.glob(pat) if os.path.exists(os.path.join(d, 'config.json')))
    if not hits:
        print(f'(skipping {pat}: no run directory with config.json)')
    for j, d in enumerate(h for h in hits if h not in dirs):
        dirs.append(d)
        if label:
            given[d] = label if len(hits) == 1 else f'{label} {j + 1}'
assert dirs, 'pass --runs'

def auto_labels(dirs):
    # tokens of 'model-dir_run-name' that not every run shares; 'base' if none
    toks = [os.path.basename(os.path.dirname(d.rstrip('/'))).split('_') + os.path.basename(d.rstrip('/')).split('_') for d in dirs]
    common = set.intersection(*(set(t) for t in toks)) if len(toks) > 1 else set()
    return [' '.join(t for t in ts if t not in common) or 'base' for ts in toks]

names = [given.get(d, a) for d, a in zip(dirs, auto_labels(dirs))]

rows = []
for d, name in zip(dirs, names):
    cfg = json.load(open(os.path.join(d, 'config.json')))
    res = cfg.get('resolved', {})
    if 'fwd_gflops' in res:
        cost = {k: res[k] for k in ('params', 'fwd_gflops') if k in res}
    else: # rebuild the network from its saved architecture and count (CPU, one image)
        net = build_denoiser(res['model_name'], res.get('img_size', img_size), res.get('num_classes', num_classes),
                             noise_scale=res.get('noise_scale', 1.0), arch=res.get('arch')).net
        cost = model_cost(net, res.get('img_size', img_size), res.get('num_classes', num_classes))
        del net
    batch = res.get('effective_batch_size') or cfg['batch_size'] * cfg.get('gradient_accumulation_steps', 1)
    ipe = res['iters_per_epoch']
    evals = []
    for f in glob.glob(os.path.join(d, 'diagnostics', 'diag_*.json')):
        it = int(f.rsplit('_', 1)[1][:-5])
        fid = json.load(open(f))['scalars'].get('val/fid')
        if fid is not None:
            evals.append((it, fid))
    evals.sort()
    if not evals: # e.g. a run that stopped before its first eval (out of memory)
        print(f'(skipping {d}: no evals yet)')
        continue
    ips = None
    log = os.path.join(d, 'train_log.jsonl')
    if os.path.exists(log): # median img/s after the first half epoch (hardware-dependent, for reference)
        v = sorted(r['imgs_per_sec'] for r in map(json.loads, open(log)) if 'imgs_per_sec' in r and r.get('epoch', 0) > 0.5)
        ips = v[len(v) // 2] if v else None
    pts = [{'iter': it, 'epoch': it / ipe, 'images': it * batch, 'train_pflops': it * batch * 3 * cost['fwd_gflops'] / 1e6, 'fid': fid}
           for it, fid in evals]
    near = min(pts, key=lambda p: abs(p['epoch'] - at_epoch)) if pts else None
    rows.append({'run': d, 'label': name, **cost, 'img_per_sec': ips, 'evals': pts,
                 'fid_at_epoch': near['fid'] if near and abs(near['epoch'] - at_epoch) <= 0.5 else None})

assert rows, 'no run has evals yet'
assert len(rows) <= len(ps.CATEGORICAL), f'{len(rows)} runs: at most {len(ps.CATEGORICAL)} per chart (split them into several charts)'
print(f"{'run':<28} {'params':>8} {'GFLOP/img':>10} {'img/s':>7} " + f"{'FID@' + format(at_epoch, 'g') + 'ep':>10}  evals (epoch: FID)")
for r in rows:
    fe = f"{r['fid_at_epoch']:.2f}" if r['fid_at_epoch'] is not None else '-'
    print(f"{r['label'][:28]:<28} {r['params'] / 1e6:>7.2f}M {r['fwd_gflops']:>10.2f} {r['img_per_sec'] or float('nan'):>7.0f} {fe:>10}  "
          + ', '.join(f"{p['epoch']:g}: {p['fid']:.2f}" for p in r['evals']))

plt = ps.setup()
panels = [p for p in plots.split(',') if p]
assert panels and all(p in ('images', 'flops', 'params') for p in panels), f"plots: images / flops / params, got {plots}"
fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.2), facecolor=ps.SURFACE, squeeze=False)
axes = dict(zip(panels, axes[0]))
curve_x = {'images': lambda p: p['images'] / 1e6, 'flops': lambda p: p['train_pflops']}
for i, r in enumerate(rows):
    if not r['evals']:
        continue
    c, m = ps.CATEGORICAL[i], ps.MARKERS[i]
    for key, fx in curve_x.items():
        if key in axes:
            axes[key].plot([fx(p) for p in r['evals']], [p['fid'] for p in r['evals']], color=c, marker=m, lw=2, ms=8,
                           mec=ps.SURFACE, mew=2, label=r['label'])
    if 'params' in axes and r['fid_at_epoch'] is not None:
        a = axes['params']
        a.plot(r['params'] / 1e6, r['fid_at_epoch'], color=c, marker=m, ls='none', ms=9, mec=ps.SURFACE, mew=2, label=r['label'])
        if len(rows) <= 4: # selective direct labels, in ink (identity is the marker + legend)
            a.annotate(r['label'], (r['params'] / 1e6, r['fid_at_epoch']), xytext=(7, 4), textcoords='offset points', fontsize=8, color=ps.INK_2)
titles = {'images': ('images seen (M)', 'FID-10k', 'FID vs images seen'),
          'flops': ('training compute (PFLOPs: images x 3 x forward FLOPs)', 'FID-10k', 'FID vs training compute'),
          'params': ('trainable parameters (M)', f'FID-10k at epoch {at_epoch:g}', f'FID vs parameters (epoch {at_epoch:g})')}
for key, a in axes.items():
    ps.style_axis(a, *titles[key])
if 'params' in axes: # at least a 10% span, so runs that differ by a few thousand parameters don't look far apart
    xs = [r['params'] / 1e6 for r in rows if r['fid_at_epoch'] is not None]
    if xs:
        mid, half = (max(xs) + min(xs)) / 2, max((max(xs) - min(xs)) * 0.6, 0.05 * max(xs))
        axes['params'].set_xlim(mid - half, mid + half)
if len(rows) >= 2: # one legend, on the first panel (same marker and colour per run in every panel)
    axes[panels[0]].legend(fontsize=8, frameon=False)
fig.tight_layout()
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
fig.savefig(out + '.png', dpi=130, facecolor=ps.SURFACE)
with open(out + '.json', 'w') as f:
    json.dump({'at_epoch': at_epoch, 'rows': rows}, f, indent=2)
print(f"wrote {out}.png, {out}.json")
