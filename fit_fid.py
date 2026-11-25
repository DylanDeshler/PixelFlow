"""
Projected FID from within-run training curves (the scaling plan): runs use a constant lr after warm-up, so every periodic eval
is a point on one curve, and each run's FID-10k (and FD-DINOv2, when logged) is fitted as

    F(E) = c + A * E^-alpha        E = epochs (evals at E >= min_epoch; warm-up and the first noisy evals excluded)

with c fixed at the pipeline's floor rather than fitted (a free c is poorly identified from a curve this far from it, and
flips the long-range projection around; fixing it keeps projections stable). Floors (FID-10k, these eval settings):
  pixel runs   --pixel_floor: FID-10k of real held-out images (latent_rfid.py 'real': 3.98 on in1k_128)
  latent runs  the VAE's reconstruction FID-10k: {latent_dir}/rfid.json if readable here, else --latent_floor, else the known
               value for SD-VAE ft-EMA at 128px (8.99)
  FD-DINOv2    0 (no floor measured)
The same fit with c = 0 is reported alongside as a sensitivity check (how much the projection depends on the floor).

Backtest (--fit_upto=E): fit only the evals <= E and compare the projection with the later measured evals. On the two long
pixel runs (row 7, class table) a fit on <= 20 epochs predicted epoch 80 within ~1 FID (42.5 vs 41.7, 41.9 vs 42.3) and
epoch 120 within ~0.5; long-range projections lean slightly optimistic (curves flatten a little more than the power law).

Writes {out}.json (fits, projections) and {out}.png: FID against training compute (trunk FLOPs = images x 3 x forward;
--vae_offset adds the VAE's own training compute to latent runs), measured evals as markers, fits as lines, dashed past the
last eval up to the largest --project epoch.

$ python fit_fid.py runs/in1k_128/jit-b16_icl0/<run> runs/in1k_128/latent-b2_p1_mar-kl16/<run> --labels="pixel B,latent B"
$ python fit_fid.py runs/in1k_128/jit-b16_icl0/<run> --fit_upto=20          # backtest
"""

import os
import glob
import json
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('dirs', nargs='+', help='run directories (globs allowed)')
parser.add_argument('--labels', type=str, default='', help='comma list, same order as the runs (default: from the directory)')
parser.add_argument('--fit_upto', type=float, default=0, help='fit only evals at <= this many epochs (backtest); 0 = all')
parser.add_argument('--min_epoch', type=float, default=5)
parser.add_argument('--project', type=str, default='80,200,400,600', help='epochs to project to')
parser.add_argument('--pixel_floor', type=float, default=3.98)
parser.add_argument('--latent_floor', type=float, default=0, help='override the latent floor (0 = rfid.json / known value)')
parser.add_argument('--vae_offset', action='store_true', help="add the VAE's training compute to latent runs on the compute axis")
parser.add_argument('--out', type=str, default='fid_projection')
args = parser.parse_args()

KNOWN_RFID = {'stabilityai/sd-vae-ft-ema': 8.99} # latent_rfid.py on in1k_128 (2026-11-23)
METRICS = {'val/fid': 'FID-10k', 'val/fd_dino': 'FD-DINOv2'}
project = [float(e) for e in args.project.split(',') if e.strip()]
dirs = list(dict.fromkeys(d.rstrip('/') for g in args.dirs for d in (sorted(glob.glob(g)) or [g]))) # command-line order (labels follow it)
labels = [s.strip() for s in args.labels.split(',')] if args.labels else [os.path.join(*d.split('/')[-2:]) for d in dirs]
assert len(labels) == len(dirs), f'{len(labels)} labels for {len(dirs)} runs'


def fit(E, F, c):
    # log(F - c) = log A - alpha log E: least squares in log space (relative error, so large early FIDs don't dominate)
    y = np.log(np.maximum(F - c, 1e-6))
    X = np.stack([np.ones_like(E), np.log(E)], 1)
    (logA, slope), *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = c + np.exp(X @ np.array([logA, slope]))
    return {'c': c, 'A': float(np.exp(logA)), 'alpha': float(-slope), 'rms_log_err': float(np.sqrt(np.mean((np.log(pred) - np.log(F)) ** 2)))}


def predict(p, E):
    return p['c'] + p['A'] * np.asarray(E, dtype=float) ** -p['alpha']


def latent_floor(res):
    if args.latent_floor:
        return args.latent_floor, '--latent_floor'
    ld = res.get('latent_dir') or ''
    path = os.path.join(ld, 'rfid.json')
    if ld and os.path.exists(path):
        return float(json.load(open(path))['rfid']), path
    if res.get('vae') in KNOWN_RFID:
        return KNOWN_RFID[res['vae']], 'known value'
    return None, 'none found'


runs = []
for d, label in zip(dirs, labels):
    cfg = json.load(open(os.path.join(d, 'config.json')))
    res = cfg.get('resolved', {})
    is_latent = 'vae' in res
    if is_latent and not res.get('latent_dir'): # runs from before train_latent.py recorded it: its default location
        short = {'stabilityai/sd-vae-ft-ema': 'sdvae-ft-ema', 'mar-kl16': 'mar-kl16'}.get(res['vae'], '')
        res = dict(res, latent_dir=cfg.get('latent_dir') or (os.path.join(cfg.get('data_dir', ''), 'latents_' + short) if short else ''))
    ipe, batch, fwd = res['iters_per_epoch'], res['effective_batch_size'], res['fwd_gflops']
    offset = res.get('train_pflops_offset', 0.0) if (is_latent and args.vae_offset) else 0.0
    evals = {}
    for f in glob.glob(os.path.join(d, 'diagnostics', 'diag_*.json')):
        r = json.load(open(f))
        for k in METRICS:
            if k in r['scalars']:
                evals.setdefault(k, []).append((r['iter'] / ipe, r['scalars'][k]))
    run = {'dir': d, 'label': label, 'latent': is_latent, 'pflops_per_epoch': ipe * batch * 3 * fwd / 1e6, 'pflops_offset': offset,
           'vae_train_complete': res.get('vae_train_complete', True), 'metrics': {}}
    for k, pts in evals.items():
        pts = np.array(sorted(pts))
        E, F = pts[:, 0], pts[:, 1]
        use = (E >= args.min_epoch) & ((E <= args.fit_upto) if args.fit_upto else True)
        if use.sum() < 3:
            print(f'{label}: {METRICS[k]}: only {use.sum()} evals to fit, skipped')
            continue
        if k == 'val/fid':
            c, src = (latent_floor(res) if is_latent else (args.pixel_floor, '--pixel_floor'))
            if c is None:
                print(f'{label}: no latent floor ({src}); fitting with c = 0 only (pass --latent_floor)')
                c = 0.0
        else:
            c, src = 0.0, 'no floor measured'
        p, p0 = fit(E[use], F[use], c), fit(E[use], F[use], 0.0)
        m = {'epochs': E.tolist(), 'values': F.tolist(), 'fit_upto': float(E[use].max()), 'floor': c, 'floor_source': src,
             'fit': p, 'fit_c0': p0, 'project': {f'{e:g}': float(predict(p, e)) for e in project},
             'project_c0': {f'{e:g}': float(predict(p0, e)) for e in project}}
        later = E > E[use].max()
        if later.any():
            err = predict(p, E[later]) - F[later]
            m['backtest'] = {'epochs': E[later].tolist(), 'measured': F[later].tolist(), 'predicted': predict(p, E[later]).tolist(),
                             'mean_abs_err': float(np.abs(err).mean())}
        run['metrics'][k] = m
    runs.append(run)

for k, name in METRICS.items():
    rows = [r for r in runs if k in r['metrics']]
    if not rows:
        continue
    print(f"\n{name}: F(E) = c + A E^-alpha, c = the pipeline floor (projection with c = 0 in brackets)")
    print(f"  {'run':<28} {'floor':>5} {'alpha':>6} {'last':>13}  " + '  '.join(f'{"@" + format(e, "g"):>13}' for e in project))
    for r in rows:
        m = r['metrics'][k]
        print(f"  {r['label']:<28} {m['floor']:5.2f} {m['fit']['alpha']:6.3f} {m['values'][-1]:6.2f} @{m['epochs'][-1]:<5g}  "
              + '  '.join(f"{m['project'][f'{e:g}']:6.2f} ({m['project_c0'][f'{e:g}']:5.2f})" for e in project))
        if 'backtest' in m:
            b = m['backtest']
            print(f"      backtest (fit <= {m['fit_upto']:g} ep): mean |error| {b['mean_abs_err']:.2f} over {len(b['epochs'])} later evals; "
                  + ', '.join(f"@{e:g} {pr:.1f} vs {me:.1f}" for e, pr, me in list(zip(b['epochs'], b['predicted'], b['measured']))[-3:]))
for r in runs:
    if r['latent'] and args.vae_offset and r['vae_train_complete'] is False:
        print(f"note: {r['label']}: its VAE's training compute is unknown, so its compute axis is a lower bound")

with open(args.out + '.json', 'w') as f:
    json.dump({'project_epochs': project, 'runs': runs}, f, indent=1)

import plot_style as ps
plt = ps.setup()
panels = [k for k in METRICS if any(k in r['metrics'] for r in runs)]
assert len(runs) <= len(ps.CATEGORICAL), f'{len(runs)} runs: at most {len(ps.CATEGORICAL)} per chart'
fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.4), facecolor=ps.SURFACE, squeeze=False)
for a, k in zip(axes[0], panels):
    for i, r in enumerate(runs):
        if k not in r['metrics']:
            continue
        m, col, mk = r['metrics'][k], ps.CATEGORICAL[i], ps.MARKERS[i]
        pf = lambda e: r['pflops_offset'] + np.asarray(e) * r['pflops_per_epoch']
        E = np.array(m['epochs'])
        a.plot(pf(E), m['values'], color=col, marker=mk, ls='none', ms=8, mec=ps.SURFACE, mew=2, label=r['label'], zorder=3)
        e_fit = np.geomspace(args.min_epoch, m['epochs'][-1], 100)
        e_proj = np.geomspace(m['epochs'][-1], max(project + [m['epochs'][-1]]), 100)
        a.plot(pf(e_fit), predict(m['fit'], e_fit), color=col, lw=2, zorder=2)
        a.plot(pf(e_proj), predict(m['fit'], e_proj), color=col, lw=2, ls='--', zorder=2)
        a.annotate(f"{predict(m['fit'], e_proj[-1]):.1f}", (pf(e_proj[-1]), predict(m['fit'], e_proj[-1])), xytext=(5, 0),
                   textcoords='offset points', fontsize=8, color=ps.INK_2, va='center')
        if m['floor']:
            a.axhline(m['floor'], color=col, lw=1, ls=':', alpha=0.7, zorder=1)
    a.set_xscale('log')
    a.set_yscale('log')
    ps.style_axis(a, 'training compute (PFLOPs' + (', incl. VAE training' if args.vae_offset else ', trunk') + ')', METRICS[k],
                  f"{METRICS[k]} vs training compute\nmarkers measured, solid fit, dashed projection to {max(project):g} ep, dotted floor")
    a.legend(fontsize=8, frameon=False, labelcolor=ps.INK_2)
fig.tight_layout()
fig.savefig(args.out + '.png', dpi=130, facecolor=ps.SURFACE)
print(f"\nwrote {args.out}.json and {args.out}.png")
