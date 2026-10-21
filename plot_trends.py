"""
Training-trend figures from train_jit.py's eval diagnostics, aimed at finding where JiT's sample efficiency is lost.

One run -> {run}/trends/*.png
$ python plot_trends.py runs/in100_128/jit-b16/base
Several runs -> per-run figures plus a comparison figure and table in --out
$ python plot_trends.py runs/in100_128/jit-b16/base runs/in100_128/jit-b16/mask0.5 --out=trends_compare

Reads {run}/diagnostics/diag_*.json (written at every FID eval with diag=True), {run}/config.json and, for the
optimization figure, {run}/train_log.jsonl. The x axis is images seen (iterations x effective batch), so runs with
different batch sizes or datasets line up by training compute rather than by epoch.

Figures:
    convergence.png         FID, eval train / val loss, precision / recall, memorization
    noise_levels.png        held-out loss per noise level relative to the first eval, recent improvement per level, train/val gap
    detail.png              per-band prediction error at low / high noise, sample vs real spectrum
    semantics_locality.png  probe accuracy per block, windowed-attention loss increase, attention distance
    optimization.png        loss, gradient norm, update/weight ratio per parameter group, lr
    attn_res.png            AttnRes: the final layer's mixing weight on the patch embedding per noise level, and over sources (attn_res runs only)
    gated_bottleneck.png    how far the gated full-rank patch path is open, per noise level (gated_bottleneck runs only)
    compare.png / .csv      key curves and latest values across runs
"""

import os
import re
import glob
import json
import argparse

import numpy as np

from plot_style import (SURFACE, INK, INK_2, BLUE, ORANGE, CATEGORICAL, MARKERS, ordinal_blues, setup,
                        style_axis, line, heatmap_style, log_axes, end_labels)


# -----------------------------------------------------------------------------
# loading

def load_run(run_dir):
    diags = []
    for path in glob.glob(os.path.join(run_dir, 'diagnostics', 'diag_*.json')):
        with open(path) as f:
            diags.append(json.load(f))
    diags.sort(key=lambda d: d['iter'])
    config = {}
    if os.path.exists(os.path.join(run_dir, 'config.json')):
        with open(os.path.join(run_dir, 'config.json')) as f:
            config = json.load(f)
    batch = config.get('resolved', {}).get('effective_batch_size')
    logs = {}
    if os.path.exists(os.path.join(run_dir, 'train_log.jsonl')):
        with open(os.path.join(run_dir, 'train_log.jsonl')) as f:
            for row in f:
                r = json.loads(row)
                logs[r['iter']] = r # a resumed run may repeat iterations: keep the latest
    logs = [logs[k] for k in sorted(logs)]
    scale = (lambda it: it * batch / 1e6) if batch else (lambda it: float(it))
    return {
        'dir': run_dir, 'name': '/'.join(os.path.abspath(run_dir).rstrip(os.sep).split(os.sep)[-3:]), 'diags': diags, 'config': config, 'logs': logs,
        'x': np.array([scale(d['iter']) for d in diags]), 'xlabel': 'images seen (M)' if batch else 'iteration', 'scale': scale,
    }

def scalar(run, key):
    return np.array([d['scalars'].get(key, np.nan) for d in run['diags']], dtype=float)

def has(values):
    return values.size and np.isfinite(values).any()

def probes(run):
    # {(block, t): [acc per eval]} for knn and linear
    out = {'knn': {}, 'linear': {}}
    for i, d in enumerate(run['diags']):
        for key, r in d.get('probes', {}).items():
            m = re.match(r'block(\d+)_t([\d.]+)', key)
            b, t = int(m.group(1)), float(m.group(2))
            for kind in ('knn', 'linear'):
                out[kind].setdefault((b, t), [np.nan] * len(run['diags']))[i] = r[kind]
    return out

def tlabels(ts):
    return [f't={t:.3f}' for t in ts]

def eval_labels(run):
    return [f'{x:.3g}' for x in run['x']]

VERBOSE = True

def save(fig, path):
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    if VERBOSE:
        print(f"wrote {path}")
    return path

def note(ax, text):
    ax.axis('off')
    ax.text(0.5, 0.5, text, ha='center', va='center', color=INK_2, fontsize=9, wrap=True)

# -----------------------------------------------------------------------------
# single-run figures

def fig_convergence(run, out):
    plt = setup()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    x, xl = run['x'], run['xlabel']
    ax = axes[0, 0]
    fid = scalar(run, 'val/fid')
    line(ax, x, fid)
    log_axes(ax, x=True, y=True)
    style_axis(ax, xl, 'FID-10k (EMA, CFG 1)', 'FID vs training (log-log: a straight line is steady power-law progress)')
    ax = axes[0, 1]
    tr, va = scalar(run, 'eval/train_loss'), scalar(run, 'eval/val_loss')
    if has(tr):
        line(ax, x, tr, BLUE, 'o', 'train')
        if has(va):
            line(ax, x, va, ORANGE, 's', 'held-out')
        log_axes(ax)
        end_labels(ax, x, [tr, va] if has(va) else [tr], ['train', 'held-out'][:2 if has(va) else 1])
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
        style_axis(ax, xl, 'flow-matching loss (fixed noise)', 'eval train / held-out loss')
    else:
        note(ax, 'no eval/train_loss in these diagnostics (older run)')
    ax = axes[1, 0]
    p, r = scalar(run, 'val/precision'), scalar(run, 'val/recall')
    if has(p):
        line(ax, x, p, BLUE, 'o', 'precision (fidelity)')
        line(ax, x, r, ORANGE, 's', 'recall (coverage)')
        log_axes(ax); ax.set_ylim(0, 1)
        end_labels(ax, x, [p, r], ['precision', 'recall'])
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
        style_axis(ax, xl, 'fraction', 'precision / recall (recall lagging precision = coverage-limited)')
    else:
        note(ax, 'no precision / recall')
    ax = axes[1, 1]
    nn = scalar(run, 'mem/nn_ratio')
    if has(nn):
        line(ax, x, nn)
        ax.axhline(1.0, color=INK_2, linewidth=1, linestyle='--')
        ax.annotate('held-out real images', (x[0], 1.0), xytext=(4, 4), textcoords='offset points', color=INK_2, fontsize=8)
        dup = scalar(run, 'mem/near_dup_rate')
        ax.text(0.98, 0.06, f'latest near-duplicate rate {dup[-1]:.1%} (novel images: 1%)', transform=ax.transAxes,
                ha='right', color=INK, fontsize=8)
        ax.margins(y=0.15)
        log_axes(ax)
        style_axis(ax, xl, 'NN distance to train / held-out', 'memorization (approaching 1 or below: copying training images)')
    else:
        note(ax, 'no memorization stats')
    fig.suptitle(f"{run['name']}: convergence", color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save(fig, os.path.join(out, 'convergence.png'))

def fig_noise_levels(run, out):
    plt = setup()
    from matplotlib.colors import TwoSlopeNorm
    diags = run['diags']
    ts = diags[-1]['t']
    val = np.array([d['passes']['val']['v_loss'] for d in diags]).T # [t, eval]
    train = np.array([d['passes']['train']['v_loss'] for d in diags]).T
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=SURFACE, gridspec_kw={'width_ratios': [1.3, 0.8, 1.3]})
    ax = axes[0]
    rel = val / val[:, :1]
    im = ax.imshow(rel, aspect='auto', cmap='Blues', vmin=min(0.0, rel.min()), vmax=max(1.0, rel.max()))
    heatmap_style(ax, im, fig, eval_labels(run), tlabels(ts), 'held-out loss / first eval (dark = little improvement)', 'fraction remaining')
    ax.set_xlabel(run['xlabel'], color=INK_2, fontsize=8)
    ax = axes[1]
    if len(diags) >= 2:
        # % loss decrease per doubling of images seen, between the last two evals
        x0, x1 = run['x'][-2], run['x'][-1]
        rate = (1 - (val[:, -1] / val[:, -2]) ** (1 / np.log2(x1 / x0))) * 100
        ax.barh(range(len(ts)), rate, color=BLUE, height=0.7)
        ax.set_yticks(range(len(ts)), tlabels(ts), fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color=INK_2, linewidth=0.8)
        style_axis(ax, '% loss decrease per 2x images', '', 'recent improvement per noise level')
        ax.grid(axis='y', visible=False)
    else:
        note(ax, 'needs at least two evals')
    ax = axes[2]
    gap = (val - train) / val
    lim = max(1e-6, np.abs(gap).max())
    im = ax.imshow(gap, aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim))
    heatmap_style(ax, im, fig, eval_labels(run), tlabels(ts), 'train/held-out gap: (val - train) / val (red = memorizing there)', 'relative gap')
    ax.set_xlabel(run['xlabel'], color=INK_2, fontsize=8)
    fig.suptitle(f"{run['name']}: where learning happens (small t = high noise)", color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, os.path.join(out, 'noise_levels.png'))

def band_error(run, t_select, key='freq_rel_err'):
    ts = np.array(run['diags'][-1]['t'])
    sel = t_select(ts)
    return np.array([np.array(d['passes']['val'][key])[sel].mean(0) for d in run['diags']]).T, ts[sel] # [band, eval]

def fig_detail(run, out):
    plt = setup()
    from matplotlib.colors import TwoSlopeNorm
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8), facecolor=SURFACE)
    for ax, (name, sel) in zip(axes[:2], (('low noise', lambda ts: ts >= 0.8), ('high noise', lambda ts: ts <= 0.1))):
        err, tsel = band_error(run, sel)
        im = ax.imshow(err, aspect='auto', cmap='Blues', vmin=0, vmax=1)
        bands = [f'band {k}' for k in range(err.shape[0])]
        heatmap_style(ax, im, fig, eval_labels(run), bands, f'relative error by band, {name} (t={", ".join(f"{t:.2f}" for t in tsel)})', 'error / signal power')
        ax.set_xlabel(run['xlabel'], color=INK_2, fontsize=8)
    # model vs passing the noisy input through, at low noise: red = worse than copying, i.e. detail it was given is lost
    ax = axes[2]
    with_copy = [d for d in run['diags'] if 'freq_rel_err_copy' in d['passes']['val']]
    if with_copy:
        sub = dict(run, diags=with_copy, x=np.array([run['scale'](d['iter']) for d in with_copy]))
        err, tsel = band_error(sub, lambda ts: ts >= 0.8)
        copy, _ = band_error(sub, lambda ts: ts >= 0.8, 'freq_rel_err_copy')
        ratio = np.log2(err / copy)
        lim = max(1e-6, np.abs(ratio).max())
        im = ax.imshow(ratio, aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim))
        heatmap_style(ax, im, fig, eval_labels(sub), [f'band {k}' for k in range(ratio.shape[0])],
                      f'model vs copying the input, low noise (t={", ".join(f"{t:.2f}" for t in tsel)}; red = worse)', 'log2 error ratio')
        ax.set_xlabel(run['xlabel'], color=INK_2, fontsize=8)
    else:
        note(ax, 'no copy baseline (diagnostics from before it existed)')
    ax = axes[3]
    nb = sum(1 for k in run['diags'][-1]['scalars'] if k.startswith('spectrum_log_ratio/'))
    if nb:
        spec = np.array([[d['scalars'][f'spectrum_log_ratio/band{k}'] for d in run['diags']] for k in range(nb)])
        lim = max(1e-6, np.abs(spec).max())
        im = ax.imshow(spec, aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim))
        heatmap_style(ax, im, fig, eval_labels(run), [f'band {k}' for k in range(nb)], 'samples vs real spectrum, log10 (blue = too little power)', 'log10 ratio')
        ax.set_xlabel(run['xlabel'], color=INK_2, fontsize=8)
    else:
        note(ax, 'no spectrum stats')
    fig.suptitle(f"{run['name']}: fine detail (band 0 = coarsest, last band = finest)", color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, os.path.join(out, 'detail.png'))

def fig_attn_res(run, out):
    # how much the final layer reads the patch embedding directly, per noise level, and its weights over all sources
    diags = [d for d in run['diags'] if 'attn_res_alpha' in d['passes']['val']]
    if not diags:
        return None
    plt = setup()
    ts = diags[-1]['t']
    alpha = [np.array(d['passes']['val']['attn_res_alpha'], dtype=float) for d in diags] # each [t, rows, sources]
    emb = np.array([a[:, -1, 0] for a in alpha]).T # [t, eval]
    sub = dict(run, diags=diags, x=np.array([run['scale'](d['iter']) for d in diags]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE, gridspec_kw={'width_ratios': [1.2, 1]})
    ax = axes[0]
    im = ax.imshow(emb, aspect='auto', cmap='Blues', vmin=0, vmax=max(1e-6, np.nanmax(emb)))
    heatmap_style(ax, im, fig, eval_labels(sub), tlabels(ts), "final layer's weight on the patch embedding, by noise level", 'mixing weight')
    ax.set_xlabel(run['xlabel'], color=INK_2, fontsize=8)
    ax = axes[1]
    last = alpha[-1][:, -1, :] # [t, sources] at the latest eval
    picks = sorted({0, len(ts) // 2, len(ts) - 1})
    for color, i in zip(ordinal_blues(len(picks)), picks):
        line(ax, np.arange(last.shape[1]), last[i], color, label=f't={ts[i]:.2f}')
    ax.set_ylim(0, 1.1 * max(1e-6, np.nanmax(last)))
    ax.ticklabel_format(axis='y', useOffset=False)
    ax.set_xlabel('source (0 = patch embedding, then in depth order)', color=INK_2, fontsize=8)
    ax.set_ylabel('mixing weight', color=INK_2, fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"final layer's weights over sources (iter {diags[-1]['iter']:,})", color=INK, fontsize=9, loc='left')
    fig.suptitle(f"{run['name']}: attention residuals", color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, os.path.join(out, 'attn_res.png'))

def fig_gated_bottleneck(run, out):
    # RMS of the gated full-rank patch embedding relative to the bottleneck path, per noise level (0 = gate shut)
    diags = [d for d in run['diags'] if 'bypass_ratio' in d['passes']['val']]
    if not diags:
        return None
    plt = setup()
    ts = diags[-1]['t']
    ratio = np.array([np.array(d['passes']['val']['bypass_ratio'])[:, 0] for d in diags]).T # [t, eval]
    sub = dict(run, diags=diags, x=np.array([run['scale'](d['iter']) for d in diags]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE, gridspec_kw={'width_ratios': [1.3, 1]})
    ax = axes[0]
    im = ax.imshow(ratio, aspect='auto', cmap='Blues', vmin=0, vmax=max(1e-6, ratio.max()))
    heatmap_style(ax, im, fig, eval_labels(sub), tlabels(ts), 'full-rank path / bottleneck path (RMS), by noise level', 'ratio')
    ax.set_xlabel(run['xlabel'], color=INK_2, fontsize=8)
    ax = axes[1]
    picks = sorted({0, len(diags) // 2, len(diags) - 1})
    for color, i in zip(ordinal_blues(len(picks)), picks):
        line(ax, ts, ratio[:, i], color, label=f"iter {diags[i]['iter']:,}")
    ax.set_xlabel('t (1 = clean)', color=INK_2, fontsize=8)
    ax.set_ylabel('full-rank / bottleneck (RMS)', color=INK_2, fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('gate opening vs noise level', color=INK, fontsize=9, loc='left')
    fig.suptitle(f"{run['name']}: gated bottleneck", color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, os.path.join(out, 'gated_bottleneck.png'))

def fig_semantics_locality(run, out):
    plt = setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=SURFACE)
    x, xl = run['x'], run['xlabel']
    pr = probes(run)
    t_probe = max(t for _, t in pr['linear']) if pr['linear'] else None
    for ax, kind in zip(axes[0], ('linear', 'knn')):
        if t_probe is None:
            note(ax, 'no probes'); continue
        blocks = sorted(b for b, t in pr[kind] if t == t_probe)
        for color, b in zip(ordinal_blues(len(blocks)), blocks):
            line(ax, x, np.array(pr[kind][(b, t_probe)], dtype=float), color, 'o', f'block {b}')
        log_axes(ax)
        end_labels(ax, x, [np.array(pr[kind][(b, t_probe)], dtype=float) for b in blocks], [f'block {b}' for b in blocks])
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
        style_axis(ax, xl, 'top-1 accuracy (%)', f"{'linear' if kind == 'linear' else 'k-NN'} probe at t={t_probe:g} (null label)")
    ax = axes[1, 0]
    wins = {k: v for k, v in (run['diags'][-1].get('window_loss_increase') or {}).items()}
    ts = run['diags'][-1]['t']
    if wins:
        tiles = sorted(wins, key=lambda k: int(k[4:]))
        for color, k in zip(ordinal_blues(len(tiles)), tiles):
            y = np.array([(d.get('window_loss_increase') or {}).get(k, [np.nan])[0] * 100 for d in run['diags']])
            line(ax, x, y, color, 'o', f'{k[4:]}x{k[4:]} tiles', f'{k[4:]}x{k[4:]}')
        log_axes(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
        style_axis(ax, xl, 'held-out loss increase (%)', f'loss increase when attention is confined to tiles, t={ts[0]:.3f} (high noise)')
    else:
        note(ax, 'no windowed-attention ablation (diag_window_tiles empty)')
    ax = axes[1, 1]
    if 'attn_dist' in run['diags'][-1]['passes']['val']:
        dist = np.array([np.array(d['passes']['val']['attn_dist'])[0] for d in run['diags']]).T # [block, eval] at the highest noise
        im = ax.imshow(dist, aspect='auto', cmap='Blues', vmin=0, vmax=max(1.0, dist.max()))
        heatmap_style(ax, im, fig, eval_labels(run), [f'block {b + 1}' for b in range(dist.shape[0])],
                      f'attention distance / uniform at t={ts[0]:.3f} (dark = more global)', 'relative distance')
        ax.set_xlabel(xl, color=INK_2, fontsize=8)
    else:
        note(ax, 'no attention distance (older run)')
    fig.suptitle(f"{run['name']}: semantics and locality", color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, os.path.join(out, 'semantics_locality.png'))

def group_order(name):
    m = re.match(r'blocks\.(\d+)\.(\w+)', name)
    if m:
        return (1, int(m.group(1)), ['attn', 'mlp', 'adaLN', 'norm'].index(m.group(2)) if m.group(2) in ('attn', 'mlp', 'adaLN', 'norm') else 9)
    return (0 if 'embed' in name else 2, 0, name)

def fig_optimization(run, out):
    logs = run['logs']
    if not logs:
        return None
    plt = setup()
    x = np.array([run['scale'](r['iter']) for r in logs])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=SURFACE, gridspec_kw={'height_ratios': [1, 1.6]})
    ax = axes[0, 0]
    loss = np.array([r['loss'] for r in logs])
    k = max(1, len(loss) // 50)
    smooth = np.convolve(loss, np.ones(k) / k, mode='valid')
    ax.plot(x, loss, color='#b7d3f6', linewidth=1)
    ax.plot(x[k - 1:], smooth, color=BLUE, linewidth=2)
    log_axes(ax, x=True, y=True)
    style_axis(ax, run['xlabel'], 'training loss (one batch, light) / smoothed', 'training loss')
    ax = axes[0, 1]
    gl = [r for r in logs if 'grad_norm' in r]
    if gl:
        gx = np.array([run['scale'](r['iter']) for r in gl])
        line(ax, gx, np.array([np.sqrt(sum(v ** 2 for v in r['grad_norm'].values())) for r in gl]), marker='')
        log_axes(ax)
        style_axis(ax, run['xlabel'], 'gradient norm (total)', 'gradient norm')
        ax = axes[1, 0]
        groups = sorted(gl[-1]['update_ratio'], key=group_order)
        cols = np.array_split(np.arange(len(gl)), min(60, len(gl)))
        ratio = np.array([[np.mean([gl[i]['update_ratio'].get(g, np.nan) for i in c]) for c in cols] for g in groups])
        im = ax.imshow(np.log10(np.clip(ratio, 1e-12, None)), aspect='auto', cmap='Blues')
        xt = [f"{np.mean([run['scale'](gl[i]['iter']) for i in c]):.3g}" for c in cols]
        step = max(1, len(xt) // 12)
        heatmap_style(ax, im, fig, [v if j % step == 0 else '' for j, v in enumerate(xt)], groups,
                      'update / weight ratio per group (light = barely updating; adaLN starts at zero, so its early ratio is large)', 'log10 ratio')
        ax.set_xlabel(run['xlabel'], color=INK_2, fontsize=8)
    else:
        note(ax, 'no gradient stats (diag_opt_stats off)')
        note(axes[1, 0], '')
    ax = axes[1, 1]
    line(ax, x, np.array([r['lr'] for r in logs]), marker='')
    style_axis(ax, run['xlabel'], 'learning rate', 'learning rate')
    fig.suptitle(f"{run['name']}: optimization", color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, os.path.join(out, 'optimization.png'))

# -----------------------------------------------------------------------------
# comparison across runs

def compare(runs, out):
    plt = setup()
    metrics = [
        ('FID-10k (EMA, CFG 1)', lambda r: scalar(r, 'val/fid'), True),
        ('held-out loss', lambda r: scalar(r, 'eval/val_loss') if has(scalar(r, 'eval/val_loss'))
         else np.array([np.mean(d['passes']['val']['v_loss']) for d in r['diags']]), True),
        ('best linear probe, top-1 %', lambda r: np.array([max((v['linear'] for v in d.get('probes', {}).values()), default=np.nan) for d in r['diags']]), False),
        ('per-patch attention: loss increase %, highest noise', lambda r: np.array([(d.get('window_loss_increase') or {}).get('tile1', [np.nan])[0] * 100 for d in r['diags']]), False),
        ('finest-band error at low noise', lambda r: band_error(r, lambda ts: ts >= 0.8)[0][-1] if r['diags'] else np.array([]), False),
        ('recall', lambda r: scalar(r, 'val/recall'), False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=SURFACE)
    table = []
    for ax, (name, fn, logy) in zip(axes.flatten(), metrics):
        for i, r in enumerate(runs):
            y = fn(r)
            if not has(y):
                continue
            label = r['name'].split('/')[-1] if i < len(CATEGORICAL) else None
            line(ax, r['x'], y, CATEGORICAL[i % len(CATEGORICAL)], MARKERS[i % len(MARKERS)], label,
                 label if len(runs) <= 4 else None)
        log_axes(ax, x=True, y=logy)
        style_axis(ax, runs[0]['xlabel'], name, name)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    for r in runs:
        row = {'run': r['name'], 'images_M' if r['xlabel'].startswith('images') else 'iter': r['x'][-1] if len(r['x']) else np.nan}
        for name, fn, _ in metrics:
            y = fn(r)
            row[name] = float(y[-1]) if has(y) else np.nan
        table.append(row)
    fig.suptitle('run comparison (marker shape + label identify runs)', color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, os.path.join(out, 'compare.png'))
    keys = list(table[0].keys())
    with open(os.path.join(out, 'compare.csv'), 'w') as f:
        f.write(','.join(f'"{k}"' for k in keys) + '\n')
        for row in table:
            f.write(','.join(f'"{row[k]}"' if isinstance(row[k], str) else f'{row[k]:.4g}' for k in keys) + '\n')
    print(f"wrote {os.path.join(out, 'compare.csv')}")
    print('\nlatest values:')
    for row in table:
        print('  ' + ', '.join(f'{k}: {v:.4g}' if not isinstance(v, str) else v for k, v in row.items()))

def make_run_figures(run_dir):
    """
    Writes every single-run figure for run_dir into {run_dir}/trends and returns {name: path} (empty if the run has no
    diagnostics yet). Used by train_jit.py after each eval.
    """
    run = load_run(run_dir)
    if not run['diags']:
        return {}
    out = os.path.join(run_dir, 'trends')
    os.makedirs(out, exist_ok=True)
    figures = {}
    for name, fn in (('convergence', fig_convergence), ('noise_levels', fig_noise_levels), ('detail', fig_detail),
                     ('semantics_locality', fig_semantics_locality), ('optimization', fig_optimization),
                     ('attn_res', fig_attn_res), ('gated_bottleneck', fig_gated_bottleneck)):
        path = fn(run, out)
        if path:
            figures[name] = path
    return figures

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', help='run directories (containing diagnostics/)')
    parser.add_argument('--out', type=str, default='', help='output dir for the comparison (default: {first run}/trends)')
    args = parser.parse_args()
    runs = [load_run(d) for d in args.runs]
    for r in runs:
        if not r['diags']:
            print(f"{r['dir']}: no diagnostics/diag_*.json, skipping")
            continue
        print(f"{r['name']}: {len(r['diags'])} evals, {len(r['logs'])} training log rows")
        make_run_figures(r['dir'])
    runs = [r for r in runs if r['diags']]
    if len(runs) >= 2:
        out = args.out or os.path.join(runs[0]['dir'], 'trends')
        os.makedirs(out, exist_ok=True)
        compare(runs, out)
