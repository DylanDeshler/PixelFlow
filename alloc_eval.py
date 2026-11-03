"""
Evaluation of a two-token-size JiT's allocation (train_jit.py --token_sizes, model_jit.TokenAllocation), at every eval:
  matched loss       held-out images with the same noise and t for both token sizes: each path's v-loss per t and the
                     matched relative gain (L_c - L_f) / L_c with its paired standard error, next to the gain from the running
                     training statistics (different images) that drives the allocation
  policy             the training draw's fine share per t (Monte Carlo of the Gumbel top-k over the training t sampler) and
                     the per-path sample counts behind the statistics
  FID by allocation  FID-N (default 1k) of the EMA weights under fixed allocations, all from the same labels and noise
                     (common random numbers: differences between allocations are far less noisy than N alone suggests;
                     the absolute values are biased upward, so compare them with each other, not with FID-10k):
                       coarse / fine    every call on one size
                       budget[:f]       the learned ranking with fine tokens on a share f of the calls (default: the
                                        training share), the sampling-compute knob
                       window:lo-hi     fine tokens only for t in [lo, hi): what fine tokens are worth in FID per t range,
                                        to set against the loss gain there
                     each with its sampling cost relative to all-coarse (analytic FLOPs over the calls actually made)
  samples            fixed labels and noise under every allocation (rows), the same at every eval: allocation_samples_{iter}.png
Appends one record per eval to {out_dir}/allocation.jsonl; plots allocation_{iter}.png (this eval) and allocation_trends.png
(every eval so far) into {out_dir}/diagnostics.
"""

import os
import json
import math
from contextlib import contextmanager

import numpy as np
import torch

from data_jit import to_model_input
import plot_style as ps

DEFAULT_CONDITIONS = 'coarse,budget,budget:0.5,fine,window:0-0.25,window:0.25-0.5,window:0.5-0.75,window:0.75-1'


def gflop_fwd(n_tokens, hidden, depth, mlp_ratio=4.0):
    # per image, forward: per block 2 x (4 d^2 attention projections + 2 x mlp_ratio d^2 MLP) x N, + 2 x 2 N^2 d attention
    return depth * (2 * (4 + 2 * mlp_ratio) * hidden * hidden * n_tokens + 4 * n_tokens * n_tokens * hidden) / 1e9


def parse_conditions(spec, train_frac):
    # 'coarse,budget:0.5,window:0.5-1' -> [(name, infer_frac or None, windows or None)]
    out = []
    for c in (v.strip() for v in spec.split(',') if v.strip()):
        if c == 'coarse':
            out.append((c, 0.0, None))
        elif c == 'fine':
            out.append((c, 1.0, None))
        elif c.startswith('budget'):
            f = float(c.split(':')[1]) if ':' in c else train_frac
            out.append((f'budget{f:g}', f, None))
        elif c.startswith('window:'):
            lo, hi = (float(v) for v in c.split(':')[1].split('-'))
            out.append((f'window{lo:g}-{hi:g}', None, [(lo, hi)]))
        else:
            raise ValueError(f'unknown allocation condition {c}')
    return out


@contextmanager
def allocation(alloc, infer_frac, windows):
    saved = alloc.infer_frac, alloc.fine_windows
    alloc.infer_frac = infer_frac if infer_frac is not None else alloc.infer_frac
    alloc.fine_windows = windows
    alloc.calls = [0, 0]
    try:
        yield
    finally:
        alloc.infer_frac, alloc.fine_windows = saved


@torch.no_grad()
def matched_loss(net, images, labels, ts, noise_scale, t_eps, ctx, device, chunk=256, seed=0):
    """
    Per t: both token sizes on the same images, noise and t (conditional, no label dropout). Returns {'ts', 'loss' [2, T]
    (coarse, fine), 'gain' [T] relative, 'gain_se' [T] (paired standard error of the gain)}.
    """
    was_training = net.training
    net.eval()
    sizes = net.token_sizes
    loss = np.zeros((2, len(ts)))
    gain, gain_se = np.zeros(len(ts)), np.zeros(len(ts))
    g = torch.Generator(device=device)
    for i, t in enumerate(ts):
        g.manual_seed(seed + i)
        per = [[], []]
        for s in range(0, len(images), chunk):
            x = to_model_input(images[s:s + chunk], flip=False, device=device)
            y = torch.from_numpy(labels[s:s + chunk]).to(device)
            tt = torch.full((len(x),), float(t), device=device)
            z = t * x + (1 - t) * torch.randn(x.shape, generator=g, device=device) * noise_scale
            for j, p in enumerate(sizes):
                with ctx:
                    pred = net(z, tt, y, patch=p)
                per[j].append(((x - pred.float()) / max(1 - t, t_eps)).pow(2).mean(dim=(1, 2, 3)))
        lc, lf = torch.cat(per[0]).double(), torch.cat(per[1]).double()
        loss[:, i] = lc.mean().item(), lf.mean().item()
        d = lc - lf
        gain[i] = (d.mean() / lc.mean()).item()
        gain_se[i] = (d.std() / math.sqrt(len(d)) / lc.mean()).item()
    net.train(was_training)
    return {'ts': list(map(float, ts)), 'loss': loss.tolist(), 'gain': gain.tolist(), 'gain_se': gain_se.tolist()}


@torch.no_grad()
def policy_share(alloc, P_mean, P_std, batch, reps=64, seed=0):
    # the training draw (fixed quota, Gumbel top-k in proportion to the weights) simulated over the training t sampler:
    # fine share and sample density per allocation bin
    g = torch.Generator().manual_seed(seed)
    k = max(1, min(batch - 1, round(alloc.fine_frac * batch)))
    fine, tot = torch.zeros(alloc.bins), torch.zeros(alloc.bins)
    for _ in range(reps):
        t = torch.sigmoid(torch.randn(batch, generator=g) * P_std + P_mean).to(alloc.ema.device)
        w = alloc.weights(t).cpu()
        gum = -torch.log(-torch.log(torch.rand(batch, generator=g).clamp(1e-20, 1.0)))
        m = torch.zeros(batch, dtype=torch.bool)
        m[torch.topk(torch.log(w.clamp_min(1e-20)) + gum, k).indices] = True
        b = alloc.bin(t).cpu()
        tot += torch.bincount(b, minlength=alloc.bins).float()
        fine += torch.bincount(b[m], minlength=alloc.bins).float()
    share = torch.where(tot >= 20, fine / tot.clamp_min(1), torch.full_like(tot, float('nan'))) # (too few draws: not shown)
    return share.tolist(), (tot / tot.sum()).tolist()


def raw_gain(alloc):
    # the running statistics' gain before clipping at 0 (what the plots compare with the matched gain), nan without data
    d = alloc.ema[0] - alloc.ema[1]
    if alloc.gain_kind == 'rel':
        d = d / alloc.ema[0].clamp_min(1e-12)
    return torch.where(alloc.valid(), d, torch.full_like(d, float('nan'))).tolist()


def fid_label(n):
    return f'FID-{n / 1000:g}k'


def bin_centres(alloc):
    edges = torch.linspace(-alloc.logit_range, alloc.logit_range, alloc.bins + 1)
    return torch.sigmoid((edges[:-1] + edges[1:]) / 2).tolist()


def set_sampler(model, cfg, interval, steps, method, eta):
    model.cfg_scale, model.cfg_interval, model.steps, model.method, model.eta = cfg, interval, steps, method, eta


@torch.no_grad()
def fid_by_allocation(denoiser, conditions, generate_and_score, metrics, ref_stats, n_images, num_classes, device, ctx,
                      sampler, batch_size, cost):
    # generate_and_score with seed 0 under each allocation: identical labels and noise for every condition
    alloc, rows = denoiser.net.alloc, []
    for i, (name, frac, windows) in enumerate(conditions):
        with allocation(alloc, frac, windows):
            out = generate_and_score(denoiser, metrics, ref_stats, n_images, num_classes, device, ctx, batch_size=batch_size,
                                     desc=f'allocation {name} ({i + 1}/{len(conditions)}), FID-{n_images / 1000:g}k', **sampler)
            nc, nf = alloc.calls
        share = nf / max(1, nc + nf)
        rows.append({'name': name, 'fid': out['fid'], 'fine_calls': share, 'cost': (1 - share) + share * cost})
    return rows


@torch.no_grad()
def samples_by_allocation(denoiser, conditions, labels, sampler, ctx, seed=1234):
    # the same labels and noise under every allocation (and at every eval): [conditions, n, 3, H, W] in [-1, 1]
    alloc, out = denoiser.net.alloc, []
    was_training = denoiser.training
    denoiser.eval()
    set_sampler(denoiser, sampler['cfg'], sampler['interval'], sampler['steps'], sampler['method'], sampler['eta'])
    for name, frac, windows in conditions:
        with allocation(alloc, frac, windows), torch.random.fork_rng(devices=[labels.device] if labels.is_cuda else []):
            torch.manual_seed(seed)
            with ctx:
                out.append(denoiser.generate(labels).float().cpu())
    denoiser.train(was_training)
    return torch.stack(out)


def save_sample_rows(x, row_names, col_names, path):
    from PIL import Image, ImageDraw
    from fid import to_uint8
    R, C, _, H, W = x.shape
    left, top = 150, 12
    canvas = Image.new('RGB', (left + C * W, top + R * H), 'white')
    draw = ImageDraw.Draw(canvas)
    imgs = to_uint8(x.flatten(0, 1)).permute(0, 2, 3, 1).numpy().reshape(R, C, H, W, 3)
    for c, n in enumerate(col_names):
        draw.text((left + c * W + 2, 0), n[: max(4, W // 6)], fill='black')
    for r, n in enumerate(row_names):
        draw.text((2, top + r * H + H // 2 - 5), n, fill='black')
        for c in range(C):
            canvas.paste(Image.fromarray(imgs[r, c]), (left + c * W, top + r * H))
    canvas.save(path)


def plot_eval(rec, path):
    plt = ps.setup()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), facecolor=ps.SURFACE)
    (a, b, c), (d, e, f) = axes
    tc = rec['bin_t']
    ml = rec['matched']
    run = rec['running']
    # (a) loss per path vs t
    for j, (nm, col, mk) in enumerate((('coarse', ps.CATEGORICAL[0], 'o'), ('fine', ps.CATEGORICAL[1], 's'))):
        a.plot(ml['ts'], ml['loss'][j], color=col, marker=mk, ms=4, label=f'{nm} {rec["sizes"][j]}px, held-out (matched)')
        ok = [i for i, n in enumerate(run['count'][j]) if n >= rec['min_count']]
        a.plot([tc[i] for i in ok], [run['ema'][j][i] for i in ok], color=col, ls='--', lw=1, label=f'{nm}, training running mean')
    a.set_yscale('log')
    ps.style_axis(a, 't (1 = clean)', 'v-loss', 'loss per token size')
    a.legend(fontsize=7, frameon=False)
    # (b) gain vs t
    g, se = np.array(ml['gain']), np.array(ml['gain_se'])
    b.fill_between(ml['ts'], g - 2 * se, g + 2 * se, color=ps.CATEGORICAL[0], alpha=0.2, lw=0)
    b.plot(ml['ts'], g, color=ps.CATEGORICAL[0], marker='o', ms=4, label='held-out, matched (±2 s.e.)')
    ok = [i for i, n in enumerate(np.min(run['count'], axis=0)) if n >= rec['min_count']]
    b.plot([tc[i] for i in ok], [run['gain_raw'][i] for i in ok], color=ps.CATEGORICAL[1], ls='--', marker='.',
           label='training statistics (the allocation uses it clipped at 0)')
    b.axhline(0, color=ps.INK_2, lw=0.6)
    ps.style_axis(b, 't', '(L_coarse - L_fine) / L_coarse', 'relative gain of fine tokens')
    b.legend(fontsize=7, frameon=False)
    # (c) allocation: training draw, inference region, t density
    pol = rec['policy']
    c.plot(tc, pol['share'], color=ps.CATEGORICAL[0], marker='o', ms=3, label=f'training: fine share (budget {rec["fine_frac"]:g})')
    c.axhline(rec['fine_frac'], color=ps.CATEGORICAL[0], lw=0.6, ls=':')
    dens = np.array(pol['density'])
    c.bar(tc, dens / max(dens.max(), 1e-12), width=0.02, color=ps.GRID, label='training t density (scaled)')
    inf = rec['inference_fine_t']
    c.plot(inf['t'], inf['fine'], color=ps.CATEGORICAL[1], lw=1.5, drawstyle='steps-mid', label=f'sampling: fine (budget {rec["infer_frac"]:g})')
    ps.style_axis(c, 't', 'share of samples / calls', 'where fine tokens go')
    c.set_ylim(-0.02, 1.05)
    c.legend(fontsize=7, frameon=False)
    # (d) FID value of fine tokens per t window, next to the loss gain
    fids = {r['name']: r for r in rec['fid']}
    wins = [r for r in rec['fid'] if r['name'].startswith('window')]
    if 'coarse' in fids and wins:
        base = fids['coarse']['fid']
        for r in wins:
            lo, hi = (float(v) for v in r['name'][len('window'):].split('-'))
            d.bar((lo + hi) / 2, base - r['fid'], width=(hi - lo) * 0.9, color=ps.CATEGORICAL[2], alpha=0.8)
        d.axhline(0, color=ps.INK_2, lw=0.6)
        ps.style_axis(d, 't window with fine tokens', f'{fid_label(rec["fid_images"])} (coarse) - FID (higher = fine helps)',
                      'FID gain of fine tokens per t window')
        d2 = d.twinx()
        d2.plot(ml['ts'], ml['gain'], color=ps.CATEGORICAL[0], marker='o', ms=3, lw=1)
        d2.set_ylabel('matched loss gain', color=ps.CATEGORICAL[0], fontsize=8)
        d2.tick_params(colors=ps.CATEGORICAL[0], labelsize=7)
    else:
        d.set_axis_off()
    # (e) FID vs sampling cost
    for i, r in enumerate(rec['fid']):
        col = ps.CATEGORICAL[2] if r['name'].startswith('window') else ps.CATEGORICAL[0]
        e.scatter(r['cost'], r['fid'], color=col, s=22, zorder=3)
        e.annotate(r['name'], (r['cost'], r['fid']), fontsize=7, color=ps.INK_2, xytext=(3, 3), textcoords='offset points')
    curve = sorted([r for r in rec['fid'] if not r['name'].startswith('window')], key=lambda r: r['cost'])
    e.plot([r['cost'] for r in curve], [r['fid'] for r in curve], color=ps.CATEGORICAL[0], lw=1)
    ps.style_axis(e, 'sampling cost (x all-coarse, analytic FLOPs)', f'{fid_label(rec["fid_images"])} (same noise and labels)',
                  'FID by allocation')
    # (f) statistics behind the allocation
    for j, (nm, col) in enumerate((('coarse', ps.CATEGORICAL[0]), ('fine', ps.CATEGORICAL[1]))):
        f.plot(tc, np.maximum(run['count'][j], 0.5), color=col, marker='.', label=nm)
    f.axhline(rec['min_count'], color=ps.INK_2, lw=0.6, ls=':')
    f.set_yscale('log')
    ps.style_axis(f, 't (bin centre)', 'training samples seen', 'samples per bin and path')
    f.legend(fontsize=7, frameon=False)
    fig.suptitle(f'token allocation at iter {rec["iter"]} ({rec["sizes"][0]}/{rec["sizes"][1]}px)', color=ps.INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=ps.SURFACE)
    plt.close(fig)
    return path


def plot_trends(history, path):
    plt = ps.setup()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=ps.SURFACE)
    cols = ps.ordinal_blues(len(history))
    for rec, col in zip(history, cols):
        a.plot(rec['matched']['ts'], rec['matched']['gain'], color=col, lw=1.2, label=f'iter {rec["iter"]}')
        b.plot(rec['bin_t'], rec['policy']['share'], color=col, lw=1.2)
    a.axhline(0, color=ps.INK_2, lw=0.6)
    ps.style_axis(a, 't', 'matched relative gain', 'gain of fine tokens over training (dark = later)')
    a.legend(fontsize=6, frameon=False, ncol=2)
    ps.style_axis(b, 't', 'fine share', 'training allocation over training')
    names = [r['name'] for r in history[-1]['fid']]
    for i, nm in enumerate(names):
        pts = [(rec['iter'], r['fid']) for rec in history for r in rec['fid'] if r['name'] == nm]
        c.plot(*zip(*pts), color=ps.CATEGORICAL[i % len(ps.CATEGORICAL)], marker=ps.MARKERS[i % len(ps.MARKERS)], ms=4, lw=1,
               ls='-' if not nm.startswith('window') else '--', label=nm)
    ps.style_axis(c, 'iteration', fid_label(history[-1]['fid_images']), 'FID by allocation over training')
    c.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=ps.SURFACE)
    plt.close(fig)
    return path


@torch.no_grad()
def evaluate_allocation(step, denoiser, raw_net, val_images, val_labels, out_dir, *, generate_and_score, metrics, ref_stats,
                        num_classes, device, ctx, sampler, gen_batch_size, conditions, fid_images, show_labels, class_names,
                        loss_ts, ema_context, show_sampler=None):
    """
    One eval's allocation record. raw_net: the network with the raw weights (matched loss); ema_context(): a context
    manager that swaps in the EMA weights (FID and samples); sampler: generate_and_score's sampler arguments (cfg,
    interval, steps, method, eta) for the FIDs, show_sampler the same for the sample rows (default: sampler).
    Returns (scalars for logging, {name: image path}, the record).
    """
    net = denoiser.net
    alloc = net.alloc
    conds = parse_conditions(conditions, alloc.fine_frac)
    ml = matched_loss(raw_net, val_images, val_labels, loss_ts, denoiser.noise_scale, denoiser.t_eps, ctx, device)
    share, density = policy_share(alloc, denoiser.P_mean, denoiser.P_std, 1024)
    n_c, n_f = ((net.input_size // p) ** 2 for p in net.token_sizes)
    cost = gflop_fwd(n_f, net.hidden_size, len(net.blocks)) / gflop_fwd(n_c, net.hidden_size, len(net.blocks))
    grid = torch.linspace(0.0025, 0.9975, 200, device=alloc.ema.device)
    with allocation(alloc, None, None):
        inf_fine = alloc.fine_mask(grid).float().tolist()
    with ema_context():
        fid_rows = fid_by_allocation(denoiser, conds, generate_and_score, metrics, ref_stats, fid_images, num_classes, device, ctx,
                                     sampler, gen_batch_size, cost)
        samples = samples_by_allocation(denoiser, conds, show_labels, show_sampler or sampler, ctx)
    rec = {'iter': step, 'sizes': list(net.token_sizes), 'fine_frac': alloc.fine_frac, 'infer_frac': alloc.infer_frac,
           'min_count': alloc.min_count, 'fid_images': fid_images, 'bin_t': bin_centres(alloc),
           'running': {'ema': alloc.ema.tolist(), 'count': alloc.count.tolist(), 'gain': alloc.gain().tolist(),
                       'gain_raw': raw_gain(alloc)},
           'policy': {'share': share, 'density': density}, 'inference_fine_t': {'t': grid.tolist(), 'fine': inf_fine},
           'matched': ml, 'fid': fid_rows, 'fine_call_cost': cost}
    hist_path = os.path.join(out_dir, 'allocation.jsonl')
    with open(hist_path, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')
    history = []
    with open(hist_path) as fh:
        for line in fh:
            r = json.loads(line)
            history = [h for h in history if h['iter'] < r['iter']] + [r] # a resumed run's re-done evals replace the old ones
    ddir = os.path.join(out_dir, 'diagnostics')
    os.makedirs(ddir, exist_ok=True)
    images = {'allocation': plot_eval(rec, os.path.join(ddir, f'allocation_{step}.png')),
              'allocation_trends': plot_trends(history, os.path.join(ddir, 'allocation_trends.png'))}
    sp = os.path.join(ddir, f'allocation_samples_{step}.png')
    save_sample_rows(samples, [r['name'] + f' ({r["cost"]:.2f}x)' for r in fid_rows], [class_names[i] for i in show_labels.tolist()], sp)
    images['allocation_samples'] = sp
    log = {f'alloc/fid{fid_images / 1000:g}k_{r["name"]}': r['fid'] for r in fid_rows}
    log.update({f'alloc/cost_{r["name"]}': r['cost'] for r in fid_rows})
    for t, lc, lf, g in zip(ml['ts'], ml['loss'][0], ml['loss'][1], ml['gain']):
        log[f'alloc/val_loss_coarse_t{t:.2f}'] = lc
        log[f'alloc/val_loss_fine_t{t:.2f}'] = lf
        log[f'alloc/val_gain_t{t:.2f}'] = g
    return log, images, rec
