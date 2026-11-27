"""
Total-training-compute frontiers for the pixels-vs-DiTo-latents allocation study (queues/study.sh), the cost ledger, the
backtests and the frozen forecasts for B.

Input: {study}/manifest.jsonl, one line per generator run: {"id", "family" ('pixel' or 'latent-<tokenizer level>'),
"width", "data_frac", "run" (run directory), "tokenizer" (checkpoint or null), "tokenizer_pflops", "encode_pflops"}. For every
run it reads config.json (measured forward FLOPs, batch, trunk params, training images; latent runs' train_pflops_offset =
tokenizer training + dataset encoding) and its in-loop guided FID at every eval (val/guided_fid: post-hoc EMA 0.10 with a
small CFG search, the same sample count everywhere; {run}/eval_final holds the finished run's final evaluation, reported in the
ledger). A point = (family, trunk params N, presentations T, training images D, total training compute C = offset + T x 3 x
forward FLOPs, guided FID).

Model, per family (fitted in log-FID, least squares, positive coefficients):
    F(N, T, D) = E + A N^-a + B T^-b + G D^-g
the irreducible part E, a size term, a training-length term and a data term (the latent families' E absorbs their
tokenizer's floor). Frontier at data size D and total budget C: for every family and every candidate width (the tested ones,
or --widths), T = (C - offset_family(D)) / (3 x forward FLOPs(N)); the family's envelope is the best width, the overall
choice the best family. Reported: the allocation (family, width, duration) at each budget on a log grid, the budgets where
the winning family changes (the inflection points), and the measured lower envelope for coverage.

Backtests (refit without some points, predict them): withholding S (the largest width), the late anchors (> half of each
run), and each data size in turn; reported as mean |log FID error| and whether the predicted winner at each withheld
point's budget matches the measured one. Checkpoints of one run are correlated: errors are also averaged per run first.

--predict_width: forecast curves for a held-out width (B = 768) for every family at the full data size, written with a
sha256 to {study}/predictions_w{width}.json before B is trained (study.sh freezes it; later runs compare against it).

$ python frontier.py --study=runs/study
$ python frontier.py --study=runs/study --predict_width=768
"""

import os
import re
import glob
import json
import hashlib
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--study', default='runs/study')
parser.add_argument('--widths', default='', help='candidate widths for the allocation (default: the tested ones)')
parser.add_argument('--predict_width', type=int, default=0)
parser.add_argument('--grid', type=int, default=60, help='budgets on the log grid')
args = parser.parse_args()


def load_eval(d):
    fs = glob.glob(os.path.join(d, 'eval_*.json'))
    if not fs:
        return None
    r = json.load(open(fs[0]))
    f = r.get('final', {})
    re_ = [v['fid'] for k, v in f.items() if isinstance(v, dict) and 'fid' in v]
    return min(re_) if re_ else min(r['fid'].values())


runs, points = [], []
for line in open(os.path.join(args.study, 'manifest.jsonl')):
    m = json.loads(line)
    cfg = json.load(open(os.path.join(m['run'], 'config.json')))
    res = cfg['resolved']
    fwd, batch = res['fwd_gflops'], res['effective_batch_size']
    offset = res.get('train_pflops_offset', 0.0) or 0.0
    N = res.get('params') or res.get('trunk_params')
    D = res.get('train_images')
    hours = 0.0
    log = os.path.join(m['run'], 'train_log.jsonl')
    if os.path.exists(log):
        rows = [json.loads(l) for l in open(log)]
        hours = max((r.get('train_hours', 0) for r in rows), default=0.0)
    run = dict(m, fwd_gflops=fwd, batch=batch, offset_pflops=offset, params=N, train_images=D, hours=hours, points=[])
    import predict_fid
    for e, fid in predict_fid.history(m['run'], 'val/guided_fid'): # the in-loop guided curve (post-hoc EMA 0.10, small CFG search)
        it = int(round(e * predict_fid.FULL_TRAIN / batch))
        p = dict(run=m['id'], family=m['family'], width=m['width'], data_frac=m['data_frac'], N=N, T=it * batch, D=D, it=it,
                 gen_pflops=it * batch * 3 * fwd / 1e6, C=offset + it * batch * 3 * fwd / 1e6, fid=fid)
        run['points'].append(p)
        points.append(p)
    if run['points']:
        last = max(run['points'], key=lambda p: p['it'])
        for p in run['points']:
            p['late'] = p['it'] > 0.5 * last['it']
    run['final_fid'] = load_eval(os.path.join(m['run'], 'eval_final'))
    runs.append(run)
assert points, 'no anchor evaluations yet'
families = sorted({p['family'] for p in points})
print(f"{len(runs)} runs, {len(points)} guided anchor points; families {families}")

# ---- the model ------------------------------------------------------------------------------------------------------------
def predict(theta, N, T, D):
    E, A, a, B, b, G, g = np.exp(theta)
    return E + A * N ** -a + B * T ** -b + G * D ** -g

def fit(pts):
    N = np.array([p['N'] for p in pts], float); T = np.array([p['T'] for p in pts], float)
    D = np.array([p['D'] for p in pts], float); y = np.log([p['fid'] for p in pts])
    def resid(th):
        return np.log(np.maximum(predict(th, N, T, D), 1e-6)) - y
    best = None
    try:
        from scipy.optimize import least_squares
        for init in ([1.0, 5, -1.0, 8, -1.0, 5, -1.0], [2.0, 8, -1.5, 10, -1.2, 8, -1.5], [0.5, 3, -0.7, 6, -0.7, 3, -0.7]):
            r = least_squares(resid, np.array(init), method='trf', max_nfev=20000)
            if best is None or r.cost < best.cost:
                best = r
        return best.x, float(np.sqrt(np.mean(best.fun ** 2)))
    except ImportError: # coarse fallback: random search
        rng = np.random.default_rng(0)
        cand = rng.normal([1, 5, -1, 8, -1, 5, -1], 1.5, size=(20000, 7))
        errs = [np.mean(resid(c) ** 2) for c in cand]
        c = cand[int(np.argmin(errs))]
        return c, float(np.sqrt(min(errs)))

fits = {}
for fam in families:
    pts = [p for p in points if p['family'] == fam]
    if len(pts) < 8:
        print(f"  {fam}: {len(pts)} points, too few to fit (needs >= 8)")
        continue
    th, rms = fit(pts)
    fits[fam] = th
    E, A, a, B, b, G, g = np.exp(th)
    print(f"  {fam}: F = {E:.2f} + {A:.3g} N^-{a:.3f} + {B:.3g} T^-{b:.3f} + {G:.3g} D^-{g:.3f}   (rms log err {rms:.3f}, {len(pts)} points)")

# per family and width: forward FLOPs and params (the tested configurations); per family and data size: the offset
cfgs = {}
for r in runs:
    cfgs.setdefault(r['family'], {})[r['width']] = (r['params'], r['fwd_gflops'], r['batch'])
offsets = {}
for r in runs:
    offsets[(r['family'], r['data_frac'])] = r['offset_pflops']

def allocation(fam_fits, data_frac, D, C, widths=None):
    # best (family, width, T, predicted FID) at total budget C
    best = None
    for fam, th in fam_fits.items():
        off = offsets.get((fam, data_frac))
        if off is None or C <= off:
            continue
        for w, (N, fwd, batch) in cfgs.get(fam, {}).items():
            if widths and w not in widths:
                continue
            T = (C - off) * 1e6 / (3 * fwd)
            if T < batch * 100:
                continue
            f = float(predict(th, N, T, D))
            if best is None or f < best[3]:
                best = (fam, w, T, f)
    return best

widths = [int(w) for w in args.widths.split(',')] if args.widths else None
out = {'fits': {f: np.exp(th).tolist() for f, th in fits.items()}, 'frontier': {}, 'backtests': {}}
for frac in sorted({r['data_frac'] for r in runs}):
    D = max(r['train_images'] for r in runs if r['data_frac'] == frac)
    Cs = [p['C'] for p in points if p['data_frac'] == frac]
    if not Cs or not fits:
        continue
    grid = np.geomspace(min(Cs) / 2, max(Cs) * 4, args.grid)
    alloc, switches, prev = [], [], None
    for C in grid:
        a_ = allocation(fits, frac, D, C, widths)
        if a_ is None:
            continue
        per_family = {}
        for fam in fits:
            b_ = allocation({fam: fits[fam]}, frac, D, C, widths)
            if b_:
                per_family[fam] = {'width': b_[1], 'presentations': b_[2], 'fid': b_[3]}
        alloc.append({'C': float(C), 'family': a_[0], 'width': a_[1], 'presentations': a_[2], 'fid': a_[3], 'per_family': per_family})
        if prev and a_[0] != prev:
            switches.append({'C': float(C), 'from': prev, 'to': a_[0]})
        prev = a_[0]
    # measured envelope: the best measured FID at or below each budget, per family
    meas = {}
    for fam in families:
        pts = sorted((p['C'], p['fid']) for p in points if p['family'] == fam and p['data_frac'] == frac)
        env, best = [], float('inf')
        for c, f in pts:
            best = min(best, f)
            env.append((c, best))
        meas[fam] = env
    out['frontier'][str(frac)] = {'D': D, 'allocation': alloc, 'switches': switches, 'measured_envelope': meas}
    print(f"\ndata {frac:g} (D = {D:,}): winning family changes at " + (', '.join(f"{s['C']:,.0f} PF ({s['from']} -> {s['to']})" for s in switches) or 'no budget in range'))
    for row in alloc[::max(1, len(alloc) // 8)]:
        print(f"  {row['C']:>12,.0f} PF: {row['family']:<14} width {row['width']:>4}  {row['presentations'] / 1e6:8.1f}M images  FID {row['fid']:.2f}")

# ---- backtests --------------------------------------------------------------------------------------------------------------
def backtest(name, held):
    train = [p for p in points if not held(p)]
    test = [p for p in points if held(p)]
    if not test:
        return
    errs, per_run, agree = [], {}, []
    fam_fits = {}
    for fam in families:
        pts = [p for p in train if p['family'] == fam]
        if len(pts) >= 8:
            fam_fits[fam] = fit(pts)[0]
    for p in test:
        if p['family'] not in fam_fits:
            continue
        e = float(np.log(predict(fam_fits[p['family']], p['N'], p['T'], p['D'])) - np.log(p['fid']))
        errs.append(abs(e))
        per_run.setdefault(p['run'], []).append(abs(e))
    # decision agreement: at each withheld point's budget, the predicted winning family vs the best measured family there
    for p in test:
        a_ = allocation(fam_fits, p['data_frac'], p['D'], p['C'])
        near = [q for q in points if q['data_frac'] == p['data_frac'] and abs(np.log(q['C'] / p['C'])) < 0.15]
        if a_ and near:
            agree.append(a_[0] == min(near, key=lambda q: q['fid'])['family'])
    r = {'n': len(errs), 'mean_abs_log_err': float(np.mean(errs)) if errs else None,
         'mean_abs_log_err_per_run': float(np.mean([np.mean(v) for v in per_run.values()])) if per_run else None,
         'decision_agreement': float(np.mean(agree)) if agree else None}
    out['backtests'][name] = r
    print(f"  {name:<22} {r['n']:>4} points: |log err| {r['mean_abs_log_err'] or float('nan'):.3f} (per run {r['mean_abs_log_err_per_run'] or float('nan'):.3f}), "
          f"winner agreement {r['decision_agreement'] if r['decision_agreement'] is not None else float('nan'):.2f}")

print("\nbacktests (refit without, predict):")
wmax = max(r['width'] for r in runs if r['width'] != args.predict_width) if runs else 0
backtest(f'withhold width {wmax}', lambda p: p['width'] == wmax)
backtest('withhold late anchors', lambda p: p.get('late', False))
for frac in sorted({p['data_frac'] for p in points}):
    backtest(f'withhold data {frac:g}', lambda p, f=frac: p['data_frac'] == f)

# ---- ledger --------------------------------------------------------------------------------------------------------------
ledger = []
for r in runs:
    last = max(r['points'], key=lambda p: p['it']) if r['points'] else None
    gen_pf = last['gen_pflops'] if last else None
    ledger.append({'id': r['id'], 'family': r['family'], 'width': r['width'], 'data_frac': r['data_frac'], 'params': r['params'],
                   'updates': last['it'] if last else None, 'generator_pflops': gen_pf, 'tokenizer_pflops': r.get('tokenizer_pflops', 0),
                   'encode_pflops': r.get('encode_pflops', 0), 'total_training_pflops': (r['offset_pflops'] + gen_pf) if gen_pf else None,
                   'train_hours': r['hours'], 'final_fid': r['final_fid']})
with open(os.path.join(args.study, 'ledger.csv'), 'w') as f:
    keys = list(ledger[0])
    f.write(','.join(keys) + '\n')
    for row in ledger:
        f.write(','.join('' if row[k] is None else str(row[k]) for k in keys) + '\n')
out['ledger'] = ledger
out['experiment_train_hours'] = sum(r['hours'] for r in runs)

# ---- forecasts for a held-out width ----------------------------------------------------------------------------------------
if args.predict_width:
    w = args.predict_width
    fc = {'width': w, 'families': {}}
    D = max(r['train_images'] for r in runs)
    for fam, th in fits.items():
        ref = cfgs[fam][max(cfgs[fam])] # the largest tested configuration: params and FLOPs scaled to the new width
        wref = max(cfgs[fam])
        N = ref[0] * (w / wref) ** 2
        fwd = ref[1] * (w / wref) ** 2
        Ts = np.geomspace(1e6, 1e9, 40)
        fc['families'][fam] = {'params_est': N, 'fwd_gflops_est': fwd, 'offset_pflops': offsets.get((fam, 1.0), 0.0),
                               'curve': [{'presentations': float(T), 'total_pflops': offsets.get((fam, 1.0), 0.0) + T * 3 * fwd / 1e6,
                                          'fid': float(predict(th, N, T, D))} for T in Ts]}
    body = json.dumps(fc, indent=1, sort_keys=True)
    path = os.path.join(args.study, f'predictions_w{w}.json')
    with open(path, 'w') as f:
        f.write(body)
    with open(path + '.sha256', 'w') as f:
        f.write(hashlib.sha256(body.encode()).hexdigest() + '\n')
    print(f"\nforecasts for width {w} written to {path} (sha256 alongside)")

with open(os.path.join(args.study, 'frontier.json'), 'w') as f:
    json.dump(out, f, indent=1)

# ---- plots ---------------------------------------------------------------------------------------------------------------
try:
    import plot_style as ps
    plt = ps.setup()
    fracs = list(out['frontier'])
    fig, axes = plt.subplots(1, len(fracs), figsize=(5.6 * len(fracs), 4.4), facecolor=ps.SURFACE, squeeze=False)
    fam_color = {f: ps.CATEGORICAL[i % len(ps.CATEGORICAL)] for i, f in enumerate(families)}
    fam_marker = {f: ps.MARKERS[i % len(ps.MARKERS)] for i, f in enumerate(families)}
    for a, frac in zip(axes[0], fracs):
        fr = out['frontier'][frac]
        for fam in families:
            pts = [p for p in points if p['family'] == fam and str(p['data_frac']) == frac]
            if pts:
                a.plot([p['C'] for p in pts], [p['fid'] for p in pts], ls='none', marker=fam_marker[fam], ms=6, color=fam_color[fam],
                       mec=ps.SURFACE, mew=1.5, alpha=0.8, label=f'{fam} (measured)')
            cur = [(r_['C'], r_['per_family'][fam]['fid']) for r_ in fr['allocation'] if fam in r_['per_family']]
            if cur:
                a.plot(*zip(*cur), color=fam_color[fam], lw=2)
        for s in fr['switches']:
            a.axvline(s['C'], color=ps.INK_2, lw=1, ls=':')
        a.set_xscale('log'); a.set_yscale('log')
        ps.style_axis(a, 'total training compute (PFLOPs: tokenizer + encoding + generator)', 'guided FID',
                      f"data {float(frac):g}: fitted envelopes (lines), switches (dotted)")
        a.legend(fontsize=7, frameon=False, labelcolor=ps.INK_2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.study, 'frontier.png'), dpi=130, facecolor=ps.SURFACE)
    print(f"wrote {args.study}/frontier.json, frontier.png, ledger.csv")
except Exception as e: # plotting never blocks the numbers
    print(f"(plot skipped: {type(e).__name__}: {e}); wrote {args.study}/frontier.json, ledger.csv")
