"""
Which discovery-matrix runs to extend (queues/study.sh): every matrix run first trains until its predicted converged guided FID
has settled (predict-and-stop, as the tuning candidates); a run is then extended toward the full length only if the
pixel-vs-latent decision could depend on it.

For one data size: each run's guided FID as a function of total training compute (tokenizer training + dataset encoding +
generator training): measured (the best in-loop guided FID so far) up to where it stopped, predicted (predict_fid on its
in-loop guided curve) beyond, up to its full length (--full_epochs). On a log grid of budgets, each family's best run at each
budget (pixel vs latent: latent-early and latent-full pooled, since the tokenizer investment is part of the allocation) is
found. Where the two families' best values are within --margin of each other, the decision there is uncertain, and each
family's best run at that budget is extended if the budget lies beyond what it has measured. Prints the run ids to extend.

$ python extend_plan.py --study=runs/study --frac=1 --margin=0.10 m_pixel_w256_frac1 m_early_w192_frac1 ...
"""

import os
import json
import argparse

import numpy as np

import predict_fid

p = argparse.ArgumentParser()
p.add_argument('ids', nargs='+')
p.add_argument('--study', default='runs/study')
p.add_argument('--frac', type=float, required=True)
p.add_argument('--margin', type=float, default=0.10)
p.add_argument('--full_epochs', type=float, default=100)
p.add_argument('--grid', type=int, default=80)
a = p.parse_args()

runs = []
for i in a.ids:
    d = os.path.join(a.study, 'runs', i)
    r = json.load(open(os.path.join(d, 'config.json')))['resolved']
    per_epoch = predict_fid.FULL_TRAIN * 3 * r['fwd_gflops'] / 1e6 # PFLOPs per full-data epoch of generator training
    off = r.get('train_pflops_offset', 0.0) or 0.0
    h = predict_fid.history(d, 'val/guided_fid')
    if len(h) < 3:
        continue
    famf = os.path.join(a.study, 'args', i + '.fam') # the study records each cell's family; else: latent runs record their VAE
    fam = open(famf).read().strip() if os.path.exists(famf) else ('latent' if 'vae' in r else 'pixel')
    runs.append({'id': i, 'family': fam, 'off': off, 'per_epoch': per_epoch, 'hist': h, 'measured_to': h[-1][0], 'floor': predict_fid.floor_of(d)})


def value(run, C):
    # guided FID of `run` at total training compute C (None if C is before its first eval or past its full length)
    e = (C - run['off']) / run['per_epoch']
    if e < run['hist'][0][0] or e > a.full_epochs + 1e-6:
        return None, e
    if e <= run['measured_to']:
        return min(f for x, f in run['hist'] if x <= e), e
    return predict_fid.predict(run['hist'], e, run['floor']), e


Cs = [r['off'] + r['per_epoch'] * x for r in runs for x in (r['hist'][0][0], a.full_epochs)]
extend, uncertain = set(), []
for C in np.geomspace(min(Cs), max(Cs), a.grid):
    best = {}
    for r in runs:
        v, e = value(r, C)
        if v is not None and (r['family'] not in best or v < best[r['family']][0]):
            best[r['family']] = (v, r, e)
    if len(best) == 2:
        (vp, rp, ep), (vl, rl, el) = best['pixel'], best['latent']
        if abs(vp / vl - 1) < a.margin:
            uncertain.append(float(C))
            for v, r, e in (best['pixel'], best['latent']):
                if e > r['measured_to'] + 1e-6:
                    extend.add(r['id'])
if uncertain:
    print(f"# data {a.frac:g}: pixel and latent within {a.margin:.0%} between {min(uncertain):,.0f} and {max(uncertain):,.0f} PFLOPs", file=os.sys.stderr)
print(' '.join(sorted(extend)))
