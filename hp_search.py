"""
Adaptive, fixed-budget search over one hyperparameter in log space, from short runs (queues/study.sh), in the spirit of the
CFG search: start at --start; the next candidate is a step up; while the best is at an edge of what has been tried, extend
past it by --step (higher better -> even higher); once the best has a worse neighbour on both sides, bisect (geometric mean)
toward its better neighbour; stop at --budget runs or when the bracket is narrower than --min_ratio.
Candidates are scored by their predicted converged guided FID (predict_fid on the in-loop guided curve, up to the
candidates' length), once the candidate has stopped (settled.json). Candidate ids are {prefix}{value:g}; --alias value=id points a value at an existing run (e.g. the
conditioning run is the 1e-4 candidate). Deterministic given the results, so a restarted queue proposes the same sequence.

  next  print the next value to run, 'wait' if a proposed candidate has no results yet, or 'done'
  ids   print the candidate ids tried so far (for tune_rank.py)

$ python hp_search.py next --study=runs/study --prefix=tune_pix_lr --start=1e-4 --step=2 --budget=4 --alias=1e-4=cond_kvtable
"""

import os
import re
import glob
import json
import argparse

import numpy as np

FULL_TRAIN = 1281167

p = argparse.ArgumentParser()
p.add_argument('action', choices=('next', 'ids'))
p.add_argument('--study', default='runs/study')
p.add_argument('--prefix', required=True)
p.add_argument('--start', type=float, required=True)
p.add_argument('--step', type=float, default=2.0)
p.add_argument('--budget', type=int, default=4)
p.add_argument('--min_ratio', type=float, default=1.25)
p.add_argument('--target_epochs', type=float, default=100)
p.add_argument('--max_epochs', type=float, default=50)
p.add_argument('--alias', action='append', default=[])
a = p.parse_args()
alias = {float(k): v for k, v in (x.split('=', 1) for x in a.alias)}
seq_path = os.path.join(a.study, 'decisions', f'{a.prefix}search.json') # the proposed values, in order


def cid(v):
    return alias.get(v, f'{a.prefix}{v:g}')


def score(v):
    # the candidate's predicted converged guided FID (predict_fid on its in-loop guided curve), or None without enough evals yet
    import predict_fid
    d = os.path.join(a.study, 'runs', cid(v))
    if not os.path.exists(os.path.join(d, 'config.json')):
        return None
    done = os.path.exists(os.path.join(d, 'settled.json')) or os.path.islink(d) or cid(v) in alias.values()
    pts = predict_fid.history(d, 'val/guided_fid', a.max_epochs)
    return predict_fid.predict(pts, a.target_epochs, predict_fid.floor_of(d)) if (done and len(pts) >= 3) else None


seq = json.load(open(seq_path)) if os.path.exists(seq_path) else []
if a.action == 'ids':
    print(' '.join(cid(v) for v in seq))
    raise SystemExit
res = {v: score(v) for v in seq}
if any(s is None for s in res.values()):
    print('wait')
    raise SystemExit
if len(seq) >= a.budget:
    print('done')
    raise SystemExit
if not seq:
    nxt = a.start
elif len(seq) == 1:
    nxt = seq[0] * a.step
else:
    pts = sorted(res)
    i = min(range(len(pts)), key=lambda j: res[pts[j]])
    if i == len(pts) - 1:
        nxt = pts[-1] * a.step # best at the top: go higher
    elif i == 0:
        nxt = pts[0] / a.step # best at the bottom: go lower
    else: # bracketed: bisect toward the better neighbour (log space)
        lo, hi = pts[i - 1], pts[i + 1]
        nb = lo if res[lo] < res[hi] else hi
        if max(nb, pts[i]) / min(nb, pts[i]) < a.min_ratio:
            print('done')
            raise SystemExit
        nxt = float(np.sqrt(nb * pts[i]))
nxt = float(f'{nxt:.3g}')
if nxt in seq:
    print('done')
    raise SystemExit
seq.append(nxt)
os.makedirs(os.path.dirname(seq_path), exist_ok=True)
json.dump(seq, open(seq_path, 'w'))
print(f'{nxt:g}')
