"""
Rank hyperparameter candidates from short runs by their projected end-of-training guided FID (queues/study.sh's tuning).

Each candidate's in-loop guided FID (val/guided_fid in its diagnostics: the post-hoc EMA 0.10 with a small CFG search at every
eval, the same sample count for all) against training progress in full-data epochs (subsets and widths share one axis) is
fitted and projected to --target_epochs by predict_fid.predict (the unguided prediction is reported alongside). Candidates are ranked by the projection; the
ranking at the last measured checkpoint is printed next to it, and disagreements are flagged.
Writes {study}/decisions/{name}.txt ("best_id projected_fid") and {name}.json (the table).

$ python tune_rank.py --study=runs/study --name=pixel_lr --target_epochs=100 tune_pixel_lr1e-4 tune_pixel_lr2e-4 tune_pixel_lr4e-4
"""

import os
import re
import glob
import json
import argparse

import numpy as np

FULL_TRAIN = 1281167 # ImageNet-1k train images: the epoch unit

p = argparse.ArgumentParser()
p.add_argument('ids', nargs='+')
p.add_argument('--study', default='runs/study')
p.add_argument('--name', required=True)
p.add_argument('--target_epochs', type=float, default=100)
p.add_argument('--max_epochs', type=float, default=0, help='use only checkpoints up to this many full-data epochs (0 = all)')
a = p.parse_args()

import predict_fid
rows = []
for i in a.ids:
    d = os.path.join(a.study, 'runs', i)
    pts = predict_fid.history(d, 'val/guided_fid', a.max_epochs) # the in-loop guided curve (up to the candidates' length)
    proj = predict_fid.predict(pts, a.target_epochs, predict_fid.floor_of(d))
    assert proj is not None, f'{i}: {len(pts)} guided evals, need >= 3'
    uproj = predict_fid.predict(predict_fid.history(d, 'val/fid', a.max_epochs), a.target_epochs)
    E, F = (np.array(pts).T if pts else ([], []))
    rows.append({'id': i, 'epochs': list(map(float, E)), 'guided_fid': list(map(float, F)), 'last': float(F[-1]), 'projected': proj,
                 'projected_unguided': uproj, 'alpha': float('nan')})
rows.sort(key=lambda r: r['projected'])
last_best = min(rows, key=lambda r: r['last'])['id']
print(f"{a.name}: guided FID by checkpoint (full-data epochs) and projected to epoch {a.target_epochs:g}")
for r in rows:
    curve = ', '.join(f'{e:.0f}: {f:.1f}' for e, f in zip(r['epochs'], r['guided_fid']))
    print(f"  {r['id']:<44} {curve}  ->  guided {r['projected']:.1f}, unguided {r['projected_unguided'] or float('nan'):.1f}")
best = rows[0]
note = '' if best['id'] == last_best else f"  [at the last checkpoint {last_best} is ahead: the curves cross]"
print(f">>> {a.name}: {best['id']} (projected {best['projected']:.1f}){note}")
os.makedirs(os.path.join(a.study, 'decisions'), exist_ok=True)
with open(os.path.join(a.study, 'decisions', a.name + '.txt'), 'w') as f:
    f.write(f"{best['id']} {best['projected']}\n")
with open(os.path.join(a.study, 'decisions', a.name + '.json'), 'w') as f:
    json.dump({'target_epochs': a.target_epochs, 'rows': rows, 'last_checkpoint_best': last_best}, f, indent=1)
