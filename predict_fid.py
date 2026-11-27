"""
Predicting a run's converged FID from its training curve (the study's decisions: tuning candidates' stopping and ranking,
the frontier). Numpy only, so the trainers, the queue's tools and frontier.py share one definition.

  predict(points, target, floor)  points [(x, fid)] with x the training progress (full-data epochs for generators, updates for
                           tokenizers): log F = a + b log x by least squares, the first point dropped when >= 4 later ones
                           exist (the earliest eval sits off the power law), fitted as F = floor + A x^-a, evaluated at
                           x = target. None with < 3 points above the floor
  floor_of(run)            the run's FID floor at the in-loop sample count: a latent run's tokenizer reconstruction FID, a pixel
                           run's real held-out FID (both measured at 2K, latent_rfid.py)
  settled(preds, tol, n)   the last n relative changes of a run's successive predictions are all < tol
  history(run, key)        [(full-data epochs, value)] of a scalar from a run's diagnostics/diag_*.json (both trainers)
Backtests on the 80 / 120-epoch pixel runs (fit on <= 20 epochs): the 80-epoch value within ~1 FID; long extrapolations
lean optimistic by a few percent (the curves flatten a little faster than a power law), so predictions are compared across
candidates (the same bias) rather than read as absolute values.
"""

import os
import glob
import json

import numpy as np

FULL_TRAIN = 1281167 # ImageNet-1k training images: the epoch unit for every run, subsets included


def predict(points, target, floor=0.0):
    # F = floor + A x^-a: the floor is the run's measured asymptote bound (floor_of), so a run approaching it is not extrapolated
    # as if it could keep falling toward zero
    pts = sorted((float(x), float(f)) for x, f in points if f is not None and np.isfinite(f) and f > floor)
    if len(pts) < 3:
        return None
    X, F = np.array(pts).T
    use = X > X[0] if (X > X[0]).sum() >= 4 else np.ones_like(X, bool)
    (a, b), *_ = np.linalg.lstsq(np.stack([np.ones(use.sum()), np.log(X[use])], 1), np.log(F[use] - floor), rcond=None)
    return float(floor + np.exp(a + b * np.log(target)))


def floor_of(run, n_k=2):
    # the lower bound of a run's FID-{n_k}K (the in-loop guided FID's sample count): a latent run's tokenizer reconstruction FID
    # ({latent_dir}/rfid_{n}k.json, latent_rfid.py), a pixel run's real held-out FID ({data_dir}/fid_real_{n}k.json); 0 if not measured
    try:
        cfg = json.load(open(os.path.join(run, 'config.json')))
    except OSError:
        return 0.0
    res = cfg.get('resolved', {})
    if res.get('vae') and res.get('latent_dir'):
        path, key = os.path.join(res['latent_dir'], f'rfid_{n_k}k.json'), 'rfid'
    else:
        path, key = os.path.join(cfg.get('data_dir') or res.get('data_dir', ''), f'fid_real_{n_k}k.json'), 'fid_real'
    try:
        return float(json.load(open(path))[key])
    except (OSError, KeyError, ValueError):
        return 0.0


def settled(preds, tol, n=2):
    p = [v for v in preds if v is not None]
    if len(p) < n + 1:
        return False
    return all(abs(p[-k] / p[-k - 1] - 1) < tol for k in range(1, n + 1))


def history(run, key, max_epochs=0):
    batch = json.load(open(os.path.join(run, 'config.json')))['resolved']['effective_batch_size']
    out = []
    for f in glob.glob(os.path.join(run, 'diagnostics', 'diag_*.json')):
        r = json.load(open(f))
        v = r.get('scalars', {}).get(key)
        e = r['iter'] * batch / FULL_TRAIN
        if v is not None and (not max_epochs or e <= max_epochs + 1e-6):
            out.append((e, float(v)))
    return sorted(out)
