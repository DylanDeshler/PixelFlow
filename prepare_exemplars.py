"""
Exemplar memory for the KV prefix (train_jit.py --kv_prefix_exemplar_n): for every class, its N most representative
training images and their DINOv2 patch features pooled to a grid x grid summary, cached once so training only reads and
pools them (no encoding during training).

For each class: DINOv2-B CLS features of its training images (all, or --pool_per_class at random) and their N medoids:
k-medoids within the class (cosine distance), i.e. the N images that best cover it (the summed distance of every class
image to its nearest medoid is minimized): greedy initialization (each step adds the image that most reduces that sum),
then alternating refinement (assign each image to its nearest medoid; replace each medoid by the member minimizing the
summed distance to its group) until no medoid changes. For the medoids: DINOv2-B/14 patch features of the image upsampled
to 224 px (128 px isn't a multiple of the 14 px patch): the full 16 x 16 grid of 768-d tokens is stored, so train_jit can
pool it to any K tokens per exemplar.

Writes {data_dir}/exemplars_{encoder}_N{N}.npz (indices [C, N] dataset rows, cls [C, N, D] float16, meta) and
{data_dir}/exemplars_{encoder}_N{N}_patches.npy (float16 [C, N, 256, D], memory-mapped when loaded; ~6 GB for N = 16).

$ python prepare_exemplars.py --data_dir=/data/in1k_128 --n_medoids=16
"""

import os
import json

import numpy as np
import torch
import torch.nn.functional as F

from data_jit import MemmapImages
from util.progress import bar

# -----------------------------------------------------------------------------
data_dir = ''
n_medoids = 16 # N exemplars per class (its most representative images)
pool_per_class = 0 # images per class to cluster (0 = all)
medoid_iters = 50 # max alternating refinement rounds
encoder = 'dinov2_vitb14'
batch = 256
seed = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out = '' # default {data_dir}/exemplars_{encoder}_N{n_medoids}.npz
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

assert data_dir, 'pass --data_dir'
out = out or os.path.join(data_dir, f'exemplars_{encoder}_N{n_medoids}.npz')
data = MemmapImages(data_dir)
C = data.num_classes
reader = torch.hub.load('facebookresearch/dinov2', encoder).to(device).eval().requires_grad_(False)
MEAN = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
ctx = torch.autocast('cuda', dtype=torch.bfloat16) if 'cuda' in device else torch.autocast('cpu', enabled=False)

@torch.no_grad()
def read(idx, patches=False):
    x = torch.from_numpy(np.asarray(data.images[np.sort(idx)])).to(device).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate((x - MEAN) / STD, size=(224, 224), mode='bicubic', align_corners=False)
    with ctx:
        f = reader.forward_features(x)
    cls = f['x_norm_clstoken'].float().clone()
    if not patches:
        return cls
    return cls, f['x_norm_patchtokens'].float().clone() # [B, 256, D]: the full 16 x 16 grid

def k_medoids(x, k):
    # x: unit rows [n, D] (one class). Returns k row indices minimizing sum_i min_m d(i, m), d = 1 - cosine similarity
    d = (1 - x @ x.T).clamp_min(0) # [n, n]
    n = len(x)
    chosen = [int(d.sum(1).argmin())] # the single best medoid
    nearest = d[chosen[0]].clone()
    for _ in range(1, k): # greedy: the candidate that most reduces the total distance to the nearest medoid
        gain = (nearest.unsqueeze(0) - d).clamp_min(0).sum(1) # [n]: reduction if each row were added
        gain[chosen] = -1
        j = int(gain.argmax())
        chosen.append(j)
        nearest = torch.minimum(nearest, d[j])
    med = torch.tensor(chosen, device=x.device)
    for _ in range(medoid_iters): # alternating refinement
        assign = d[:, med].argmin(1)
        new = med.clone()
        for j in range(k):
            members = torch.nonzero(assign == j).flatten()
            if len(members):
                new[j] = members[d[members][:, members].sum(1).argmin()]
        if torch.equal(new, med) or len(torch.unique(new)) < k:
            break
        med = new
    return med.tolist(), float(d[:, med].min(1).values.mean())

rng = np.random.default_rng(seed)
indices = np.zeros((C, n_medoids), dtype=np.int64)
cover = []
for c in bar(range(C), f'k-medoids: {n_medoids} per class ({encoder} CLS features, cosine distance)'):
    idx = np.flatnonzero(data.labels == c)
    if pool_per_class and len(idx) > pool_per_class:
        idx = np.sort(rng.choice(idx, pool_per_class, replace=False))
    assert len(idx) >= n_medoids, f'class {c} has {len(idx)} images, fewer than n_medoids {n_medoids}'
    feats = F.normalize(torch.cat([read(idx[i:i + batch]) for i in range(0, len(idx), batch)]), dim=-1)
    med, mean_dist = k_medoids(feats, n_medoids)
    indices[c] = idx[med]
    cover.append(mean_dist)

flat = indices.reshape(-1)
order = np.argsort(flat)
patches_path = out[:-4] + '_patches.npy'
patch_all, cls_all = None, None
for i in bar(range(0, len(flat), batch), f'exemplar patch features ({len(flat)} images, full 16 x 16 grid)'):
    sel = order[i:i + batch]
    cls, p = read(flat[sel], patches=True) # flat[sel] is sorted, as read() expects
    if patch_all is None:
        patch_all = np.lib.format.open_memmap(patches_path, mode='w+', dtype=np.float16, shape=(C, n_medoids, p.shape[1], p.shape[2]))
        cls_all = np.zeros((len(flat), cls.shape[1]), np.float16)
    rows = np.unravel_index(sel, (C, n_medoids))
    patch_all[rows] = p.cpu().numpy().astype(np.float16)
    cls_all[sel] = cls.cpu().numpy().astype(np.float16)
patch_all.flush()
np.savez(out, indices=indices, cls=cls_all.reshape(C, n_medoids, -1),
         meta=json.dumps({'encoder': encoder, 'n_medoids': n_medoids, 'grid': int(round(patch_all.shape[2] ** 0.5)),
                          'pool_per_class': pool_per_class, 'img_size': data.img_size, 'num_classes': C,
                          'patches': os.path.basename(patches_path), 'mean_cover_distance': float(np.mean(cover))}))
print(f'wrote {out} and {patches_path}: {C} classes x {n_medoids} medoids, {patch_all.shape[2]} DINO patch tokens each '
      f'(dim {patch_all.shape[3]}); mean distance of a class image to its nearest medoid {np.mean(cover):.3f}')
