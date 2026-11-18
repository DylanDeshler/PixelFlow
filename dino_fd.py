"""
Fréchet distance in DINOv2 CLS-feature space (FD-DINOv2), for evaluations where Inception FID is a confounded judge
(Inception's feature space is close to ImageNet classification; DINOv2 is self-supervised and more sensitive to texture
and layout). Reference statistics come from real training images (optionally only some classes) and are cached as npz.

    fd = DinoFD(device)
    ref = fd.reference(data, n_per_class=10, classes=None, cache='out/dino_ref.npz')
    feats = fd.features(x_uint8)            # [B, D], call per generated batch
    fd.distance(torch.cat(all_feats), ref)
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from fid import frechet_distance
from util.progress import bar


class DinoFD:
    def __init__(self, device, encoder='dinov2_vitb14', grid=16):
        self.device, self.grid = device, grid
        self.model = torch.hub.load('facebookresearch/dinov2', encoder).to(device).eval().requires_grad_(False)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
        self.ctx = torch.autocast('cuda', dtype=torch.bfloat16) if 'cuda' in str(device) else torch.autocast('cpu', enabled=False)

    @torch.no_grad()
    def features(self, x_uint8):
        # x_uint8: [B, 3, H, W] uint8 -> CLS features [B, D] (float32, on CPU)
        x = (x_uint8.to(self.device).float() / 255 - self.mean) / self.std
        x = F.interpolate(x, size=(self.grid * 14, self.grid * 14), mode='bicubic', align_corners=False)
        with self.ctx:
            f = self.model.forward_features(x)['x_norm_clstoken']
        return f.float().clone().cpu()

    def reference(self, data, n_per_class=10, classes=None, cache='', seed=0, batch=250):
        # mu / sigma of real training images: n_per_class per class (classes: a list of labels, or all)
        if cache and os.path.exists(cache):
            d = np.load(cache)
            return {'mu': d['mu'], 'sigma': d['sigma']}
        rng = np.random.default_rng(seed)
        cl = range(data.num_classes) if classes is None else classes
        pools = [np.flatnonzero(data.labels == c) for c in cl]
        idx = np.sort(np.concatenate([rng.choice(p, min(n_per_class, len(p)), replace=False) for p in pools]))
        feats = []
        for i in bar(range(0, len(idx), batch), f'FD-DINOv2 reference ({len(idx)} real images)'):
            x = torch.from_numpy(np.asarray(data.images[idx[i:i + batch]])).permute(0, 3, 1, 2)
            feats.append(self.features(x))
        f = torch.cat(feats).double()
        mu = f.mean(0)
        fc = f - mu
        ref = {'mu': mu.float().numpy(), 'sigma': (fc.T @ fc / (len(f) - 1)).numpy()}
        if cache:
            os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
            np.savez(cache, **ref)
        return ref

    def distance(self, feats, ref):
        return frechet_distance(feats, ref, self.device)
