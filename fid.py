"""
In-memory FID / Inception Score, numerically equivalent to JiT's evaluation.

JiT writes PNGs and runs its torch-fidelity fork against precomputed reference stats
(the fork only adds loading mu/sigma from an .npz). Here the same torch-fidelity
Inception extractor ('inception-v3-compat') and metric functions run on uint8 tensors
directly, so nothing is written to disk.

Reference data for a prepared dataset (see prepare_jit_data.py):
$ python fid.py --data_dir=/data/jit/in100_128
writes, over every training image (and the val split if present):
    fid_stats.npz           mu / sigma for FID
    train_inception.npy     float16 [N, 2048] pool features, for precision/recall and memorization checks
    val/val_inception.npy   same for the held-out split
"""

import os
import argparse

import numpy as np
import torch
from tqdm import tqdm


class InceptionMetrics:
    def __init__(self, device):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        self.extractor = FeatureExtractorInceptionV3('inception-v3-compat', ['2048', 'logits_unbiased']).to(device).eval()
        self.device = device

    @torch.no_grad()
    def features(self, images_uint8):
        # images_uint8: [B, 3, H, W] uint8
        # float32 like torch-fidelity's own pipeline: mu stays float32, np.cov upcasts sigma to float64 (same as JiT's stats files)
        f, logits = self.extractor(images_uint8.to(self.device))
        return f.float().cpu(), logits.float().cpu()

    def compute(self, features, logits, ref_stats):
        from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
        from torch_fidelity.metric_isc import isc_features_to_metric
        stats = fid_features_to_statistics(features)
        fid = fid_statistics_to_metric(stats, ref_stats, verbose=False)['frechet_inception_distance']
        isc = isc_features_to_metric(logits)['inception_score_mean']
        return fid, isc

def load_ref_stats(path):
    d = np.load(path)
    return {'mu': d['mu'], 'sigma': d['sigma']}

def to_uint8(x):
    # [-1, 1] float -> [0, 255] uint8, rounding the same way JiT does before writing PNGs
    return torch.round(((x.float() + 1) / 2).clamp(0, 1) * 255).to(torch.uint8)

def precision_recall(real, fake, k=3, chunk=4096):
    """
    Improved precision / recall (Kynkaenniemi et al. 2019) on Inception pool features: a sample counts as covered
    if it falls inside the k-NN ball of some sample from the other set. Precision ~ fidelity, recall ~ coverage.
    """
    def kth_radius(x):
        radii = []
        for i in range(0, len(x), chunk):
            d = torch.cdist(x[i:i + chunk], x)
            radii.append(d.kthvalue(k + 1, dim=1).values) # +1 skips the zero distance to itself
        return torch.cat(radii)

    def coverage(queries, refs, ref_radii):
        hits = []
        for i in range(0, len(queries), chunk):
            d = torch.cdist(queries[i:i + chunk], refs)
            hits.append((d <= ref_radii.unsqueeze(0)).any(dim=1))
        return torch.cat(hits).float().mean().item()

    real, fake = real.float(), fake.float()
    precision = coverage(fake, real, kth_radius(real))
    recall = coverage(real, fake, kth_radius(fake))
    return precision, recall

if __name__ == '__main__':
    from data_jit import MemmapImages

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    metrics = InceptionMetrics(args.device)

    def extract(split_dir, out_name, with_stats):
        data = MemmapImages(split_dir)
        feats = np.lib.format.open_memmap(os.path.join(split_dir, out_name), mode='w+', dtype=np.float16, shape=(len(data), 2048))
        # streaming float64 sums, so FID stats over 1.28M images don't need the features in RAM
        s1, s2 = torch.zeros(2048, dtype=torch.float64), torch.zeros(2048, 2048, dtype=torch.float64)
        for start in tqdm(range(0, len(data), args.batch_size), desc=f'inception features {split_dir}'):
            idx = np.arange(start, min(start + args.batch_size, len(data)))
            x, _ = data.get(idx)
            f = metrics.features(torch.from_numpy(x).permute(0, 3, 1, 2))[0]
            feats[idx] = f.numpy().astype(np.float16)
            f = f.double()
            s1 += f.sum(0)
            s2 += f.T @ f
        feats.flush()
        print(f"wrote {len(data):,} features to {os.path.join(split_dir, out_name)}")
        if with_stats:
            n = len(data)
            mu = s1 / n
            sigma = (s2 - n * torch.outer(mu, mu)) / (n - 1) # unbiased, same as np.cov
            out_path = os.path.join(split_dir, 'fid_stats.npz')
            # mu float32 / sigma float64, matching torch-fidelity and JiT's stats files
            np.savez(out_path, mu=mu.numpy().astype(np.float32), sigma=sigma.numpy())
            print(f"wrote reference stats to {out_path}")
            return out_path

    out_path = extract(args.data_dir, 'train_inception.npy', with_stats=True)
    val_dir = os.path.join(args.data_dir, 'val')
    if os.path.exists(os.path.join(val_dir, 'meta.json')):
        extract(val_dir, 'val_inception.npy', with_stats=False)

    # if JiT's official stats are here (in1k at 256/512), our preprocessing should reproduce them almost exactly
    official_path = os.path.join(args.data_dir, 'fid_stats_jit.npz')
    if os.path.exists(official_path):
        from torch_fidelity.metric_fid import fid_statistics_to_metric
        fid = fid_statistics_to_metric(load_ref_stats(out_path), load_ref_stats(official_path), verbose=False)['frechet_inception_distance']
        print(f"FID between our reference stats and JiT's official ones: {fid:.4f} (should be ~0)")
