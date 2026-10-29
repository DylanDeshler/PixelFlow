"""
A synthetic dataset of images made of anisotropic, rotated 2D Gaussian blobs, in the memmap layout data_jit.MemmapImages
reads (the same as prepare_jit_data.py's), for fast GPU experiments on adaptive tokenization (GaussJiT): the right token
shapes are known, and detail is unevenly spread, so where tokens go and how they're shaped should matter.

Each image: a softly graded background plus M blobs (M from --min_blobs to --max_blobs), each a Gaussian
a * colour * exp(-d^T Sigma^-1 d / 2) with standard deviations s_minor (log-uniform in [--min_sigma, --max_sigma]) and
s_major = aspect * s_minor (capped at --max_major), centre, angle, colour and amplitude random. --cluster of the blobs
fall in one disc per image (detail concentrated in one region), the rest anywhere.
Classes (10), by blob shape:
  0..7   elongated blobs (aspect 3..8), all oriented within one 22.5 degree bin (class k: k * 22.5 +- 11.25 degrees)
  8      round blobs (aspect 1)
  9      elongated blobs (aspect 3..8), random orientations
Also writes blobs.npz: every blob's parameters [N, max_blobs, 9] (cx, cy, s_major, s_minor, angle (rad), r, g, b, amplitude;
NaN past the image's M), for checking learned token geometry against the true shapes.

$ python prepare_toy_gaussians.py --out_dir=/data/toy_gauss_128                  # train (50k) and val (5k)
$ python fid.py --data_dir=/data/toy_gauss_128                                   # FID reference stats, Inception features
$ python train_jit.py config/toy_gauss_128.py config/model_gaussjit_b16.py --optimizer_type=muon
"""

import os
import json
import math
import argparse

import numpy as np
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--n_train', type=int, default=50000)
parser.add_argument('--n_val', type=int, default=5000)
parser.add_argument('--min_blobs', type=int, default=4)
parser.add_argument('--max_blobs', type=int, default=24)
parser.add_argument('--min_sigma', type=float, default=1.0, help='smallest minor-axis std (pixels)')
parser.add_argument('--max_sigma', type=float, default=6.0, help='largest minor-axis std (pixels)')
parser.add_argument('--max_major', type=float, default=40.0, help='cap on the major-axis std (pixels)')
parser.add_argument('--cluster', type=float, default=0.7, help='share of blobs placed in one disc per image')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--chunk', type=int, default=256)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

NUM_CLASSES = 10
CLASS_NAMES = [f'elongated {k * 22.5:.1f} deg' for k in range(8)] + ['round', 'elongated, random orientation']


def sample_blobs(labels, g):
    # blob parameters [B, M, 9] and validity mask [B, M] for a chunk with the given labels
    B, M, R = len(labels), args.max_blobs, args.img_size
    dev = labels.device
    rand = lambda *s: torch.rand(*s, generator=g, device=dev)
    count = torch.randint(args.min_blobs, args.max_blobs + 1, (B, 1), generator=g, device=dev)
    valid = torch.arange(M, device=dev)[None] < count
    # centres: a cluster disc per image (centre anywhere, radius 15-40% of the image) or uniform
    disc_c = rand(B, 1, 2) * R
    disc_r = (0.15 + 0.25 * rand(B, 1, 1)) * R
    ang, rad = rand(B, M) * 2 * math.pi, disc_r[..., 0] * rand(B, M).sqrt()
    in_disc = disc_c + torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], -1)
    centre = torch.where((rand(B, M) < args.cluster)[..., None], in_disc, rand(B, M, 2) * R).clamp(0, R)
    s_minor = torch.exp(math.log(args.min_sigma) + rand(B, M) * (math.log(args.max_sigma) - math.log(args.min_sigma)))
    lab = labels[:, None].expand(B, M)
    aspect = torch.where(lab == 8, torch.ones_like(s_minor), 3 + 5 * rand(B, M))
    s_major = (s_minor * aspect).clamp(max=args.max_major)
    bin_angle = (lab.clamp(max=7).float() + rand(B, M) - 0.5) * (math.pi / 8) # class k: k * 22.5 deg +- 11.25
    angle = torch.where(lab == 9, rand(B, M) * math.pi, bin_angle)
    colour = rand(B, M, 3)
    amp = 0.5 + 0.5 * rand(B, M)
    params = torch.cat([centre, s_major[..., None], s_minor[..., None], angle[..., None], colour, amp[..., None]], -1)
    return params, valid


def render(params, valid, g):
    # images [B, R, R, 3] in [0, 1]: graded background + the blobs (summed, clipped)
    B, R, dev = params.shape[0], args.img_size, params.device
    r = torch.arange(R, device=dev, dtype=torch.float32) + 0.5
    py, px = torch.meshgrid(r, r, indexing='ij')
    base = torch.rand(B, 1, 1, 3, generator=g, device=dev) * 0.6 + 0.2
    grad_dir = torch.rand(B, 1, 1, 2, generator=g, device=dev) * 2 - 1
    img = base + 0.15 * (grad_dir[..., 0:1] * (px[None, :, :, None] / R - 0.5) + grad_dir[..., 1:2] * (py[None, :, :, None] / R - 0.5))
    for m in range(params.shape[1]): # blobs one at a time: [B, R, R] intermediates
        p = params[:, m]
        dx, dy = px[None] - p[:, 0, None, None], py[None] - p[:, 1, None, None]
        c, s = torch.cos(p[:, 4])[:, None, None], torch.sin(p[:, 4])[:, None, None]
        u = (c * dx + s * dy) / p[:, 2, None, None]
        v = (-s * dx + c * dy) / p[:, 3, None, None]
        w = torch.exp(-0.5 * (u * u + v * v)) * (p[:, 8] * valid[:, m].float())[:, None, None]
        img = img + w[..., None] * (p[:, None, None, 5:8] - img) # alpha-blend towards the blob's colour
    return img.clamp(0, 1)


def write_split(out_dir, n, seed):
    os.makedirs(out_dir, exist_ok=True)
    R = args.img_size
    images = np.memmap(os.path.join(out_dir, 'images.u8'), dtype=np.uint8, mode='w+', shape=(n, R, R, 3))
    g = torch.Generator(device=args.device).manual_seed(seed)
    labels = torch.arange(n, device=args.device) % NUM_CLASSES # balanced classes
    labels = labels[torch.randperm(n, generator=g, device=args.device)]
    blobs = np.full((n, args.max_blobs, 9), np.nan, dtype=np.float32)
    for i in tqdm(range(0, n, args.chunk), desc=os.path.basename(out_dir.rstrip('/')) or out_dir):
        lab = labels[i:i + args.chunk]
        params, valid = sample_blobs(lab, g)
        images[i:i + len(lab)] = (render(params, valid, g) * 255 + 0.5).to(torch.uint8).cpu().numpy()
        blobs[i:i + len(lab)] = torch.where(valid[..., None], params, torch.full_like(params, float('nan'))).cpu().numpy()
    images.flush()
    np.save(os.path.join(out_dir, 'labels.npy'), labels.cpu().numpy().astype(np.int16))
    np.savez_compressed(os.path.join(out_dir, 'blobs.npz'), params=blobs,
                        columns=np.array(['cx', 'cy', 's_major', 's_minor', 'angle', 'r', 'g', 'b', 'amplitude']))
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump({'n': n, 'img_size': R, 'num_classes': NUM_CLASSES, 'classes': list(range(NUM_CLASSES)), 'class_names': CLASS_NAMES,
                   'toy': 'gaussian blobs', 'generator': vars(args)}, f)


write_split(args.out_dir, args.n_train, args.seed)
write_split(os.path.join(args.out_dir, 'val'), args.n_val, args.seed + 1)
print(f'done. next: python fid.py --data_dir={args.out_dir}')
