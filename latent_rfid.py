"""
Reconstruction floor of a latent pipeline: FID of VAE reconstructions (rFID), measured exactly like the generated-sample
FID of train_jit.py / train_latent.py (same Inception extractor, same reference stats, uint8 rounding), on held-out images
with balanced classes (num_images / classes per class, the same count as FID-10k).

  real      FID-{n} of the held-out images themselves: the finite-sample floor every model shares (FID-10k of perfect
            samples against full-train stats is not 0)
  recon     FID-{n} of their reconstructions decode(encode(x)) from the stored latents (prepare_latents.py, unflipped
            copy): the latent model's floor. A latent model can't get below it; a pixel model's floor is 'real'
Writes {latent_dir}/rfid.json.

$ python latent_rfid.py --data_dir=/data/in1k_128
$ python latent_rfid.py --data_dir=/data/in1k_128 --vae=mar-kl16          # MAR's ImageNet-only KL-16
"""

import os
import json
import argparse

import numpy as np
import torch

from data_jit import MemmapImages
from fid import InceptionMetrics, load_ref_stats, frechet_distance, to_uint8
from util.progress import bar

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='prepared pixel dataset (reference stats, val/ split)')
parser.add_argument('--vae', type=str, default='stabilityai/sd-vae-ft-ema', help="latent_vae.py's names, e.g. 'mar-kl16'")
parser.add_argument('--latent_dir', type=str, default='', help='default: {data_dir}/latents_{latent_vae.SHORT[vae]}')
parser.add_argument('--num_images', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

from latent_vae import SHORT
latent_dir = args.latent_dir or os.path.join(args.data_dir, 'latents_' + SHORT[args.vae])
lmeta = json.load(open(os.path.join(latent_dir, 'meta.json')))
val = MemmapImages(os.path.join(args.data_dir, 'val'))
z_all = np.load(os.path.join(latent_dir, 'val', 'latents.npy'), mmap_mode='r') # [N, 2 flips, C, H, W]
assert len(z_all) == len(val), f'{len(z_all)} latents vs {len(val)} val images'

# balanced classes: the first num_images / classes images of every class (sorted index order)
labels = np.asarray(val.labels)
classes = int(labels.max()) + 1
per = args.num_images // classes
assert per >= 1 and per * classes == args.num_images, f'num_images must be a multiple of the class count ({classes})'
idx = np.sort(np.concatenate([np.flatnonzero(labels == c)[:per] for c in range(classes)]))
assert len(idx) == args.num_images, f'some class has fewer than {per} held-out images'

ref_path = os.path.join(args.data_dir, 'fid_stats_jit.npz')
ref_path = ref_path if os.path.exists(ref_path) else os.path.join(args.data_dir, 'fid_stats.npz')
ref_stats = load_ref_stats(ref_path)
metrics = InceptionMetrics(args.device)
from latent_vae import load_vae
vae = load_vae(lmeta['vae'], args.device)

feats = {'real': [], 'recon': []}
mse = 0.0
for s in bar(range(0, len(idx), args.batch_size), f'real + reconstructed features ({len(idx)} images)'):
    b = idx[s:s + args.batch_size]
    x_u8, _ = val.get(b)
    x_u8 = torch.from_numpy(x_u8).to(args.device).permute(0, 3, 1, 2) # [B, 3, R, R] uint8
    z = torch.from_numpy(np.asarray(z_all[b, 0]).astype(np.float32)).to(args.device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled='cuda' in args.device):
        rec = vae.decode_img(z / lmeta['scale']).float().clamp(-1, 1)
    rec_u8 = to_uint8(rec)
    mse += (rec_u8.float() - x_u8.float()).div(127.5).pow(2).mean().item() * len(b)
    feats['real'].append(metrics.features(x_u8)[0])
    feats['recon'].append(metrics.features(rec_u8)[0])

res = {k: frechet_distance(torch.cat(v), ref_stats, args.device) for k, v in feats.items()}
psnr = 10 * np.log10(4.0 / (mse / len(idx))) # images in [-1, 1]: peak-to-peak 2
out = {'num_images': len(idx), 'ref_stats': ref_path, 'vae': lmeta['vae'], 'fid_real': res['real'], 'rfid': res['recon'], 'psnr': psnr}
with open(os.path.join(latent_dir, 'rfid.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"FID-{len(idx) / 1000:g}k vs {os.path.basename(ref_path)}: real held-out images {res['real']:.2f} (the shared floor), "
      f"VAE reconstructions {res['recon']:.2f} (the latent floor); reconstruction PSNR {psnr:.2f} dB\nwrote {os.path.join(latent_dir, 'rfid.json')}")
