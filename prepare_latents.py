"""
Encode a prepared pixel dataset (prepare_jit_data.py layout: {data_dir}/images.u8 + labels.npy + meta.json, and val/ alike)
into Stable Diffusion VAE latents for latent-space baselines (train_latent.py), as DiT / SiT train: 128px -> 4 x 16 x 16
(downsampling 8), scaled by the VAE's 0.18215 so the latents have roughly unit variance. Both horizontal flips of every
image are encoded (a VAE doesn't commute with flipping), stored float16 as [N, 2, 4, 16, 16] (0 = as is, 1 = flipped), and
train_latent.py picks one per sample, as random flips would. The posterior mean is stored (deterministic; DiT's feature
extraction samples once instead, a negligible difference at this noise level).

Writes {out_dir}/train/latents.npy and {out_dir}/val/latents.npy (memmaps; labels are read from the pixel dataset) and
{out_dir}/meta.json. ImageNet-1k at 128px: ~5.2 GB train. Needs `pip install diffusers` and the VAE weights from the hub.

$ python prepare_latents.py --data_dir=/data/in1k_128          # -> /data/in1k_128/latents_sdvae-ft-ema
"""

import os
import json
import argparse

import numpy as np
import torch

from data_jit import MemmapImages
from util.progress import bar

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, default='', help="default: {data_dir}/latents_{vae name}")
parser.add_argument('--vae', type=str, default='stabilityai/sd-vae-ft-ema', help="a diffusers AutoencoderKL ('stabilityai/sd-vae-ft-mse' also works)")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

SCALE = 0.18215 # the SD VAE's latent scaling (DiT / SiT / LDM)
out_dir = args.out_dir or os.path.join(args.data_dir, 'latents_' + args.vae.split('/')[-1].replace('sd-vae-', 'sdvae-'))

from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained(args.vae).to(args.device).eval()

@torch.no_grad()
def encode(x_uint8):
    x = torch.from_numpy(x_uint8).to(args.device).permute(0, 3, 1, 2).float().div_(127.5).sub_(1) # [-1, 1], as the VAE was trained
    both = torch.cat([x, x.flip(-1)])
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled='cuda' in args.device):
        z = vae.encode(both).latent_dist.mean * SCALE
    z = z.float()
    n = x.shape[0]
    return torch.stack([z[:n], z[n:]], dim=1).half().cpu().numpy() # [n, 2, 4, h, w]

shape = None
for split, src in (('train', args.data_dir), ('val', os.path.join(args.data_dir, 'val'))):
    if not os.path.exists(os.path.join(src, 'meta.json')):
        print(f'no {split} split at {src}, skipped')
        continue
    data = MemmapImages(src)
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    path = os.path.join(out_dir, split, 'latents.npy')
    probe = encode(data.images[:2])
    shape = probe.shape[2:]
    arr = np.lib.format.open_memmap(path + '.tmp', mode='w+', dtype=np.float16, shape=(len(data), 2, *shape))
    stats = np.zeros(2, dtype=np.float64) # running sum and sum of squares of the as-is latents, for the variance check
    for s in bar(range(0, len(data), args.batch_size), f'encoding {split} ({len(data)} images)'):
        z = encode(np.asarray(data.images[s:s + args.batch_size]))
        arr[s:s + len(z)] = z
        stats += [z[:, 0].astype(np.float64).sum(), (z[:, 0].astype(np.float64) ** 2).sum()]
    arr.flush()
    del arr
    os.replace(path + '.tmp', path)
    n_el = len(data) * int(np.prod(shape))
    mean = stats[0] / n_el
    std = np.sqrt(stats[1] / n_el - mean ** 2)
    print(f'{split}: {len(data)} images -> {path} ({tuple(shape)} per image and flip), latent mean {mean:.3f}, std {std:.3f}')

with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
    json.dump({'vae': args.vae, 'scale': SCALE, 'latent_shape': list(map(int, shape)), 'flips': 2, 'pixel_data_dir': args.data_dir,
               'img_size': json.load(open(os.path.join(args.data_dir, 'meta.json')))['img_size']}, f, indent=2)
print(f'done: {out_dir}')
