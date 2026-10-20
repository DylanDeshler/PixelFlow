"""
Builds the uint8 memmap datasets used by train_jit.py from the ImageFolder made by prepare_imagenet.py.

$ python prepare_jit_data.py --data_dir=/data/imagenet --out_dir=/data/jit/in100_128 --img_size=128 --subset=in100
$ python prepare_jit_data.py --data_dir=/data/imagenet --out_dir=/data/jit/in1k_256 --img_size=256 --subset=in1k

Add the held-out split (from ImageNet val, same classes) for the training diagnostics:
$ python prepare_jit_data.py --data_dir=/data/imagenet --out_dir=/data/jit/in100_128 --img_size=128 --subset=in100 --split=val

Then compute Inception features / FID reference stats (train, and val if present):
$ python fid.py --data_dir=/data/jit/in100_128

For in1k at 256/512 this also downloads JiT's official reference stats (fid_stats_jit.npz), which
train_jit.py/eval_jit.py prefer so FIDs are directly comparable with the paper.

Subsets:
    in1k    all 1000 classes
    in100   every 10th class of the sorted class folders (100 classes, ~128k images)
"""

import os
import json
import argparse
import urllib.request
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

from util.crop import center_crop_arr

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='ImageFolder root containing train/')
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--img_size', type=int, required=True)
parser.add_argument('--subset', type=str, default='in1k', choices=['in1k', 'in100'])
parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='val is written to {out_dir}/val')
parser.add_argument('--workers', type=int, default=os.cpu_count())
args = parser.parse_args()

OFFICIAL_STATS = {
    256: 'https://github.com/LTH14/JiT/raw/main/fid_stats/jit_in256_stats.npz',
    512: 'https://github.com/LTH14/JiT/raw/main/fid_stats/jit_in512_stats.npz',
}

def list_samples():
    # classes always come from train/ so both splits share the same label mapping
    train_dir = os.path.join(args.data_dir, 'train')
    classes = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    assert len(classes) == 1000, f'expected 1000 class folders in {train_dir}, found {len(classes)}'
    imagenet_index = list(range(1000))[::10] if args.subset == 'in100' else list(range(1000))
    if args.subset == 'in100':
        classes = classes[::10]
    split_dir = os.path.join(args.data_dir, args.split)
    samples = []
    for label, cls in enumerate(classes):
        for fname in sorted(os.listdir(os.path.join(split_dir, cls))):
            samples.append((os.path.join(split_dir, cls, fname), label))
    return classes, imagenet_index, samples

def process(chunk):
    start, paths = chunk
    images = np.memmap(os.path.join(args.out_dir, 'images.u8'), dtype=np.uint8, mode='r+',
                       shape=(n_total, args.img_size, args.img_size, 3))
    for i, path in enumerate(paths):
        with Image.open(path) as img:
            images[start + i] = np.asarray(center_crop_arr(img.convert('RGB'), args.img_size))
    images.flush()
    return len(paths)

classes, imagenet_index, samples = list_samples()
n_total = len(samples)
if args.split == 'val':
    args.out_dir = os.path.join(args.out_dir, 'val')

if __name__ == '__main__':
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{args.subset} {args.split}: {len(classes)} classes, {n_total:,} images at {args.img_size}px "
          f"({n_total * args.img_size ** 2 * 3 / 1e9:.1f} GB)")

    np.memmap(os.path.join(args.out_dir, 'images.u8'), dtype=np.uint8, mode='w+',
              shape=(n_total, args.img_size, args.img_size, 3)).flush()
    paths = [p for p, _ in samples]
    chunk_size = 1000
    chunks = [(s, paths[s:s + chunk_size]) for s in range(0, n_total, chunk_size)]
    with Pool(args.workers) as pool:
        with tqdm(total=n_total) as pbar:
            for n in pool.imap_unordered(process, chunks):
                pbar.update(n)

    np.save(os.path.join(args.out_dir, 'labels.npy'), np.array([l for _, l in samples], dtype=np.int16))
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump({'n': n_total, 'img_size': args.img_size, 'num_classes': len(classes), 'classes': classes,
                   'imagenet_index': imagenet_index}, f)

    if args.split == 'train' and args.subset == 'in1k' and args.img_size in OFFICIAL_STATS:
        urllib.request.urlretrieve(OFFICIAL_STATS[args.img_size], os.path.join(args.out_dir, 'fid_stats_jit.npz'))
        print("downloaded JiT's official FID reference stats")
    print(f"done. next: python fid.py --data_dir={os.path.dirname(args.out_dir) if args.split == 'val' else args.out_dir}")
