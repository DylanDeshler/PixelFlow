"""
Audit of a prepared JiT dataset (prepare_jit_data.py output) against its source ImageFolder. CPU only (JPEG headers,
no decoding of pixels beyond the header), safe to run beside training.

Reports:
  source sizes      short-side histogram of the original JPEGs; how many are upsampled by the ADM center crop
                    (short side < img_size), i.e. trained on as bicubic-interpolated blur
  JPEG quality      estimated IJG quality from each file's luminance quantization table (low quality = blocking
                    artifacts the model learns); a re-encoded copy (prepare_imagenet --short_side) shows up as a
                    spike at one quality value
  consistency       memmap image / label counts vs the source, per-class counts, FID reference stats and the
                    Inception feature file built from the same memmap
Writes {out}.json with per-image short side and quality (index-aligned with the memmap) for later filtering.

$ python audit_data.py --data_dir=/data/in1k_128 --source_dir=/data/imagenet
"""

import os
import json
import argparse
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='prepared memmap dataset (prepare_jit_data.py --out_dir)')
parser.add_argument('--source_dir', type=str, required=True, help='ImageFolder root it was built from (has train/)')
parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() // 4))
parser.add_argument('--out', type=str, default='audit_data')
args = parser.parse_args()

# IJG standard luminance table (JPEG Annex K), natural order
STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)

def ijg_quality(table):
    # invert IJG scaling: q < 50 -> scale = 5000 / q, else 200 - 2q (percent); estimate from the mean ratio to the standard table
    t = np.asarray(table, dtype=np.float64)
    if t.size != 64:
        return -1
    scale = 100.0 * np.median(t / STD_LUMA) # percent
    q = 5000.0 / scale if scale > 100 else (200.0 - scale) / 2.0
    return int(round(min(max(q, 1), 100)))

def header(path):
    try:
        with Image.open(path) as img:
            w, h = img.size
            q = -1
            if img.format == 'JPEG' and getattr(img, 'quantization', None):
                q = ijg_quality(img.quantization[0]) # table 0 = luminance; PIL gives it in natural order
            return min(w, h), q, img.format or '?', img.mode
    except Exception:
        return -1, -1, 'error', '?'

if __name__ == '__main__':
    meta = json.load(open(os.path.join(args.data_dir, 'meta.json') if args.split == 'train' else os.path.join(args.data_dir, 'val', 'meta.json')))
    split_mm = args.data_dir if args.split == 'train' else os.path.join(args.data_dir, 'val')
    R, classes = meta['img_size'], meta['classes']
    # same listing order as prepare_jit_data.list_samples
    paths, labels = [], []
    for label, cls in enumerate(classes):
        d = os.path.join(args.source_dir, args.split, cls)
        for f in sorted(os.listdir(d)):
            paths.append(os.path.join(d, f)); labels.append(label)
    labels = np.array(labels)
    print(f'{args.split}: {len(paths):,} source images, memmap n = {meta["n"]:,}, img_size {R}')

    # consistency
    mm_labels = np.load(os.path.join(split_mm, 'labels.npy'))
    print(f'  counts match: {len(paths) == meta["n"] == len(mm_labels)}; labels match source order: {np.array_equal(mm_labels, labels)}')
    per_class = np.bincount(mm_labels, minlength=len(classes))
    print(f'  per-class images: min {per_class.min()} max {per_class.max()} median {int(np.median(per_class))}; '
          f'classes under 1000 images: {(per_class < 1000).sum()}')
    if args.split == 'train':
        for name in ('fid_stats.npz', 'fid_stats_jit.npz', 'train_inception.npy'):
            p = os.path.join(args.data_dir, name)
            if os.path.exists(p):
                extra = ''
                if name.endswith('.npy'):
                    extra = f', rows {np.load(p, mmap_mode="r").shape[0]:,} (memmap {meta["n"]:,})'
                stale = os.path.getmtime(p) < os.path.getmtime(os.path.join(args.data_dir, 'images.u8'))
                print(f'  {name}: present{extra}{" -- OLDER THAN images.u8 (stale?)" if stale else ""}')
            else:
                print(f'  {name}: missing')

    # headers
    with Pool(args.workers) as pool:
        res = list(tqdm(pool.imap(header, paths, chunksize=512), total=len(paths), desc='JPEG headers'))
    short = np.array([r[0] for r in res]); qual = np.array([r[1] for r in res])
    fmts = {}
    for r in res:
        fmts[(r[2], r[3])] = fmts.get((r[2], r[3]), 0) + 1
    n = len(short)
    print(f'\nformats / modes: {dict(sorted(fmts.items(), key=lambda kv: -kv[1]))}')
    print('source short side:')
    for lo, hi in ((0, 64), (64, 96), (96, R), (R, 2 * R), (2 * R, 4 * R), (4 * R, 10 ** 9)):
        m = (short >= lo) & (short < hi)
        print(f'  [{lo:>4}, {hi if hi < 10 ** 9 else "inf":>4}): {m.sum():>9,} ({100 * m.mean():6.3f}%)')
    up = (short > 0) & (short < R)
    print(f'  upsampled by the center crop (short side < {R}): {up.sum():,} ({100 * up.mean():.3f}%); '
          f'classes with > 5% upsampled: {int((np.bincount(labels[up], minlength=len(classes)) / np.maximum(per_class, 1) > 0.05).sum())}')
    print(f'  median short side {int(np.median(short[short > 0]))}')
    qv = qual[qual > 0]
    if len(qv):
        print('estimated JPEG quality (IJG, luminance table):')
        for lo, hi in ((1, 50), (50, 70), (70, 85), (85, 95), (95, 101)):
            m = (qv >= lo) & (qv < hi)
            print(f'  [{lo:>3}, {hi:>3}): {m.sum():>9,} ({100 * m.mean():6.3f}%)')
        vals, cnts = np.unique(qv, return_counts=True)
        top = np.argsort(-cnts)[:5]
        print(f'  most common values: {", ".join(f"q{vals[i]} {100 * cnts[i] / len(qv):.1f}%" for i in top)}'
              f'{"  <- one value dominates: likely re-encoded copies (prepare_imagenet --short_side)" if cnts.max() > 0.5 * len(qv) else ""}')
    with open(args.out + '.json', 'w') as f:
        json.dump({'split': args.split, 'img_size': R, 'short_side': short.tolist(), 'quality': qual.tolist()}, f)
    print(f'\nwrote {args.out}.json (per-image short side / quality, memmap order)')
