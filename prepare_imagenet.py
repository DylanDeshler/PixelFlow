"""
Builds the ImageFolder layout the training/eval scripts expect:
    {out_dir}/train/{class}/*.JPEG
    {out_dir}/val/{class}/*.JPEG

From Hugging Face (ILSVRC/imagenet-1k, gated: accept the terms on the dataset page, then `huggingface-cli login`).
Downloads the train/validation parquet shards (~155GB) into --raw_dir and writes the JPEG bytes out as-is.
Class folders are named 0000..0999 by label index (standard order, 0 = tench), since the parquet files don't carry wnids:
$ python prepare_imagenet.py --source=hf --raw_dir=/data/imagenet_hf --out_dir=/data/imagenet

From the official tarballs (ILSVRC2012_img_train.tar, ILSVRC2012_img_val.tar, ILSVRC2012_devkit_t12.tar.gz in --raw_dir),
class folders are wnids, files are symlinked:
$ python prepare_imagenet.py --source=tarballs --raw_dir=/data/imagenet_raw --out_dir=/data/imagenet

Optionally --short_side=N writes resized JPEG copies instead, which makes JPEG decoding (the usual bottleneck
for multi-crop on one GPU) several times cheaper, at the cost of a small deviation from the standard
full-resolution protocol.
"""

import io
import os
import glob
import argparse
from multiprocessing import Pool

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, required=True, choices=['hf', 'tarballs'])
parser.add_argument('--raw_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--short_side', type=int, default=0, help='resize so the shorter side is this many pixels (0 = keep originals)')
parser.add_argument('--quality', type=int, default=90)
parser.add_argument('--workers', type=int, default=os.cpu_count())
parser.add_argument('--skip_download', action='store_true', help='hf only: use parquet shards already in --raw_dir')
args = parser.parse_args()

HF_SPLITS = {'train': 'train', 'validation': 'val'}
EXPECTED = {'train': 1281167, 'val': 50000}

def resize(img):
    w, h = img.size
    scale = args.short_side / min(w, h)
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    return img

# -----------------------------------------------------------------------------
# hugging face parquet shards

def process_shard(shard_path):
    import pyarrow.parquet as pq
    split = HF_SPLITS[os.path.basename(shard_path).split('-')[0]]
    n = 0
    parquet_file = pq.ParquetFile(shard_path)
    for batch in parquet_file.iter_batches(batch_size=256, columns=['image', 'label']):
        for image, label in zip(batch.column('image').to_pylist(), batch.column('label').to_pylist()):
            dst = os.path.join(args.out_dir, split, f'{label:04d}', os.path.basename(image['path']))
            n += 1
            if os.path.exists(dst):
                continue
            if args.short_side <= 0:
                with open(dst, 'wb') as f:
                    f.write(image['bytes'])
            else:
                resize(Image.open(io.BytesIO(image['bytes'])).convert('RGB')).save(dst, quality=args.quality)
    return split, n

def prepare_hf():
    if not args.skip_download:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id='ILSVRC/imagenet-1k',
            repo_type='dataset',
            allow_patterns=['data/train-*.parquet', 'data/validation-*.parquet'], # the test split has no labels
            local_dir=args.raw_dir,
            max_workers=16,
        )
    shards = sorted(glob.glob(os.path.join(args.raw_dir, 'data', 'train-*.parquet')) + glob.glob(os.path.join(args.raw_dir, 'data', 'validation-*.parquet')))
    print(f"found {len(shards)} parquet shards")

    for split in HF_SPLITS.values():
        for label in range(1000):
            os.makedirs(os.path.join(args.out_dir, split, f'{label:04d}'), exist_ok=True)
    counts = {'train': 0, 'val': 0}
    with Pool(args.workers) as pool:
        for split, n in tqdm(pool.imap_unordered(process_shard, shards), total=len(shards), desc='shards'):
            counts[split] += n
    return counts

# -----------------------------------------------------------------------------
# official tarballs

def process_file(pair):
    src, split, wnid = pair
    dst = os.path.join(args.out_dir, split, wnid, os.path.basename(src))
    if os.path.exists(dst):
        return
    if args.short_side <= 0:
        os.symlink(os.path.abspath(src), dst)
    else:
        resize(Image.open(src).convert('RGB')).save(dst, quality=args.quality)

def prepare_tarballs():
    # torchvision knows how to unpack the official tarballs and sort val by the devkit labels (needs scipy)
    from torchvision.datasets import ImageNet
    pairs = []
    for split in ['train', 'val']:
        dataset = ImageNet(args.raw_dir, split=split)
        for path, label in dataset.samples:
            pairs.append((path, split, dataset.wnids[label]))

    for split, wnid in {(s, w) for _, s, w in pairs}:
        os.makedirs(os.path.join(args.out_dir, split, wnid), exist_ok=True)
    with Pool(args.workers) as pool:
        list(tqdm(pool.imap_unordered(process_file, pairs, chunksize=256), total=len(pairs)))
    return {split: sum(1 for _, s, _ in pairs if s == split) for split in ['train', 'val']}

if __name__ == '__main__':
    counts = prepare_hf() if args.source == 'hf' else prepare_tarballs()
    print(f"{counts['train']:,} train / {counts['val']:,} val images")
    assert counts == EXPECTED, f'unexpected image counts {counts}, expected {EXPECTED}'
    print(f"done, ImageFolder layout written to {args.out_dir}")
