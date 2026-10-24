"""
Pre-cropped uint8 memmap datasets for JiT training.

JiT's training augmentation is just ADM center crop + random horizontal flip, so the crop
can be done once offline (prepare_jit_data.py) and flips applied per batch. Training then reads
raw uint8 arrays with no JPEG decoding, which keeps a single GPU fed easily.

Layout of a prepared dataset dir:
    images.u8       uint8 memmap [N, R, R, 3]
    labels.npy      int16 [N], remapped to 0..num_classes-1
    meta.json       {"n", "img_size", "num_classes", "classes", "imagenet_index"}
    fid_stats.npz   reference Inception stats (fid.py), and fid_stats_jit.npz for the official ImageNet 256/512 stats
"""

import os
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


class MemmapImages:
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'meta.json')) as f:
            self.meta = json.load(f)
        self.n = self.meta['n']
        self.img_size = self.meta['img_size']
        self.num_classes = self.meta['num_classes']
        self.images = np.memmap(os.path.join(data_dir, 'images.u8'), dtype=np.uint8, mode='r',
                                shape=(self.n, self.img_size, self.img_size, 3))
        self.labels = np.load(os.path.join(data_dir, 'labels.npy')).astype(np.int64)

    def __len__(self):
        return self.n

    def class_names(self):
        """
        Readable name per label, from torchvision's ImageNet-1k category list (standard class order, which both the
        HF label indices and sorted wnid folders follow). Older meta.json files without 'imagenet_index' assume in1k / every-10th in100.
        """
        from torchvision.models import ResNet50_Weights
        categories = ResNet50_Weights.IMAGENET1K_V1.meta['categories']
        index = self.meta.get('imagenet_index') or list(range(0, 1000, 1000 // self.num_classes))
        return [categories[i] for i in index]

    def get(self, idx):
        idx = np.sort(idx) # sorted reads are much friendlier to the page cache / disk
        return self.images[idx], self.labels[idx]

class EpochSampler:
    """
    Shuffled without-replacement batches, reshuffling every epoch (drop_last semantics).
    """
    def __init__(self, n, batch_size, seed=0):
        self.n = n
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self.perm = self.rng.permutation(n)
        self.pos = 0

    def next(self):
        if self.pos + self.batch_size > self.n:
            self.epoch += 1
            self.perm = self.rng.permutation(self.n)
            self.pos = 0
        idx = self.perm[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return idx

class Prefetcher:
    """
    Background batch reader for a MemmapImages training set, for datasets larger than the page cache. A thread draws the
    next batches from the sampler (depth ahead) and fills reusable pinned buffers through a pool of reader threads with
    os.preadv, one read per run of consecutive rows: file reads release the GIL, so reading overlaps the whole training
    step (not just the forward), and many reads in flight reach the disk's throughput instead of paying its latency
    once per image. The kernel is told the access is random, so it doesn't read ahead (wasted bandwidth on random rows).
    next() returns what to_model_input(*data.get(sampler.next())) would: images [B, 3, R, R] in [-1, 1] on device
    (with random flips) and labels, rows in sorted index order as MemmapImages.get returns them.
    """
    def __init__(self, data, sampler, device, flip=True, workers=16, depth=3):
        self.data, self.sampler, self.device, self.flip, self.workers = data, sampler, device, flip, workers
        self.cuda = 'cuda' in str(device)
        self.row = data.img_size * data.img_size * 3
        self.fd = os.open(data.images.filename, os.O_RDONLY)
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        shape = (sampler.batch_size, data.img_size, data.img_size, 3)
        self.buffers = [torch.empty(shape, dtype=torch.uint8, pin_memory=self.cuda) for _ in range(depth + 1)]
        self.copied = [None] * len(self.buffers) # CUDA event after each buffer's last host-to-device copy
        self.free, self.ready = queue.Queue(), queue.Queue()
        for k in range(len(self.buffers)):
            self.free.put(k)
        self.pool = ThreadPoolExecutor(workers)
        threading.Thread(target=self._run, daemon=True).start()

    def _read_rows(self, out, idx):
        # rows idx (sorted) of the file into out (contiguous uint8 rows), one preadv per run of consecutive rows
        i = 0
        while i < len(idx):
            j = i + 1
            while j < len(idx) and idx[j] == idx[j - 1] + 1:
                j += 1
            view, offset = memoryview(out[i:j].reshape(-1)), int(idx[i]) * self.row
            done = 0
            while done < len(view):
                n = os.preadv(self.fd, [view[done:]], offset + done)
                if n <= 0:
                    raise IOError(f'short read at byte {offset + done} of {self.data.images.filename}')
                done += n
            i = j

    def _run(self):
        try:
            while True:
                k = self.free.get()
                if self.copied[k] is not None:
                    self.copied[k].synchronize() # the GPU has finished copying this buffer's previous batch
                idx = np.sort(self.sampler.next())
                out = self.buffers[k].numpy()
                parts = np.array_split(np.arange(len(idx)), min(self.workers, len(idx)))
                for f in [self.pool.submit(self._read_rows, out[p[0]:p[-1] + 1], idx[p[0]:p[-1] + 1]) for p in parts if len(p)]:
                    f.result()
                self.ready.put((k, self.data.labels[idx]))
        except BaseException as e: # surfaced by next()
            self.ready.put((None, e))

    def next(self):
        k, labels = self.ready.get()
        if k is None:
            raise RuntimeError('background data loading failed') from labels
        if self.cuda:
            x = self.buffers[k].to(self.device, non_blocking=True)
            self.copied[k] = torch.cuda.Event()
            self.copied[k].record()
        else:
            x = self.buffers[k].clone()
        self.free.put(k)
        x = x.permute(0, 3, 1, 2)
        if self.flip:
            mask = torch.rand(x.shape[0], device=self.device) < 0.5
            x = torch.where(mask.view(-1, 1, 1, 1), x.flip(-1), x)
        return x.float().div_(255).mul_(2).sub_(1), torch.from_numpy(labels).to(self.device, non_blocking=True)

def to_model_input(x_uint8, flip, device):
    """
    uint8 [B, R, R, 3] numpy -> float [B, 3, R, R] in [-1, 1] on device, with optional random horizontal flips.
    """
    x = torch.from_numpy(x_uint8).pin_memory() if 'cuda' in str(device) else torch.from_numpy(x_uint8)
    x = x.to(device, non_blocking=True).permute(0, 3, 1, 2)
    if flip:
        mask = torch.rand(x.shape[0], device=device) < 0.5
        x = torch.where(mask.view(-1, 1, 1, 1), x.flip(-1), x)
    return x.float().div_(255).mul_(2).sub_(1)
