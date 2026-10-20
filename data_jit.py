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
