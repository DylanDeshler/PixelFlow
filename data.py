"""
ImageNet-1k datasets and augmentations.

Expects the standard ImageFolder layout (see prepare_imagenet.py):
    {data_dir}/train/n01440764/*.JPEG
    {data_dir}/val/n01440764/*.JPEG
"""

import os
import random

import torch
from PIL import ImageFilter, ImageOps
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BICUBIC = transforms.InterpolationMode.BICUBIC

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image with a random radius (as in DINO).
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img

class DataAugmentationDINO(object):
    """
    2 global crops + n local crops. Returns a list: [global1, global2, local1, ..., localn]
    """
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, global_size=224, local_size=96):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_crops_scale, interpolation=BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_crops_scale, interpolation=BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_crops_scale, interpolation=BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

# standard eval preprocessing used by DINO / DINOv2 for k-NN and linear probe evaluation
eval_transform = transforms.Compose([
    transforms.Resize(256, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    normalize,
])

linear_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, interpolation=BICUBIC),
    transforms.RandomHorizontalFlip(),
    normalize,
])

def imagenet(data_dir, split, transform):
    dataset = datasets.ImageFolder(os.path.join(data_dir, split), transform=transform)
    assert len(dataset.classes) == 1000, f'expected 1000 classes in {data_dir}/{split}, found {len(dataset.classes)}'
    return dataset

def make_loader(dataset, batch_size, num_workers, shuffle, drop_last):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

class InfiniteLoader:
    """
    Wraps a DataLoader so the training loop can just call next() forever, tracking epochs.
    """
    def __init__(self, loader):
        self.loader = loader
        self.epoch = 0
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.iterator = iter(self.loader)
            return next(self.iterator)
