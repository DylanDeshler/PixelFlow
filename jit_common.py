"""
Shared pieces for train_jit.py / eval_jit.py: building JiT's Denoiser from a flat config,
EMA weight swapping, and FID evaluation of generated samples.
"""

from types import SimpleNamespace
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm

from denoiser import Denoiser
from fid import to_uint8


class FastDenoiser(Denoiser):
    """
    JiT's Denoiser, but skips the unconditional forward when guidance is off (cfg == 1),
    which halves sampling cost for the no-CFG FIDs used to track convergence. Identical outputs.
    """
    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        if self.cfg_scale == 1.0:
            x_cond = self.net(z, t.flatten(), labels)
            return (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        return super()._forward_sample(z, t, labels)

def build_denoiser(model, img_size, num_classes, attn_dropout=0.0, proj_dropout=0.0, label_drop_prob=0.1,
                   P_mean=-0.8, P_std=0.8, t_eps=5e-2, noise_scale=1.0, ema_decay1=0.9999, ema_decay2=0.9996,
                   sampling_method='heun', num_sampling_steps=50, cfg=1.0, interval_min=0.0, interval_max=1.0):
    args = SimpleNamespace(
        model=model, img_size=img_size, class_num=num_classes, attn_dropout=attn_dropout, proj_dropout=proj_dropout,
        label_drop_prob=label_drop_prob, P_mean=P_mean, P_std=P_std, t_eps=t_eps, noise_scale=noise_scale,
        ema_decay1=ema_decay1, ema_decay2=ema_decay2, sampling_method=sampling_method,
        num_sampling_steps=num_sampling_steps, cfg=cfg, interval_min=interval_min, interval_max=interval_max,
    )
    return FastDenoiser(args)

def ema_state_dicts(model):
    # same format as JiT's checkpoints: full state dicts with parameters replaced by their EMA values
    sd1, sd2 = dict(model.state_dict()), dict(model.state_dict())
    for i, (name, _) in enumerate(model.named_parameters()):
        sd1[name] = model.ema_params1[i].detach().clone()
        sd2[name] = model.ema_params2[i].detach().clone()
    return sd1, sd2

@contextmanager
def ema_weights(model, which=1):
    """
    Temporarily load EMA parameters into the model (JiT samples with the first EMA by default).
    """
    ema = model.ema_params1 if which == 1 else model.ema_params2
    params = list(model.parameters())
    backup = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, e in zip(params, ema):
            p.copy_(e)
    try:
        yield model
    finally:
        with torch.no_grad():
            for p, b in zip(params, backup):
                p.copy_(b)

@torch.no_grad()
def generate_and_score(model, metrics, ref_stats, num_images, num_classes, device, ctx,
                       cfg=1.0, interval=(0.0, 1.0), steps=50, method='heun', batch_size=256, seed=0, on_batch=None):
    """
    Class-balanced generation (num_images / num_classes per class, as in JiT) and FID/IS against ref_stats.
    A fixed seed gives every evaluation the same noise and labels, so FID differences between runs
    aren't just sampling noise. on_batch(x_uint8, labels) sees every generated batch (e.g. for spectra / saving).
    Returns {'fid', 'is', 'features'} where features are the samples' Inception pool features.
    """
    assert num_images % num_classes == 0, 'number of images per class must be the same'
    was_training = model.training
    model.eval()
    model.cfg_scale, model.cfg_interval, model.steps, model.method = cfg, interval, steps, method
    labels_all = np.arange(num_classes).repeat(num_images // num_classes)

    feats, logits = [], []
    with torch.random.fork_rng(devices=[device] if 'cuda' in str(device) else []):
        torch.manual_seed(seed)
        for start in tqdm(range(0, num_images, batch_size), desc=f'sampling (cfg {cfg})'):
            labels = torch.from_numpy(labels_all[start:start + batch_size]).long().to(device)
            with ctx:
                x = model.generate(labels)
            x = to_uint8(x)
            if on_batch is not None:
                on_batch(x, labels)
            f, l = metrics.features(x)
            feats.append(f)
            logits.append(l)
    model.train(was_training)
    feats = torch.cat(feats)
    fid, isc = metrics.compute(feats, torch.cat(logits), ref_stats)
    return {'fid': fid, 'is': isc, 'features': feats}

def save_labeled_grid(x, names, path, nrow=8):
    """
    Saves a grid of images ([B, 3, H, W] in [-1, 1]) with each image's class name written underneath.
    """
    from PIL import Image, ImageDraw
    imgs = to_uint8(x).permute(0, 2, 3, 1).cpu().numpy()
    B, H, W, _ = imgs.shape
    caption = 12
    ncol = min(nrow, B)
    nrows = (B + ncol - 1) // ncol
    canvas = Image.new('RGB', (ncol * W, nrows * (H + caption)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (img, name) in enumerate(zip(imgs, names)):
        r, c = divmod(i, ncol)
        canvas.paste(Image.fromarray(img), (c * W, r * (H + caption)))
        draw.text((c * W + 2, r * (H + caption) + H), name[: max(4, W // 6)], fill='black')
    canvas.save(path)
