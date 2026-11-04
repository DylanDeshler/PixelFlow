"""
Latent-space baseline: a SiT-style flow-matching transformer on Stable Diffusion VAE latents (prepare_latents.py), at the
same scale as JiT-B/16 on 128px pixels, evaluated exactly like train_jit.py (FID-10k with the same seed and sampler, the
same probes, the same diagnostic file names), so pixel and latent runs compare directly.

Network: JiT-B's blocks (model_jit.JiT: adaLN-zero, SwiGLU, RMSNorm, QK-norm, 2D RoPE) on 4 x 16 x 16 latents with 2 x 2
patches: 64 tokens at width 768 and depth 12, the same token count, width and depth as JiT-B/16 on 128px, so data space and
prediction target are the differences. (The patch embedding's 128-d bottleneck doesn't compress here: patches are 16-d.)
Objective: rectified flow z_t = t x + (1 - t) s e (t = 1 clean, as JiT), v-prediction as in SiT: the network outputs the
velocity v = x - s e and the loss is ||v - v_hat||^2, with no clamp near t = 1. Sampling: Heun with JiT's grid (50 steps,
the last Euler), the same classifier-free guidance interval rule.
Recipes (--recipe; any setting can still be overridden):
  jit  the JiT baseline's training: Muon on the block matrices (Moonshot's RMS matching, fused matrices orthogonalized per
       component) + AdamW, blr 2e-4 per 1024 images, betas (0.9, 0.95), 5-epoch linear warm-up then constant, t logit-normal
       (-0.8, 0.8), gradient clipping 1.0: compares with runs/in1k_128/jit-b16_icl0/muon (FID-10k 81.0 at 20 epochs)
  sit  SiT's: AdamW lr 1e-4 (SiT's value at batch 256, not rescaled), betas (0.9, 0.999), constant from the start, t uniform,
       no clipping
Both: label dropout 0.1, EMA 0.9999 / 0.9996 (with train_jit.py's warm-up), noise scale 1 (the scaled latents have about
unit variance), batch 1024.
Dispersive Loss (Wang & He 2025) via repa.DispCapture / dispersive_loss with --disp_weight > 0; the defaults are the paper's
(plain InfoNCE-L2, every token flattened, block depth // 4 = 3, lambda 0.5 given by --disp_weight, tau 0.5).
Evaluation every eval_every_epochs: FID-10k / IS of the EMA weights (latents decoded by the VAE), held-out v-loss, and kNN /
linear probes of mean-pooled block outputs at five noise levels (null label, EMA weights) on the same 10k train / 5k held-out
images as train_jit.py. Writes {out_dir}/diagnostics/diag_{iter}.json with train_jit.py's scalar names ('val/fid',
'eval/val_loss', 'probe/knn_block08_t0.50', ...) and {out_dir}/train_log.jsonl.

$ python prepare_latents.py --data_dir=/data/in1k_128
$ python train_latent.py --data_dir=/data/in1k_128                                  # recipe jit, 20 epochs
$ python train_latent.py --data_dir=/data/in1k_128 --disp_weight=0.5                # + Dispersive Loss, the paper's settings
$ python train_latent.py --data_dir=/data/in1k_128 --recipe=sit
"""

import os
import sys
import json
import math
import time
from contextlib import nullcontext, contextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from data_jit import MemmapImages, EpochSampler
from jit_common import generate_and_score
from fid import InceptionMetrics, load_ref_stats
from model_jit import JiT
from muon import Muon, CombinedOptimizer
from repa import DispCapture, dispersive_loss
import diagnostics as diag_lib
from util.progress import bar

sys.stdout.reconfigure(line_buffering=True) # logs redirected to a file show each line as it's printed

# -----------------------------------------------------------------------------
data_dir = '' # the prepared pixel dataset (FID reference stats, labels, class count); latents in latent_dir
latent_dir = '' # default: {data_dir}/latents_sdvae-ft-ema
runs_root = 'runs'
out_dir = '' # default: {runs_root}/{dataset}/latent-b2/{recipe}{_changes}
init_from = 'scratch' # 'scratch' or 'resume'
recipe = 'jit' # 'jit' or 'sit' (see above); settings left at their sentinel take the recipe's value
epochs = 20
batch_size = 1024
optimizer_type = '' # 'muon' / 'adamw' ('' = recipe)
blr = -1.0 # lr per 1024 images ('' / < 0 = recipe)
beta2 = -1.0
warmup_epochs = -1.0
t_sampling = '' # 'logit_normal' or 'uniform'
P_mean = -0.8 # logit-normal t
P_std = 0.8
grad_clip = -1.0 # 0 = off
weight_decay = 0.0
beta1 = 0.9
muon_momentum = 0.95
muon_split_fused = True # as train_jit.py: qkv / SwiGLU w12 / adaLN orthogonalized per component
noise_scale = 1.0
label_drop_prob = 0.1
ema_decay1 = 0.9999
ema_decay2 = 0.9996
# network (JiT blocks at B scale on 2 x 2 latent patches)
depth = 12
hidden_size = 768
num_heads = 12
patch_size = 2
bottleneck_dim = 128
attn_res = 'none' # as train_jit.py ('full' / 'block'), off for the baseline
adaln_single = False
attn_res_chunk = 1
# Dispersive Loss (off unless disp_weight > 0); defaults = the paper's
disp_weight = 0.0
disp_temp = 0.5
disp_blocks = '' # '' -> depth // 4
disp_kind = 'infonce_l2'
disp_pool = 'none'
disp_group = 0
# evaluation
eval_every_epochs = 5
fid_num_images = 10000
fid_cfg = 1.0
fid_interval_min = 0.1
fid_interval_max = 1.0
num_sampling_steps = 50
gen_batch_size = 256
probe_blocks = '4,8,12'
probe_ts = '1.0,0.75,0.5,0.25,0.1'
probe_train_images = 10000
probe_val_images = 5000
eval_loss_images = 4096
decode = 'vae' # 'vae' (the latents' VAE) or 'debug' (no VAE: the first 3 latent channels upsampled, for CPU smoke tests only)
save_every_epochs = 1
log_interval = 100
seed = 0
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
_defaults = {k: globals()[k] for k in config_keys}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

RECIPES = {
    'jit': dict(optimizer_type='muon', blr=2e-4, beta2=0.95, warmup_epochs=5.0, t_sampling='logit_normal', grad_clip=1.0),
    'sit': dict(optimizer_type='adamw', blr=1e-4, beta2=0.999, warmup_epochs=0.0, t_sampling='uniform', grad_clip=0.0), # SiT's 1e-4, unscaled
}
assert recipe in RECIPES, f"recipe must be one of {list(RECIPES)}, got {recipe}"
for k, v in RECIPES[recipe].items(): # the recipe fills what wasn't set explicitly
    if globals()[k] in ('', -1.0):
        globals()[k] = v
config = {k: globals()[k] for k in config_keys}
assert data_dir, 'pass --data_dir (the prepared pixel dataset; latents from prepare_latents.py)'
assert t_sampling in ('logit_normal', 'uniform') and optimizer_type in ('muon', 'adamw')
latent_dir = latent_dir or os.path.join(data_dir, 'latents_sdvae-ft-ema')
lmeta = json.load(open(os.path.join(latent_dir, 'meta.json')))
C, H, W = lmeta['latent_shape']
dataset = os.path.basename(os.path.normpath(data_dir))
if not out_dir:
    changed = [f'{k}{v:g}' if isinstance(v, float) else f'{k}{v}' for k, v in config.items()
               if k not in ('data_dir', 'latent_dir', 'runs_root', 'out_dir', 'init_from', 'recipe', 'epochs', 'eval_every_epochs',
                            'save_every_epochs', 'log_interval', 'device', 'dtype', 'compile', 'decode', 'attn_res_chunk', 'gen_batch_size',
                            'fid_num_images', 'fid_cfg', 'fid_interval_min', 'fid_interval_max', 'num_sampling_steps', 'probe_blocks',
                            'probe_ts', 'probe_train_images', 'probe_val_images', 'eval_loss_images', 'seed') # (evaluation / system settings)
               and v != dict(_defaults, **RECIPES[recipe])[k]] # settings that differ from the recipe's (or the script's) values
    out_dir = os.path.join(runs_root, dataset, 'latent-b2', '_'.join([recipe] + changed))
os.makedirs(os.path.join(out_dir, 'diagnostics'), exist_ok=True)
print(f"data: {data_dir} (latents {latent_dir}: {C} x {H} x {W}, {lmeta['vae']})\nrun:  {out_dir}")
with open(os.path.join(out_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' or dtype == 'float32') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
if not compile:
    torch._dynamo.config.disable = True

# data: latents [N, 2 flips, C, H, W] (memmap) with the pixel dataset's labels
pix = MemmapImages(data_dir)
num_classes = pix.num_classes
class LatentData:
    def __init__(self, split_dir, labels):
        self.z = np.load(os.path.join(split_dir, 'latents.npy'), mmap_mode='r')
        self.labels = labels
        assert len(self.z) == len(labels), f'{split_dir}: {len(self.z)} latents vs {len(labels)} labels'

    def __len__(self):
        return len(self.labels)

    def get(self, idx, flip, rng=None):
        # a batch of latents (float32 numpy [B, C, H, W]) and labels; flip: a random flip per sample (the stored flipped copy)
        idx = np.sort(idx)
        z = np.asarray(self.z[idx]) # [B, 2, C, H, W]
        f = rng.integers(0, 2, len(idx)) if flip else np.zeros(len(idx), dtype=np.int64)
        return z[np.arange(len(idx)), f].astype(np.float32), self.labels[idx]
train_data = LatentData(os.path.join(latent_dir, 'train'), pix.labels)
val_pix = MemmapImages(os.path.join(data_dir, 'val'))
val_data = LatentData(os.path.join(latent_dir, 'val'), val_pix.labels)
iters_per_epoch = len(train_data) // batch_size
max_iters = epochs * iters_per_epoch
lr = blr * batch_size / 1024
print(f"recipe {recipe}: {optimizer_type}, lr {lr:.2e}, betas ({beta1}, {beta2}), warm-up {warmup_epochs:g} epochs, t {t_sampling}, "
      f"clip {grad_clip:g}; {len(train_data):,} images, {iters_per_epoch} iterations per epoch, {epochs} epochs")

# network: v-prediction (the output is the velocity)
net = JiT(input_size=H, patch_size=patch_size, in_channels=C, hidden_size=hidden_size, depth=depth, num_heads=num_heads,
          num_classes=num_classes, bottleneck_dim=bottleneck_dim, in_context_len=0, in_context_start=0, attn_res=attn_res,
          adaln_single=adaln_single).to(device)
net.attn_res_chunk = attn_res_chunk
print(f"network: JiT blocks, {depth} x {hidden_size}, {(H // patch_size) * (W // patch_size)} tokens of {patch_size}x{patch_size}x{C} latent patches, "
      f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M params")
disp_captures = []
if disp_weight > 0:
    idx = list(range(1, depth + 1)) if disp_blocks == 'all' else [int(b) for b in disp_blocks.split(',')] if disp_blocks else [max(1, depth // 4)]
    disp_captures = [DispCapture(net, b) for b in idx] # before compiling, so the hooks are traced in
    print(f"dispersive loss ({disp_kind}): weight {disp_weight}, temperature {disp_temp}, block(s) {idx}, pooling {disp_pool}"
          + (f", groups of {disp_group}" if disp_group else ""))

def muon_split(name):
    if not muon_split_fused:
        return 1
    return 3 if name.endswith('attn.qkv.weight') else 2 if name.endswith('mlp.w12.weight') else \
        6 if (name.endswith('adaLN_modulation.1.weight') or name.endswith('t_block.1.weight')) else 1
muon_named = {id(p): n for n, p in net.blocks.named_parameters() if optimizer_type == 'muon' and p.ndim >= 2 and 'scale_shift_table' not in n}
if optimizer_type == 'muon' and adaln_single:
    muon_named[id(net.t_block[1].weight)] = 't_block.1.weight'
muon_params, decay, no_decay = {}, [], []
for n, p in net.named_parameters():
    if id(p) in muon_named:
        muon_params.setdefault(muon_split(muon_named[id(p)]), []).append(p)
    else:
        (no_decay if p.ndim == 1 or n.endswith('.bias') or 'scale_shift_table' in n else decay).append(p)
adamw = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0.0, 'peak_lr': lr}, {'params': decay, 'weight_decay': weight_decay, 'peak_lr': lr}],
                          lr=lr, betas=(beta1, beta2), fused=device_type == 'cuda')
if optimizer_type == 'muon':
    muon = Muon([{'params': ps, 'split': k, 'peak_lr': lr} for k, ps in sorted(muon_params.items())], lr=lr, momentum=muon_momentum,
                weight_decay=weight_decay, adjust_lr='match_rms_adamw')
    optimizer = CombinedOptimizer([muon, adamw])
else:
    optimizer = adamw

ema1 = [p.detach().clone() for p in net.parameters()]
ema2 = [p.detach().clone() for p in net.parameters()]
iter_num, best_fid, train_seconds = 0, float('inf'), 0.0
if init_from == 'resume':
    ck = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
    net.load_state_dict(ck['model'])
    ema1 = [ck['ema1'][n].to(device) for n, _ in net.named_parameters()]
    ema2 = [ck['ema2'][n].to(device) for n, _ in net.named_parameters()]
    optimizer.load_state_dict(ck['optimizer'])
    iter_num, best_fid, train_seconds = ck['iter_num'], ck['best_fid'], ck.get('train_seconds', 0.0)
    print(f"resumed at iter {iter_num} (epoch {iter_num / iters_per_epoch:.2f})")
    del ck
if compile:
    print("compiling the model... (the first iteration and the first eval take a few minutes)")
    net.compile()

def sample_t(n):
    if t_sampling == 'uniform':
        return torch.rand(n, device=device)
    return torch.sigmoid(torch.randn(n, device=device) * P_std + P_mean)

def get_lr(it):
    e = it / iters_per_epoch
    return lr * min(1.0, e / warmup_epochs) if warmup_epochs > 0 else lr

@contextmanager
def ema_weights(which):
    params = list(net.parameters())
    backup = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, e in zip(params, ema1 if which == 1 else ema2):
            p.copy_(e)
    try:
        yield
    finally:
        with torch.no_grad():
            for p, b in zip(params, backup):
                p.copy_(b)

# decoding latents to images for FID
SCALE = lmeta['scale']
if decode == 'vae':
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(lmeta['vae']).to(device).eval()
    def to_images(z):
        with ctx:
            return vae.decode(z / SCALE).sample.float().clamp(-1, 1)
else: # CPU smoke tests only
    def to_images(z):
        x = torch.nn.functional.interpolate(z[:, :3], scale_factor=lmeta['img_size'] // H, mode='nearest')
        return torch.tanh(x)

class LatentSampler:
    """
    What jit_common.generate_and_score drives: generate(labels) -> images in [-1, 1] (Heun on the network's velocity with JiT's
    grid and guidance rule, from t = 0, z = s e, to t = 1, then the VAE decoder).
    """
    def __init__(self):
        self.cfg_scale, self.cfg_interval, self.steps, self.method, self.eta, self.fixed_shape, self.training = 1.0, (0.0, 1.0), 50, 'heun', 1.0, False, False

    def eval(self):
        self.training = False
        net.eval()

    def train(self, mode=True):
        self.training = mode
        net.train(mode)

    def velocity(self, z, t, labels):
        tt = torch.full((len(z),), float(t), device=z.device)
        low, high = self.cfg_interval
        if self.cfg_scale == 1.0 or not (t < high and (low == 0 or t > low)):
            return net(z, tt, labels).float()
        both = net(torch.cat([z, z]), torch.cat([tt, tt]), torch.cat([labels, torch.full_like(labels, num_classes)])).float()
        vc, vu = both.chunk(2)
        return vu + self.cfg_scale * (vc - vu)

    @torch.no_grad()
    def generate(self, labels):
        z = noise_scale * torch.randn(len(labels), C, H, W, device=labels.device)
        ts = torch.linspace(0.0, 1.0, self.steps + 1).tolist()
        for i in range(self.steps):
            t, t1 = ts[i], ts[i + 1]
            v = self.velocity(z, t, labels)
            if i < self.steps - 1 and self.method == 'heun': # Heun except the last step (Euler), as JiT
                v = 0.5 * (v + self.velocity(z + (t1 - t) * v, t1, labels))
            z = z + (t1 - t) * v
        return to_images(z)
sampler_model = LatentSampler()

ref_path = os.path.join(data_dir, 'fid_stats_jit.npz')
ref_stats = load_ref_stats(ref_path if os.path.exists(ref_path) else os.path.join(data_dir, 'fid_stats.npz'))
metrics = InceptionMetrics(device)
spaced = lambda n, k: np.linspace(0, n - 1, min(k, n)).round().astype(np.int64) # as train_jit.py: the same images
pb = [int(b) for b in probe_blocks.split(',') if int(b) <= depth]
pts = [float(t) for t in probe_ts.split(',')]
probe_train = train_data.get(spaced(len(train_data), probe_train_images), flip=False)
probe_val = val_data.get(spaced(len(val_data), probe_val_images), flip=False)
loss_val = val_data.get(spaced(len(val_data), eval_loss_images), flip=False)

@torch.no_grad()
def probe_features(z_np, desc):
    # mean-pooled block outputs at each probe noise level, null label, same noise for every t (as diagnostics.probe_features)
    feats, store = {}, {}
    hooks = [net.blocks[b - 1].register_forward_hook(lambda m, i, o, b=b: store.__setitem__(b, o.float().mean(1))) for b in pb]
    try:
        for t in bar(pts, f'{desc} ({len(z_np)} latents x {len(pts)} noise levels)'):
            g = torch.Generator(device=device).manual_seed(seed)
            out = {b: [] for b in pb}
            for i in range(0, len(z_np), 256):
                x = torch.from_numpy(z_np[i:i + 256]).to(device)
                z = t * x + (1 - t) * torch.randn(x.shape, generator=g, device=device) * noise_scale
                with ctx:
                    net.forward(z, torch.full((len(x),), t, device=device), torch.full((len(x),), num_classes, device=device, dtype=torch.long))
                for b in pb:
                    out[b].append(store[b])
            for b in pb:
                feats[(b, t)] = torch.cat(out[b])
    finally:
        for h in hooks:
            h.remove()
    return feats

@torch.no_grad()
def val_loss():
    # held-out v-loss with fixed noise and t (raw weights, no label dropout)
    g = torch.Generator(device=device).manual_seed(seed)
    z_np, y_np = loss_val
    tot = 0.0
    net.eval()
    for i in range(0, len(z_np), 256):
        x = torch.from_numpy(z_np[i:i + 256]).to(device)
        y = torch.from_numpy(y_np[i:i + 256]).to(device)
        t = (torch.sigmoid(torch.randn(len(x), generator=g, device=device) * P_std + P_mean) if t_sampling == 'logit_normal'
             else torch.rand(len(x), generator=g, device=device))
        e = torch.randn(x.shape, generator=g, device=device) * noise_scale
        tt = t.view(-1, 1, 1, 1)
        with ctx:
            v_hat = net(tt * x + (1 - tt) * e, t, y).float()
        tot += (x - e - v_hat).pow(2).mean().item() * len(x)
    net.train()
    return tot / len(z_np)

def evaluate(step):
    t0 = time.time()
    log = {'eval/val_loss': val_loss()}
    with ema_weights(1):
        out = generate_and_score(sampler_model, metrics, ref_stats, fid_num_images, num_classes, device, ctx, cfg=fid_cfg,
                                 interval=(fid_interval_min, fid_interval_max), steps=num_sampling_steps, method='heun', batch_size=gen_batch_size)
        net.eval()
        tr, va = probe_features(probe_train[0], 'probe features, train'), probe_features(probe_val[0], 'probe features, held-out')
        net.train()
    probes = diag_lib.run_probes(tr, torch.from_numpy(probe_train[1]).to(device), va, torch.from_numpy(probe_val[1]).to(device), num_classes)
    log.update({'val/fid': out['fid'], 'val/is': out['is']})
    for (b, t), r in probes.items():
        log[f'probe/knn_block{b:02d}_t{t:.2f}'] = r['knn']
        log[f'probe/linear_block{b:02d}_t{t:.2f}'] = r['linear']
    with open(os.path.join(out_dir, 'diagnostics', f'diag_{step}.json'), 'w') as f:
        json.dump({'iter': step, 'scalars': log, 'probes': {f'block{b}_t{t}': r for (b, t), r in probes.items()}}, f)
    bmid = pb[len(pb) // 2]
    print(f"iter {step} (epoch {step / iters_per_epoch:.2f}): val loss {log['eval/val_loss']:.4f}, FID-{fid_num_images / 1000:g}k {out['fid']:.2f}, "
          f"IS {out['is']:.1f} (cfg {fid_cfg}); probes block {bmid} t=0.5: kNN {probes[(bmid, 0.5)]['knn']:.2f}, linear {probes[(bmid, 0.5)]['linear']:.2f} "
          f"(eval {(time.time() - t0) / 60:.1f} min)")
    return out['fid']

def save_checkpoint():
    names = [n for n, _ in net.named_parameters()]
    torch.save({'model': net.state_dict(), 'ema1': dict(zip(names, [e.cpu() for e in ema1])), 'ema2': dict(zip(names, [e.cpu() for e in ema2])),
                'optimizer': optimizer.state_dict(), 'iter_num': iter_num, 'best_fid': best_fid, 'train_seconds': train_seconds,
                'config': config}, os.path.join(out_dir, 'ckpt.pt.tmp'))
    os.replace(os.path.join(out_dir, 'ckpt.pt.tmp'), os.path.join(out_dir, 'ckpt.pt'))

# training
sampler = EpochSampler(len(train_data), batch_size, seed=seed)
sampler.epoch = iter_num // iters_per_epoch
rng = np.random.default_rng(seed + iter_num)
pool = ThreadPoolExecutor(1) # reads the next batch while the GPU trains on this one
fetch = lambda: train_data.get(sampler.next(), flip=True, rng=rng)
pending = pool.submit(fetch)
eval_iters, save_iters = eval_every_epochs * iters_per_epoch, save_every_epochs * iters_per_epoch
params = list(net.parameters())
t_last = time.time()
while True:
    if iter_num > 0 and (iter_num % eval_iters == 0 or iter_num == max_iters):
        best_fid = min(best_fid, evaluate(iter_num))
        t_last = time.time()
    if iter_num > 0 and (iter_num % save_iters == 0 or iter_num == max_iters):
        save_checkpoint()
    if iter_num >= max_iters:
        break
    for g in optimizer.param_groups:
        g['lr'] = g['peak_lr'] * get_lr(iter_num) / lr
    x_np, y_np = pending.result()
    pending = pool.submit(fetch)
    x = torch.from_numpy(x_np).to(device, non_blocking=True)
    y = torch.from_numpy(y_np).to(device, non_blocking=True)
    y = torch.where(torch.rand(len(y), device=device) < label_drop_prob, torch.full_like(y, num_classes), y)
    t = sample_t(len(x)).view(-1, 1, 1, 1)
    e = torch.randn_like(x) * noise_scale
    with ctx:
        v_hat = net(t * x + (1 - t) * e, t.flatten(), y)
        loss = (x - e - v_hat.float()).pow(2).mean() # ||v - v_hat||^2, v = x - s e
        loss_disp = (sum(dispersive_loss(c.out, disp_temp, disp_kind, pool=disp_pool, group=disp_group) for c in disp_captures)
                     / len(disp_captures)) if disp_captures else None
        total = loss + disp_weight * loss_disp if disp_captures else loss
    total.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(params, grad_clip if grad_clip > 0 else float('inf'))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(): # EMAs with train_jit.py's warm-up: decay capped at (1 + step) / (10 + step)
        cap = (1 + iter_num) / (10 + iter_num)
        torch._foreach_lerp_(ema1, params, 1 - min(ema_decay1, cap))
        torch._foreach_lerp_(ema2, params, 1 - min(ema_decay2, cap))
    iter_num += 1
    if iter_num % log_interval == 0:
        now = time.time()
        dt = (now - t_last) / log_interval
        train_seconds += now - t_last
        t_last = now
        lossf = loss.item()
        if not math.isfinite(lossf):
            raise RuntimeError(f'loss is {lossf} at iter {iter_num}')
        print(f"iter {iter_num} (epoch {iter_num / iters_per_epoch:.2f}): loss {lossf:.4f}" + (f", disp {loss_disp.item():.4f}" if disp_captures else "")
              + f", grad norm {gnorm.item():.4f}, time {dt * 1000:.0f}ms, {batch_size / dt:.0f} img/s")
        with open(os.path.join(out_dir, 'train_log.jsonl'), 'a') as f:
            f.write(json.dumps({'iter': iter_num, 'epoch': iter_num / iters_per_epoch, 'loss': lossf, 'imgs_per_sec': batch_size / dt,
                                'train_hours': train_seconds / 3600, **({'disp_loss': loss_disp.item()} if disp_captures else {})}) + '\n')
pool.shutdown()
print(f"done. best FID-{fid_num_images / 1000:g}k {best_fid:.2f}")
