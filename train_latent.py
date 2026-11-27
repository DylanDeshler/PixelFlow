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
Row-7 stack (as train_jit.py, same flags): --attn_res=full --adaln_single=True --attn_res_chunk=8 --kv_prefix_len=32
--attn_gate=head --loss_t_weight=learned --warmup_epochs=1 --grad_clip=0. The KV prefix starts at block 4 (JiT-B/16's
in_context_start) unless --kv_prefix_start is given.
Model size: --depth / --hidden_size / --num_heads (L: 24 / 1024 / 16, H: 32 / 1280 / 16), as JiT's presets.
Pipeline cost for comparing with pixel runs (config.json 'resolved', read by plot_runs_html.py / plot_scaling.py):
  training   train_pflops_offset = the tokenizer's own training compute (vae_cost.py: published training stages with measured
             FLOPs per image, or a train_dito.py checkpoint's measured record) + the one-off dataset encoding
             (cached_encoding_pflops), so training compute = offset + images x 3 x trunk forward. This is what the
             pixels-vs-latents study compares (training compute only)
  sampling   sample_gflops = network calls (the eval's sampler) x trunk forward + VAE decoder forward: recorded for reference,
             not part of the study's comparison
  params     pipeline_params = trunk + VAE encoder + decoder
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
$ python train_latent.py --data_dir=/data/in1k_128 --vae=mar-kl16 --patch_size=1   # MAR's ImageNet-only KL-16 (16 x 8 x 8): 64 tokens
$ python train_latent.py --data_dir=/data/in1k_128 --attn_res=full --adaln_single=True --attn_res_chunk=8 --kv_prefix_len=32 --attn_gate=head \
      --loss_t_weight=learned --warmup_epochs=1 --grad_clip=0                    # the row-7 stack in latent space
"""

import os
import sys
import json
import math
import time
from contextlib import nullcontext, contextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import warnings
import torch
warnings.filterwarnings('ignore', message=r'\s*Online softmax is disabled') # torch.compile's note that it skipped one softmax fusion: harmless, and printed for every compiled graph

from data_jit import MemmapImages, EpochSampler
from jit_common import generate_and_score, sampling_nfe
from fid import InceptionMetrics, load_ref_stats
from model_jit import JiT
from loss_weighting import LossWeighting
from vae_cost import vae_training_cost
from muon import Muon, CombinedOptimizer
from repa import DispCapture, dispersive_loss
import diagnostics as diag_lib
from util.progress import bar

sys.stdout.reconfigure(line_buffering=True) # logs redirected to a file show each line as it's printed

# -----------------------------------------------------------------------------
data_dir = '' # the prepared pixel dataset (FID reference stats, labels, class count); latents in latent_dir
vae = 'stabilityai/sd-vae-ft-ema' # latent_vae.py: or 'mar-kl16' (ImageNet-only, 16 x 8 x 8 at 128px: pass --patch_size=1 for 64 tokens)
latent_dir = '' # default: {data_dir}/latents_{latent_vae.SHORT[vae]} (prepare_latents.py --vae=...)
runs_root = 'runs'
out_dir = '' # default: {runs_root}/{dataset}/latent-b2[_d.._w.._h..]/{recipe}{_changes}
init_from = 'scratch' # 'scratch' or 'resume'
recipe = 'jit' # 'jit' or 'sit' (see above); settings left at their sentinel take the recipe's value
epochs = 20
batch_size = 1024
gradient_accumulation_steps = 1 # the batch is split into this many micro-batches (same effective batch / lr; for large models)
# run length by updates or by compute (the scaling study), as train_jit.py; each overrides epochs
keep_ckpt_at_eval = True # keep ckpt_{iter}.pt (EMA1) and a post-hoc EMA snapshot at every eval
max_updates = 0
# in-loop guided FID and predictions, as train_jit.py (see there): guided FID of the post-hoc EMA guided_eval_std with a small
# CFG search at every eval; pred/fid_at_target and pred/guided_at_target; settle_tol > 0 = a tuning candidate's stopping rule
guided_eval = False
guided_eval_images = 2000
guided_eval_std = 0.10
settle_target_epochs = 100.0
settle_tol = 0.0
settle_patience = 2
settle_min_epochs = 20.0
settle_max_epochs = 50.0
settle_kill = 0.0
converge_tol = 0.0 # > 0: stop once training has converged by the run's own curve (jit_common.converged): in-loop FID improved by
                   # less than this (relative) over the last converge_window of training at converge_patience consecutive evals.
                   # epochs / max_updates / budget_pflops then only cap the run
converge_window = 0.3333
converge_patience = 2
converge_min_epochs = 20.0 # never stop before this many epochs
budget_pflops = 0.0 # trunk training compute: updates x batch x 3 x measured forward FLOPs (tokenizer costs are accounted separately)
budget_keep = '' # milestones as fractions of the run ('1/16,1/8,1/4,1/2,1'): an eval, a kept ckpt_{iter}.pt and a post-hoc EMA snapshot each
train_frac = 1.0 # nested class-stratified training subset (subsets.py), shared with train_jit.py
warmup_updates = -1 # >= 0: linear warm-up over this many updates instead of warmup_epochs
aux_lr_mult = 1.0 # with Muon: the AdamW groups' lr (embeddings incl. the class-prefix table, patch embedding, readout, norms, biases) = lr x this
phema_stds = '' # post-hoc EMA (phema.py): power-function EMAs to track, e.g. '0.05,0.10', snapshotted every epoch and at the milestones
phema_dtype = 'float32'
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
attn_gate = 'none' # as train_jit.py: 'head' / 'elem' gated attention
kv_prefix_len = 0 # as train_jit.py: class + learned K/V-only prefix tokens
kv_prefix_class_tokens = False # as train_jit.py: the prefix is a learned table of kv_prefix_len tokens per class (+ null label)
kv_prefix_start = -1 # -1: block 4 (JiT-B/16's in_context_start, as the pixel runs)
in_context_len = 0 # JiT's in-context class tokens (prepended at in_context_start), as train_jit.py
prediction = 'v' # 'v': the network outputs the velocity (SiT); 'x': the clean latent (JiT's x-prediction, same v-loss, v = (x_hat - z_t) / (1 - t))
t_eps = 0.05 # x-prediction: the (1 - t) clamp, as JiT
dropout = 0.0 # attention + projection dropout in the middle half of the blocks (JiT's placement), the regularized alternative
in_context_start = 4 # the in-context tokens' block, and the KV prefix's default start
loss_t_weight = 'none' # as train_jit.py: 'learned' = EDM2-style learned per-noise-level weight (loss L / e^u(t) + u(t))
loss_weight_normalize = False
loss_weight_lr_mult = 50.0
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
fd_dino = False # also FD-DINOv2 of the eval samples ('val/fd_dino'), as train_jit.py
eval_cfgs = '' # with init_from='resume': score the checkpoint at these CFG scales (comma list, FID-10k each, EMA1 weights), then exit
eval_ckpt = '' # with eval_cfgs: score this kept ckpt_{iter}.pt instead of ckpt.pt
eval_phema = 0.0 # with eval_cfgs: score the post-hoc EMA with this sigma_rel, rebuilt from the snapshots at step eval_at (<= that step only)
eval_at = -1 # the step for eval_phema (-1: the latest snapshot step)
cfg_search = 0 # with --init_from=resume: a fixed budget of guidance scales instead of eval_cfgs (jit_common.cfg_search, as eval_jit.py)
cfg_search_start = 3.0 # the guidance search's first scale (jit_common.cfg_search: log-strength doubling / halving, then bisection)
eval_final_images = 0 # > 0: the best scale re-scored with this many fresh samples (seed + 1), as eval_jit.py
eval_out_dir = '' # write the evaluation to {eval_out_dir}/eval_search.json (eval_jit.py's layout)
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
keep_ckpt_every_epochs = 0 # also keep ckpt_{iter}.pt (EMA1 weights only) for later guided sweeps (--eval_ckpt); 0 = off
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
from latent_vae import load_vae, short_of
latent_dir = latent_dir or os.path.join(data_dir, 'latents_' + short_of(vae))
lmeta = json.load(open(os.path.join(latent_dir, 'meta.json')))
assert lmeta['vae'] == vae, f'{latent_dir} holds {lmeta["vae"]} latents, but vae = {vae}'
C, H, W = lmeta['latent_shape']
dataset = os.path.basename(os.path.normpath(data_dir))
if not out_dir: # pixel runs' naming (jit_common's short tokens); size in the model directory, as JiT's d / w / h
    from jit_common import _name_token
    ARCH = {'depth': ('d', 12), 'hidden_size': ('w', 768), 'num_heads': ('h', 12), 'patch_size': ('p', 2)}
    model_dir = 'latent-b2' + ''.join(f'_{a}{config[k]}' for k, (a, d) in ARCH.items() if config[k] != d) \
                + ('' if vae == _defaults['vae'] else '_' + short_of(vae))
    changed = [_name_token(k, v) for k, v in config.items()
               if k not in tuple(ARCH) + ('vae', 'data_dir', 'latent_dir', 'runs_root', 'out_dir', 'init_from', 'recipe', 'epochs', 'eval_every_epochs',
                            'save_every_epochs', 'log_interval', 'device', 'dtype', 'compile', 'decode', 'attn_res_chunk', 'gen_batch_size',
                            'fid_num_images', 'fid_cfg', 'fd_dino', 'eval_cfgs', 'eval_ckpt', 'eval_phema', 'eval_at', 'cfg_search', 'cfg_search_start', 'eval_final_images', 'eval_out_dir', 'keep_ckpt_every_epochs', 'gradient_accumulation_steps', 'max_updates', 'budget_pflops', 'budget_keep', 'phema_stds', 'phema_dtype', 'keep_ckpt_at_eval', 'converge_tol', 'converge_window', 'converge_patience', 'converge_min_epochs', 'guided_eval', 'guided_eval_images', 'guided_eval_std', 'settle_target_epochs', 'settle_tol', 'settle_patience', 'settle_min_epochs', 'settle_max_epochs', 'settle_kill', 'fid_interval_min', 'fid_interval_max', 'num_sampling_steps', 'probe_blocks',
                            'probe_ts', 'probe_train_images', 'probe_val_images', 'eval_loss_images', 'seed') # (evaluation / system settings)
               and v != dict(_defaults, **RECIPES[recipe])[k]] # settings that differ from the recipe's (or the script's) values
    out_dir = os.path.join(runs_root, dataset, model_dir, '_'.join([recipe] + changed))
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
train_rows = None
if train_frac < 1:
    from subsets import subset_rows
    train_rows = subset_rows(data_dir, pix.labels, train_frac)
    print(f"training subset: {train_frac:g} of the data, {len(train_rows):,} images (nested, class-stratified: {data_dir}/subsets/)")
n_train = len(train_data) if train_rows is None else len(train_rows)
iters_per_epoch = n_train // batch_size
max_iters = epochs * iters_per_epoch
lr = blr * batch_size / 1024
print(f"recipe {recipe}: {optimizer_type}, lr {lr:.2e}, betas ({beta1}, {beta2}), warm-up {warmup_epochs:g} epochs, t {t_sampling}, "
      f"clip {grad_clip:g}; {len(train_data):,} images, {iters_per_epoch} iterations per epoch, {epochs} epochs")

# network: v-prediction (the output is the velocity)
net = JiT(input_size=H, patch_size=patch_size, in_channels=C, hidden_size=hidden_size, depth=depth, num_heads=num_heads,
          num_classes=num_classes, bottleneck_dim=bottleneck_dim, in_context_len=in_context_len, in_context_start=in_context_start, attn_res=attn_res,
          attn_drop=dropout, proj_drop=dropout,
          adaln_single=adaln_single, attn_gate=attn_gate, kv_prefix_len=kv_prefix_len, kv_prefix_start=kv_prefix_start,
          kv_prefix_class_tokens=kv_prefix_class_tokens).to(device)
net.attn_res_chunk = attn_res_chunk
print(f"network: JiT blocks, {depth} x {hidden_size}, {(H // patch_size) * (W // patch_size)} tokens of {patch_size}x{patch_size}x{C} latent patches, "
      f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M params")
loss_weighter = LossWeighting(noise_scale, t_weight='learned', normalize_learned=loss_weight_normalize).to(device) if loss_t_weight == 'learned' else None
assert prediction in ('v', 'x'), f"prediction must be 'v' or 'x', got {prediction}"
def net_v(zt, t, y):
    # the velocity the sampler integrates and the v-loss compares: the output itself (v-prediction) or, for x-prediction,
    # (x_hat - z_t) / (1 - t) with JiT's clamp (t = 1 is clean)
    out = net(zt, t, y)
    if prediction == 'v':
        return out
    return (out.float() - zt.float()) / (1 - t.view(-1, 1, 1, 1)).clamp_min(t_eps)
assert loss_t_weight in ('none', 'learned'), f"loss_t_weight must be 'none' or 'learned', got {loss_t_weight}"

# pipeline cost, recorded in config.json like train_jit.py's (FLOP counter on one sample; FLOPs = 2 x multiply-adds)
from torch.utils.flop_counter import FlopCounterMode
def _flops(fn):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops() / 1e9
_was = net.training
net.eval()
fwd_gflops = _flops(lambda: net(torch.randn(1, C, H, W, device=device), torch.full((1,), 0.5, device=device), torch.zeros(1, dtype=torch.long, device=device)))
net.train(_was)
if budget_pflops > 0:
    max_iters = math.ceil(budget_pflops * 1e6 / (batch_size * 3 * fwd_gflops))
elif max_updates > 0:
    max_iters = max_updates
keep_at = set()
for _f in (v.strip() for v in budget_keep.split(',') if v.strip()):
    _n, _, _d = _f.partition('/')
    keep_at.add(max(1, round(float(_n) / float(_d or 1) * max_iters)))
if budget_pflops > 0 or max_updates > 0 or keep_at:
    print(f"run length: {max_iters:,} updates = {max_iters * batch_size / 1e6:.3f}M images ({max_iters / iters_per_epoch:.2f} epochs of {n_train:,}), "
          f"{max_iters * batch_size * 3 * fwd_gflops / 1e6:,.1f} trunk training PFLOPs" + (f"; milestones at updates {sorted(keep_at)}" if keep_at else ''))
SCALE = lmeta['scale']
vae_model = load_vae(vae, device) if decode == 'vae' else None
img_size_px = lmeta.get('img_size', pix.img_size)
if vae_model is not None:
    _x = torch.randn(1, 3, img_size_px, img_size_px, device=device)
    vae_cost = dict(vae_encoder_params=sum(p.numel() for n, p in vae_model.named_parameters() if n.startswith(('encoder', 'quant_conv'))),
                    vae_decoder_params=sum(p.numel() for n, p in vae_model.named_parameters() if n.startswith(('decoder', 'post_quant_conv', 'renderer'))),
                    vae_encode_gflops=_flops(lambda: vae_model.encode_mean(_x)),
                    vae_decode_gflops=_flops(lambda: vae_model.decode_img(torch.randn(1, C, H, W, device=device))))
else:
    vae_cost = dict(vae_encoder_params=0, vae_decoder_params=0, vae_encode_gflops=0.0, vae_decode_gflops=0.0)
n_encoded = n_train * 2 # the training images this run uses, both flips (prepare_latents.py encodes both); held-out encodings are evaluation, not training
trunk_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
vae_train = vae_training_cost(vae_model, vae, device) if vae_model is not None else None
assert vae_train is not None or vae_model is None, f"no training record for {lmeta['vae']} in vae_cost.VAE_TRAINING: add its stages (model card / paper)"
vae_train = vae_train or {'vae_train_pflops': 0.0, 'vae_train_complete': True, 'vae_train_stages': []}
if not vae_train['vae_train_complete']:
    print(f"note: {vae}'s training steps are unpublished; its training compute is not in train_pflops_offset (per image: "
          + ', '.join(f"{s['gflops_per_image']:.0f} GF" for s in vae_train['vae_train_stages'] if s['pflops'] is None) + ')')
sample_nfe = sampling_nfe(num_sampling_steps, 'heun', fid_cfg, (fid_interval_min, fid_interval_max))
config['resolved'] = dict(model_name=f'latent-{depth}x{hidden_size}/{patch_size}', img_size=img_size_px, num_classes=num_classes,
                          effective_batch_size=batch_size, iters_per_epoch=iters_per_epoch, max_iters=max_iters, keep_at=sorted(keep_at), train_images=n_train,
                          lr=lr, params=trunk_params, fwd_gflops=fwd_gflops, **vae_cost,
                          pipeline_params=trunk_params + vae_cost['vae_encoder_params'] + vae_cost['vae_decoder_params'],
                          train_pflops_offset=vae_train['vae_train_pflops'] + n_encoded * vae_cost['vae_encode_gflops'] / 1e6, **vae_train, # tokenizer training + the one-off dataset encoding
                          cached_encoding_pflops=n_encoded * vae_cost['vae_encode_gflops'] / 1e6,
                          sample_nfe=sample_nfe, sample_gflops=sample_nfe * fwd_gflops + vae_cost['vae_decode_gflops'],
                          latent_shape=[C, H, W], vae=lmeta['vae'], latent_dir=latent_dir)
with open(os.path.join(out_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print(f"cost: trunk {trunk_params / 1e6:.2f}M params, {fwd_gflops:.2f} forward GFLOPs per image; VAE encoder "
      f"{vae_cost['vae_encoder_params'] / 1e6:.1f}M / {vae_cost['vae_encode_gflops']:.1f} GFLOPs, decoder {vae_cost['vae_decoder_params'] / 1e6:.1f}M / "
      f"{vae_cost['vae_decode_gflops']:.1f} GFLOPs; sampling {sample_nfe} calls + VAE decoder = {config['resolved']['sample_gflops']:.0f} GFLOPs per image")
_trunk_pf = max_iters * batch_size * 3 * fwd_gflops / 1e6
for st in vae_train['vae_train_stages']:
    steps = 'unknown' if st['steps'] is None else f"{st['steps']:,}"
    total = 'not counted' if st['pflops'] is None else f"{st['pflops']:,.0f} PFLOPs"
    print(f"  VAE training: {st['name']}: {steps} steps x {st['batch']} at {st['res']}px, {st['trains']}, "
          f"{st['gflops_per_image']:.0f} GFLOPs/image = {total} ({st['note']})")
print(f"  training compute: VAE {vae_train['vae_train_pflops']:,.0f}{'' if vae_train['vae_train_complete'] else ' (incomplete)'} + dataset encoding "
      f"{config['resolved']['cached_encoding_pflops']:,.0f} + trunk {_trunk_pf:,.0f} PFLOPs over this run (sampling / decoding: not counted)")

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
        6 if (name.endswith('adaLN_modulation.1.weight') or name.endswith('t_block.1.weight')) else \
        2 if (name.endswith('attn_res_modulation.1.weight') or name.endswith('.kv.weight')) else 1 # (as train_jit.py)
muon_named = {id(p): n for n, p in net.blocks.named_parameters() if optimizer_type == 'muon' and p.ndim >= 2 and 'scale_shift_table' not in n}
if optimizer_type == 'muon' and adaln_single:
    muon_named[id(net.t_block[1].weight)] = 't_block.1.weight'
muon_params, decay, no_decay = {}, [], []
for n, p in net.named_parameters():
    if id(p) in muon_named:
        muon_params.setdefault(muon_split(muon_named[id(p)]), []).append(p)
    else:
        (no_decay if p.ndim == 1 or n.endswith('.bias') or 'scale_shift_table' in n else decay).append(p)
weighter_params = list(loss_weighter.parameters()) if loss_weighter is not None else [] # own group, as train_jit.py
aux_lr = lr * (aux_lr_mult if optimizer_type == 'muon' else 1.0)
adamw = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0.0, 'peak_lr': aux_lr, 'lr': aux_lr}, {'params': decay, 'weight_decay': weight_decay, 'peak_lr': aux_lr, 'lr': aux_lr}]
                          + ([{'params': weighter_params, 'weight_decay': 0.0, 'peak_lr': lr * loss_weight_lr_mult, 'lr': lr * loss_weight_lr_mult}] if weighter_params else []),
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
phema_state = None
if eval_ckpt: # a kept milestone (EMA1 weights only): evaluation with eval_cfgs
    assert (eval_cfgs or cfg_search) and init_from == 'resume', 'eval_ckpt is for guided sweeps: pass --eval_cfgs or --cfg_search and --init_from=resume'
    ck = torch.load(eval_ckpt, map_location=device, weights_only=False)
    ema1 = [ck['ema1'][n].to(device) for n, _ in net.named_parameters()]
    iter_num = ck['iter_num']
    print(f"loaded {eval_ckpt} (iter {iter_num}, epoch {iter_num / iters_per_epoch:.2f})")
    del ck
elif eval_phema: # a post-hoc EMA rebuilt from snapshots up to step eval_at: evaluation with eval_cfgs / cfg_search
    assert (eval_cfgs or cfg_search) and init_from == 'resume', 'eval_phema is for guided sweeps: pass --eval_cfgs or --cfg_search and --init_from=resume'
    from phema import reconstruct
    _snap_dir = os.path.join(out_dir, 'phema')
    _steps = sorted({int(f.split('-')[1]) for f in os.listdir(_snap_dir) if f.startswith('phema-')})
    iter_num = _steps[-1] if eval_at < 0 else eval_at
    _w = reconstruct(_snap_dir, iter_num, eval_phema)
    ema1 = [_w[n].to(device) for n, _ in net.named_parameters()]
    print(f"post-hoc EMA sigma_rel {eval_phema} at step {iter_num} (epoch {iter_num / iters_per_epoch:.2f}), from {_snap_dir}")
elif init_from == 'resume':
    ck = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
    net.load_state_dict(ck['model'])
    ema1 = [ck['ema1'][n].to(device) for n, _ in net.named_parameters()]
    ema2 = [ck['ema2'][n].to(device) for n, _ in net.named_parameters()]
    optimizer.load_state_dict(ck['optimizer'])
    if loss_weighter is not None and ck.get('loss_weighting') is not None:
        loss_weighter.load_state_dict(ck['loss_weighting'])
    iter_num, best_fid, train_seconds = ck['iter_num'], ck['best_fid'], ck.get('train_seconds', 0.0)
    phema_state = ck.get('phema')
    print(f"resumed at iter {iter_num} (epoch {iter_num / iters_per_epoch:.2f})")
    del ck
phema = None # post-hoc EMA: power-function EMAs of every parameter, snapshotted for reconstruction (phema.reconstruct)
if phema_stds and not (eval_cfgs or cfg_search):
    from phema import PowerEMA
    phema = PowerEMA(net.parameters(), [float(v) for v in phema_stds.split(',') if v])
    if init_from == 'resume' and iter_num > 0:
        assert phema_state is not None, f'--phema_stds on a resumed run without power EMAs in its checkpoint (iter {iter_num}): they must start at step 1'
        phema.load_state(phema_state, device)
if compile:
    print("compiling the model... (the first iteration and the first eval take a few minutes)")
    net.compile()

def sample_t(n):
    if t_sampling == 'uniform':
        return torch.rand(n, device=device)
    return torch.sigmoid(torch.randn(n, device=device) * P_std + P_mean)

def get_lr(it):
    if warmup_updates >= 0:
        return lr * min(1.0, it / warmup_updates) if warmup_updates > 0 else lr
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
if decode == 'vae':
    def to_images(z):
        with ctx:
            return vae_model.decode_img(z / SCALE).float().clamp(-1, 1)
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
            return net_v(z, tt, labels).float()
        both = net_v(torch.cat([z, z]), torch.cat([tt, tt]), torch.cat([labels, torch.full_like(labels, num_classes)])).float()
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
if fd_dino:
    from dino_fd import DinoFD
    dino = DinoFD(device)
    dino_ref = dino.reference(pix, n_per_class=10, cache=os.path.join(data_dir, 'fd_dino_ref_10pc.npz')) # the same reference as train_jit.py
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
            v_hat = net_v(tt * x + (1 - tt) * e, t, y).float()
        tot += (x - e - v_hat).pow(2).mean().item() * len(x)
    net.train()
    return tot / len(z_np)

import predict_fid
FULL_TRAIN = predict_fid.FULL_TRAIN
hist_fid, hist_gfid = [], []
_pred_floor = None
def pred_floor():
    # the guided FID's floor at the in-loop sample count (predict_fid.floor_of: real held-out images for pixels, the tokenizer's
    # reconstructions for latents); predictions are F = floor + A x^-a
    global _pred_floor
    if _pred_floor is None:
        _pred_floor = predict_fid.floor_of(out_dir, int(round(guided_eval_images / 1000)))
        print(f"prediction floor (guided FID-{guided_eval_images / 1000:g}k): {_pred_floor:.2f}")
    return _pred_floor
guided_cfg_prev = None
last_log = {}

@torch.no_grad()
def guided_eval_log(step):
    # guided FID of the post-hoc EMA guided_eval_std with a small CFG search (as train_jit.py)
    global guided_cfg_prev
    k = min(range(len(phema.stds)), key=lambda i: abs(phema.stds[i] - guided_eval_std))
    params = list(net.parameters())
    backup = [p.detach().clone() for p in params]
    for p, e in zip(params, phema.emas[k]):
        p.copy_(e)
    res = {}
    def score(c):
        res[c] = generate_and_score(sampler_model, metrics, ref_stats, guided_eval_images, num_classes, device, ctx, cfg=c,
                                    interval=(fid_interval_min, fid_interval_max), steps=num_sampling_steps, method='heun',
                                    batch_size=gen_batch_size, desc=f'guided FID (cfg {c:g})')['fid']
        return res[c]
    try:
        if guided_cfg_prev is None:
            from jit_common import cfg_search as _cs
            _cs(score, 6, 3.0)
        else:
            from jit_common import cfg_track
            cfg_track(score, guided_cfg_prev) # the previous best's strength x 0.8 / 1 / 1.25, extended while the best is at an end
    finally:
        for p, b in zip(params, backup):
            p.copy_(b)
    best = min(res, key=res.get)
    guided_cfg_prev = best
    print(f"guided FID-{guided_eval_images / 1000:g}k (post-hoc EMA {phema.stds[k]:g}): " + ', '.join(f'cfg {c:g} {v:.2f}' for c, v in sorted(res.items()))
          + f" -> {res[best]:.2f} at cfg {best:g}")
    return {'val/guided_fid': res[best], 'val/guided_cfg': best}

def evaluate(step):
    global last_log
    t0 = time.time()
    log = {'eval/val_loss': val_loss()}
    with ema_weights(1):
        dino_feats = []
        out = generate_and_score(sampler_model, metrics, ref_stats, fid_num_images, num_classes, device, ctx, cfg=fid_cfg,
                                 interval=(fid_interval_min, fid_interval_max), steps=num_sampling_steps, method='heun', batch_size=gen_batch_size,
                                 on_batch=(lambda x, labels=None: dino_feats.append(dino.features(x))) if fd_dino else None)
        net.eval()
        tr, va = probe_features(probe_train[0], 'probe features, train'), probe_features(probe_val[0], 'probe features, held-out')
        net.train()
    probes = diag_lib.run_probes(tr, torch.from_numpy(probe_train[1]).to(device), va, torch.from_numpy(probe_val[1]).to(device), num_classes)
    log.update({'val/fid': out['fid'], 'val/is': out['is']})
    if guided_eval and phema is not None:
        log.update(guided_eval_log(step))
    hist_fid.append((step * batch_size / FULL_TRAIN, out['fid']))
    if 'val/guided_fid' in log:
        hist_gfid.append((step * batch_size / FULL_TRAIN, log['val/guided_fid']))
    for key, h, fl in (('pred/fid_at_target', hist_fid, 0.0), ('pred/guided_at_target', hist_gfid, pred_floor())):
        v = predict_fid.predict(h, settle_target_epochs, fl)
        if v is not None:
            log[key] = v
    last_log = log
    if fd_dino:
        log['val/fd_dino'] = dino.distance(torch.cat(dino_feats), dino_ref)
    if loss_weighter is not None:
        log.update({f'loss_weight/{k}': v for k, v in loss_weighter.summary().items()})
    for (b, t), r in probes.items():
        log[f'probe/knn_block{b:02d}_t{t:.2f}'] = r['knn']
        log[f'probe/linear_block{b:02d}_t{t:.2f}'] = r['linear']
    with open(os.path.join(out_dir, 'diagnostics', f'diag_{step}.json'), 'w') as f:
        json.dump({'iter': step, 'scalars': log, 'probes': {f'block{b}_t{t}': r for (b, t), r in probes.items()}}, f)
    bmid = pb[len(pb) // 2]
    print(f"iter {step} (epoch {step / iters_per_epoch:.2f}): val loss {log['eval/val_loss']:.4f}, FID-{fid_num_images / 1000:g}k {out['fid']:.2f}, "
          f"IS {out['is']:.1f} (cfg {fid_cfg})" + (f", FD-DINOv2 {log['val/fd_dino']:.1f}" if fd_dino else '') + f"; probes block {bmid} t=0.5: kNN {probes[(bmid, 0.5)]['knn']:.2f}, linear {probes[(bmid, 0.5)]['linear']:.2f} "
          f"(eval {(time.time() - t0) / 60:.1f} min)")
    return out['fid']

def save_phema_snapshots():
    # one file per power EMA at this step (the format phema.reconstruct reads)
    d = os.path.join(out_dir, 'phema')
    os.makedirs(d, exist_ok=True)
    store = torch.bfloat16 if phema_dtype == 'bfloat16' else torch.float32
    names = [n for n, _ in net.named_parameters()]
    base = net.state_dict()
    for std, ema in zip(phema.stds, phema.emas):
        sd = dict(base)
        sd.update({n: e.detach().to('cpu', store) for n, e in zip(names, ema)})
        path = os.path.join(d, f'phema-{iter_num:08d}-{std:.3f}.pt')
        torch.save({'t': iter_num, 'std': std, 'model': sd, 'params': names}, path + '.tmp')
        os.replace(path + '.tmp', path)

def save_checkpoint():
    names = [n for n, _ in net.named_parameters()]
    torch.save({'model': net.state_dict(), 'ema1': dict(zip(names, [e.cpu() for e in ema1])), 'ema2': dict(zip(names, [e.cpu() for e in ema2])),
                'optimizer': optimizer.state_dict(), 'loss_weighting': loss_weighter.state_dict() if loss_weighter is not None else None, 'iter_num': iter_num, 'best_fid': best_fid, 'train_seconds': train_seconds,
                'phema': phema.state() if phema is not None else None, 'config': config}, os.path.join(out_dir, 'ckpt.pt.tmp'))
    os.replace(os.path.join(out_dir, 'ckpt.pt.tmp'), os.path.join(out_dir, 'ckpt.pt'))
    if (keep_ckpt_every_epochs and iter_num % (keep_ckpt_every_epochs * iters_per_epoch) == 0) or iter_num in keep_at \
            or (keep_ckpt_at_eval and (iter_num % eval_iters == 0 or iter_num == max_iters)): # EMA1 only, for --eval_ckpt
        torch.save({'ema1': dict(zip(names, [e.cpu() for e in ema1])), 'iter_num': iter_num, 'config': config},
                   os.path.join(out_dir, f'ckpt_{iter_num}.pt'))

if eval_cfgs or cfg_search: # evaluation only: a guided sweep / search of the resumed checkpoint, as eval_jit.py does for pixel runs
    assert init_from == 'resume', 'eval_cfgs evaluates a checkpoint: pass --init_from=resume'
    res = {'iter': iter_num, 'epoch': iter_num / iters_per_epoch, 'interval': [fid_interval_min, fid_interval_max], 'fid': {}, 'is': {},
           'weights': f'phema {eval_phema}' if eval_phema else (eval_ckpt or 'ema1')}
    def score(cg):
        o = generate_and_score(sampler_model, metrics, ref_stats, fid_num_images, num_classes, device, ctx, cfg=cg, seed=seed,
                               interval=(fid_interval_min, fid_interval_max), steps=num_sampling_steps, method='heun', batch_size=gen_batch_size)
        res['fid'][str(cg)], res['is'][str(cg)] = o['fid'], o['is']
        print(f"cfg {cg:g}: FID-{fid_num_images / 1000:g}k {o['fid']:.3f}, IS {o['is']:.1f}")
        return o['fid']
    with ema_weights(1):
        if cfg_search:
            from jit_common import cfg_search as cfg_search_fn
            cfg_search_fn(score, cfg_search, cfg_search_start)
            res['cfg_search'] = {'budget': cfg_search, 'start': cfg_search_start}
        else:
            for cg in (float(v) for v in eval_cfgs.split(',') if v.strip()):
                score(cg)
    cfgs = [float(c) for c in res['fid']]
    best_c = float(min(res['fid'], key=res['fid'].get))
    res['best_fid_cfg'] = best_c
    if eval_final_images > 0: # the reported number: samples the search never saw
        with ema_weights(1):
            o = generate_and_score(sampler_model, metrics, ref_stats, eval_final_images, num_classes, device, ctx, cfg=best_c, seed=seed + 1,
                                   interval=(fid_interval_min, fid_interval_max), steps=num_sampling_steps, method='heun', batch_size=gen_batch_size)
        res['final'] = {'num_images': eval_final_images, 'seed': seed + 1, str(best_c): {'fid': o['fid'], 'is': o['is']}}
        print(f"re-scored: cfg {best_c:g}: FID-{eval_final_images / 1000:g}k {o['fid']:.3f}, IS {o['is']:.1f} (fresh samples)")
    best = min(res['fid'], key=res['fid'].get)
    print(f"iter {iter_num} (epoch {res['epoch']:.1f}): best FID {res['fid'][best]:.3f} at cfg {best}")
    if eval_out_dir:
        os.makedirs(eval_out_dir, exist_ok=True)
        with open(os.path.join(eval_out_dir, 'eval_search.json'), 'w') as f:
            json.dump(res, f, indent=1)
    path = os.path.join(out_dir, f"eval_cfgs_{iter_num}" + (f'_phema{eval_phema:g}' if eval_phema else '')
                        + (f'_search{cfg_search}' if cfg_search else f"_cfg{'_'.join(f'{c:g}' for c in cfgs)}") + '.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'wrote {path}')
    sys.exit(0)

# training
sampler = EpochSampler(len(train_data), batch_size, seed=seed, indices=train_rows)
sampler.epoch = iter_num // iters_per_epoch
rng = np.random.default_rng(seed + iter_num)
pool = ThreadPoolExecutor(1) # reads the next batch while the GPU trains on this one
fetch = lambda: train_data.get(sampler.next(), flip=True, rng=rng)
pending = pool.submit(fetch)
eval_iters, save_iters = eval_every_epochs * iters_per_epoch, save_every_epochs * iters_per_epoch
from jit_common import fid_history, converged
fid_hist = [h for h in fid_history(out_dir) if h[0] <= iter_num] if converge_tol > 0 else [] # the evals so far (resume)
if iter_num > 0 and os.path.isdir(os.path.join(out_dir, 'diagnostics')): # the prediction histories and the last guided scale
    _cut = iter_num * batch_size / FULL_TRAIN + 1e-9
    hist_fid[:] = [h for h in predict_fid.history(out_dir, 'val/fid') if h[0] <= _cut]
    hist_gfid[:] = [h for h in predict_fid.history(out_dir, 'val/guided_fid') if h[0] <= _cut]
    _cfgs = [h for h in predict_fid.history(out_dir, 'val/guided_cfg') if h[0] <= _cut]
    guided_cfg_prev = _cfgs[-1][1] if _cfgs else None
settle_preds = [predict_fid.predict(hist_gfid[:i], settle_target_epochs, pred_floor()) for i in range(1, len(hist_gfid) + 1)]
params = list(net.parameters())
t_last = time.time()
while True:
    if iter_num > 0 and (iter_num % eval_iters == 0 or iter_num == max_iters or iter_num in keep_at):
        _fid = evaluate(iter_num)
        best_fid = min(best_fid, _fid)
        if settle_tol > 0:
            settle_preds.append(last_log.get('pred/guided_at_target'))
            _ep = iter_num * batch_size / FULL_TRAIN
            _p = settle_preds[-1]
            _why = None
            if _ep >= settle_max_epochs - 1e-6:
                _why = f'reached {settle_max_epochs:g} epochs'
            elif _ep >= settle_min_epochs - 1e-6 and predict_fid.settled(settle_preds, settle_tol, settle_patience):
                _why = f'guided prediction settled ({_p:.2f})'
            elif _ep >= settle_min_epochs - 1e-6 and settle_kill > 0 and _p is not None and _p > settle_kill:
                _why = f'guided prediction {_p:.2f} above the search best x margin ({settle_kill:.2f})'
            print(f"settle check: predicted guided FID at epoch {settle_target_epochs:g}: {_p}" + (f" -> stopping: {_why}" if _why else ""))
            if _why: # (the candidate's end, whether by the rule or at its cap: the search reads settled.json)
                max_iters = min(max_iters, iter_num)
                with open(os.path.join(out_dir, 'settled.json'), 'w') as _f:
                    json.dump({'iter': iter_num, 'epochs': _ep, 'reason': _why, 'pred_guided': _p, 'pred_fid': last_log.get('pred/fid_at_target'),
                               'preds': settle_preds}, _f, indent=1)
        if converge_tol > 0:
            fid_hist.append((iter_num, float(_fid)))
            _conv, _detail = converged(fid_hist, converge_tol, converge_window, converge_patience, converge_min_epochs * iters_per_epoch)
            if _detail:
                print(f"convergence check: {_detail}" + (" -> converged, stopping" if _conv else ""))
            if _conv and iter_num < max_iters:
                max_iters = iter_num # this step's checkpoint, snapshots and the loop's exit follow as for the planned end
                with open(os.path.join(out_dir, 'converged.json'), 'w') as _f:
                    json.dump({'iter': iter_num, 'epoch': iter_num / iters_per_epoch, 'detail': _detail, 'tol': converge_tol,
                               'window': converge_window, 'patience': converge_patience}, _f, indent=1)
        t_last = time.time()
    if phema is not None and iter_num > 0 and (iter_num % iters_per_epoch == 0 or iter_num == max_iters or iter_num in keep_at or iter_num % eval_iters == 0):
        save_phema_snapshots()
    if iter_num > 0 and (iter_num % save_iters == 0 or iter_num == max_iters or iter_num in keep_at or iter_num % eval_iters == 0):
        save_checkpoint()
    if iter_num >= max_iters:
        break
    for g in optimizer.param_groups:
        g['lr'] = g['peak_lr'] * get_lr(iter_num) / lr
    x_np, y_np = pending.result()
    pending = pool.submit(fetch)
    x_all = torch.from_numpy(x_np).to(device, non_blocking=True)
    y_all = torch.from_numpy(y_np).to(device, non_blocking=True)
    for x, y in zip(x_all.chunk(gradient_accumulation_steps), y_all.chunk(gradient_accumulation_steps)): # micro-batches (1 = the whole batch)
        y = torch.where(torch.rand(len(y), device=device) < label_drop_prob, torch.full_like(y, num_classes), y)
        t = sample_t(len(x)).view(-1, 1, 1, 1)
        e = torch.randn_like(x) * noise_scale
        with ctx:
            v_hat = net_v(t * x + (1 - t) * e, t.flatten(), y)
            if loss_weighter is not None: # learned per-level weight on the per-sample v-loss (train_jit.py's residual is the same v - v_hat)
                loss = loss_weighter(x - e - v_hat.float(), t.flatten())
            else:
                loss = (x - e - v_hat.float()).pow(2).mean() # ||v - v_hat||^2, v = x - s e
            loss_disp = (sum(dispersive_loss(c.out, disp_temp, disp_kind, pool=disp_pool, group=disp_group) for c in disp_captures)
                         / len(disp_captures)) if disp_captures else None
            total = loss + disp_weight * loss_disp if disp_captures else loss
        (total / gradient_accumulation_steps).backward()
    gnorm = torch.nn.utils.clip_grad_norm_(params, grad_clip if grad_clip > 0 else float('inf'))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(): # EMAs with train_jit.py's warm-up: decay capped at (1 + step) / (10 + step)
        cap = (1 + iter_num) / (10 + iter_num)
        torch._foreach_lerp_(ema1, params, 1 - min(ema_decay1, cap))
        torch._foreach_lerp_(ema2, params, 1 - min(ema_decay2, cap))
    if phema is not None:
        phema.update(params, iter_num + 1) # t = optimizer steps taken, counting this one
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
