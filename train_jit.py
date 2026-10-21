"""
JiT (Li & He 2025) pixel-space flow matching on ImageNet, single GPU, with optional REPA alignment.

Model and denoiser are JiT's own code (model_jit.py, denoiser.py); this script replaces their
DDP training loop. Defaults follow JiT-B/16: AdamW (0.9, 0.95), no weight decay, base lr 5e-5
scaled to the effective batch (2e-4 at 1024), constant lr after 5 warmup epochs, EMA 0.9999 / 0.9996,
logit-normal t (-0.8, 0.8), x-prediction with v-loss.

Datasets come from prepare_jit_data.py into {data_root}/{dataset}. Settings come from config files and/or --key=value,
and the run directory is derived from them: {runs_root}/{dataset}/{model}/{changes}, e.g.
$ python train_jit.py config/jit_in100_128.py                                  -> runs/in100_128/jit-b16/base
$ python train_jit.py config/jit_in100_128.py --repa_encoder=dino_vitb16       -> runs/in100_128/jit-b16/repa-dino_vitb16
$ python train_jit.py config/jit_in100_128.py config/model_jit_s16.py          -> runs/in100_128/jit-b16_w384_h6/base
$ python train_jit.py config/jit_in100_128.py --repa_encoder=dino_vitb16 --init_from=resume   # same command + resume after preemption

FID (no CFG, fixed seed) is tracked during training; final CFG-swept numbers come from eval_jit.py.
With diag=True every FID eval also runs the data-efficiency diagnostics in diagnostics.py (needs the val split
and fid.py's feature files) and writes them to wandb and {out_dir}/diagnostics/.
"""

import os
import re
import time
import math
import uuid
import json
from contextlib import nullcontext
from torchinfo import summary

import numpy as np
import torch

from data_jit import MemmapImages, EpochSampler, to_model_input
from jit_common import build_denoiser, resolve_arch, describe_arch, run_dir, legacy_run_dir, ema_state_dicts, ema_weights, generate_and_score, save_labeled_grid
from fid import InceptionMetrics, load_ref_stats, precision_recall
from repa import FrozenEncoder, Projector, MaskedTransformerAdapter, BlockCapture, repa_loss, mta_loss
from masking import TokenMasker, flow_loss
from muon import Muon, CombinedOptimizer
import diagnostics as diag_lib
import plot_trends
from progress_gif import make_progress_gifs

# -----------------------------------------------------------------------------
# I/O
runs_root = 'runs'
out_dir = '' # '' -> {runs_root}/{dataset}/{model}/{settings changed from these defaults}, see jit_common.run_dir
run_name = '' # optional tag appended to the derived out_dir (e.g. seed2)
eval_every_epochs = 10 # FID eval
save_every_epochs = 1 # overwrite ckpt.pt
keep_ckpt_every_epochs = 0 # also keep ckpt_{iter}.pt (EMA1 weights only, ~0.5 GB for JiT-B) for eval_jit.py / eval_linear.py; 0 = off
log_interval = 100
eval_only = False # if True, script exits right after one FID eval (use with init_from='resume')
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = True
wandb_project = 'pixelflow'
wandb_run_name = '' # defaults to the run's path under runs_root
# data
data_root = os.environ.get('JIT_DATA_ROOT', '/data') # where prepare_jit_data.py wrote the datasets (or set $JIT_DATA_ROOT)
dataset = 'in100_128' # in100_128, in100_256, in1k_128, in1k_256, ... (a directory under data_root)
data_dir = '' # '' -> {data_root}/{dataset}; set it to use a dataset outside data_root
gradient_accumulation_steps = 1
batch_size = 1024 # micro batch; effective batch = batch_size * gradient_accumulation_steps (JiT uses 1024)
# model: a JiT preset, optionally with any architecture field overridden (-1 keeps the preset's value)
model = 'JiT-B/16' # JiT-B/16, JiT-B/32, JiT-L/16, JiT-L/32, JiT-H/16, JiT-H/32
depth = -1
hidden_size = -1
num_heads = -1
patch_size = -1
bottleneck_dim = -1 # width of the low-rank patch embedding
in_context_len = -1 # number of in-context class tokens (0 disables them)
in_context_start = -1 # block where the in-context tokens are prepended
mlp_ratio = -1.0
attn_res = 'none' # 'full' | 'block': Attention Residuals (arXiv 2603.15031), softmax mixing over depth instead of the plain residual sum
attn_res_block_size = 0 # block only: JiT blocks per AttnRes block; 0 -> about 8 blocks (the paper's default)
dcae_stem = False # DC-AE style residual conv branches before the patch embedding (full resolution and the 4px level), zero-init gates
dcae_head = False # DC-AE style residual conv branches on the predicted image (token level and the 4px level), zero-init gates
dcae_width = 128 # hidden channels of the DC-AE branches (the full-resolution stem branch uses width / 4)
dcae_kernel2 = 1 # kernel of each DC-AE branch's second conv (the first is 3x3)
layernorm = False # every RMSNorm in the network (block pre-norms, QK-norm, final layer, AttnRes keys, DC-AE branches) becomes a LayerNorm
patch_overlap = 0 # pixels each patch overlaps its neighbours on every side, in the patch embedding and the output (overlap-add)
skip_pathway = False # full-rank input patches merged into the trunk's output before the final layer, behind a gate from the conditioning (zero-init)
skip_head_blocks = 0 # skip_pathway only: JiT blocks after the merge, so input detail can be combined with trunk features
skip_residual = False # with skip_final_layer: the with-skip prediction = trunk prediction (detached) + a correction from the skip branch's final layer
skip_final_layer = False # skip_pathway + skip_stop_grad: the with-skip prediction gets its own final layer (trunk trained exactly as vanilla when injected after the trunk)
skip_stop_grad = False # skip_pathway only: also train the trunk-only prediction; the skip branch's loss stops at the injection point (trunk trained as vanilla JiT)
skip_pathway_drop_prob = 0.0 # skip_pathway only: in training, drop the skip for this fraction of samples (the trunk then reconstructs them alone)
skip_inject_block = 0 # skip_pathway only: 0 -> after the trunk; -k -> before the last k blocks; k -> after block k
gated_bottleneck = False # add a full-rank patch embedding behind a gate from the conditioning (zero-init: starts as the bottleneck)
attn_res_adaln = False # adaLN on the AttnRes queries from the conditioning (time + class), one shared projection (~1.2M params on JiT-B)
adaln_single = False # adaLN-single (PixArt-alpha): one modulation projection shared by all blocks + a learned table per block (~40M fewer params on JiT-B)
attn_dropout = 0.0
proj_dropout = 0.0
# flow matching
P_mean = -0.8
P_std = 0.8
t_eps = 5e-2
noise_scale = 0.0 # 0 -> img_size / 256, JiT's resolution rule (1.0 at 256, 2.0 at 512)
label_drop_prob = 0.1
# optimizer
epochs = 200
warmup_epochs = 5
blr = 2e-4 # base lr per 1024 images
lr_schedule = 'constant' # 'constant' or 'cosine'
min_lr = 0.0
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
optimizer_type = 'adamw' # 'adamw' (JiT) or 'muon': Muon for the 2D weight matrices inside the transformer blocks, AdamW for everything else
muon_adjust_lr = 'match_rms_adamw' # 'match_rms_adamw' (Moonshot: update RMS matched to AdamW, Muon uses the AdamW lr and weight_decay)
                                   # or 'original' (Keller Jordan: muon_lr / muon_weight_decay below)
muon_lr = 0.02 # 'original' only (reference default)
muon_weight_decay = 0.0 # 'original' only (reference default)
muon_momentum = 0.95 # reference default, Nesterov
muon_ns_steps = 5 # Newton-Schulz iterations, reference default
muon_split_fused = True # orthogonalize fused weights per component: qkv -> q, k, v; SwiGLU w12 -> w1, w2; adaLN -> its 6 projections
muon_adaln = True # adaLN modulation projections on Muon (2D hidden weights) or, if False, on AdamW (keeps their zero init gentle)
grad_clip = 1.0
ema_decay1 = 0.9999
ema_decay2 = 0.9996
ema_warmup = True # decay = min(decay, (1+step)/(10+step)); False is JiT's update (full decay from step 0)
# representation alignment ('' disables). see repa.py for the encoders and the two adapters
repa_encoder = '' # e.g. timm:vit_base_patch14_dinov2.lvd142m (PixelREPA's DINOv2 setup), dinov2_vitb14, dino_vitb16
repa_adapter = 'mlp' # 'mlp' (REPA's MLP projector; hurts JiT per Shin et al. 2026) or 'mta' (PixelREPA's Masked Transformer Adapter)
repa_depth = 4 # align the output of this block (1-indexed); JiT-B's in_context_start, the block PixelREPA uses
repa_coeff = -1.0 # alignment loss weight; -1 -> the method's published value: 0.5 for mlp (REPA), 0.1 for mta (PixelREPA)
repa_proj_dim = 2048 # mlp only: projector width (REPA)
repa_mask_ratio = 0.2 # mta only: fraction of adapter-input tokens masked (PixelREPA)
repa_adapter_depth = 2 # mta only: transformer blocks in the adapter (PixelREPA)
# MAE-style token masking (masking.py): same loss, masked patches predicted from context. 0 disables
mask_ratio = 0.0 # fraction of patches replaced by a learned mask token in a masked sample
mask_prob = 0.0 # fraction of each batch that is masked (the rest trains unmasked, as at inference)
mask_weighting = 'effective_t' # 'effective_t' (masked patches weighted as t=0) or 'jit' (sampled-t weight for all patches)
# FID eval during training
fid_num_images = 10000 # must be divisible by the number of classes
fid_cfg = 1.0
fid_interval_min = 0.1
fid_interval_max = 1.0
sample_cfg = 3.0 # also save a sample grid (one batch, no FID) at this CFG scale; 0 disables
sample_interval_min = 0.1 # CFG interval for that grid: guidance applied only for t in [min, max]
sample_interval_max = 1.0
sampling_method = 'heun'
num_sampling_steps = 50
time_schedule = 'uniform' # sampling time grid: 'uniform' (JiT), 'shift' (FLUX/SD3-style, uses P_mean), 'train_quantile'
gen_batch_size = 256
eval_loss_images = 4096 # fixed train / held-out images for the eval train / val loss (training objective, same noise every eval)
# diagnostics, run at every FID eval (see diagnostics.py)
diag = True
diag_raw = False # 3: also FID / precision / recall of the raw (non-EMA) weights (a second full FID run)
diag_images = 1024 # 1/4/6: fixed train and held-out images for the per-noise-level pass
diag_t_bins = 10
diag_freq_bands = 8
diag_probe_blocks = '' # comma separated, 1-indexed; '' -> a third, two thirds and the last block of the model
diag_probe_ts = '1.0,0.75,0.5,0.25,0.1' # noise levels for the probes (t=1 is clean)
diag_probe_train_images = 10000
diag_probe_val_images = 5000
diag_spectrum_images = 2000 # generated samples for the sample-vs-real spectrum
diag_window_tiles = '1,2,4' # windowed-attention ablation: attention confined to k x k patch tiles ('' disables; k >= grid skipped)
progress_gif = True # after each eval, rebuild samples/progress*.gif from all saved sample grids (same seed / classes) and log to wandb
trend_plots = True # after each eval, regenerate the plot_trends.py figures in {out_dir}/trends and log them to wandb (needs diag)
diag_opt_stats = True # 4: grad norms and update/weight ratios per block, every log_interval
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = True # torch.compile the whole network; False disables dynamo entirely
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
_defaults = {k: globals()[k] for k in config_keys} # to name the run by what changed
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# derived paths and architecture
if data_dir:
    dataset = os.path.basename(os.path.normpath(data_dir))
else:
    data_dir = os.path.join(data_root, dataset)
if not os.path.exists(os.path.join(data_dir, 'meta.json')):
    found = sorted(d for d in os.listdir(data_root) if os.path.exists(os.path.join(data_root, d, 'meta.json'))) if os.path.isdir(data_root) else []
    raise FileNotFoundError(f"no prepared dataset at {data_dir} (dataset='{dataset}', data_root='{data_root}'). "
                            f"Datasets found under {data_root}: {found or 'none'}. "
                            f"Point --data_root / $JIT_DATA_ROOT at the folder containing the dataset, or pass --data_dir directly.")

def check_prepared(data_dir, diag):
    """
    Fails fast, before building the model, if any file produced by prepare_jit_data.py / fid.py is missing,
    listing the commands that create them.
    """
    with open(os.path.join(data_dir, 'meta.json')) as f:
        meta = json.load(f)
    subset = 'in100' if meta['num_classes'] == 100 else 'in1k'
    val_dir = os.path.join(data_dir, 'val')
    has_stats = any(os.path.exists(os.path.join(data_dir, n)) for n in ('fid_stats_jit.npz', 'fid_stats.npz'))
    missing, steps = [], []
    if diag and not os.path.exists(os.path.join(val_dir, 'meta.json')):
        missing.append(f'{val_dir} (held-out split, needed for diagnostics)')
        steps.append(f"python prepare_jit_data.py --data_dir=<ImageNet ImageFolder root, e.g. /data/imagenet> --out_dir={data_dir} "
                     f"--img_size={meta['img_size']} --subset={subset} --split=val")
    needs_fid = not has_stats or (diag and (not os.path.exists(os.path.join(data_dir, 'train_inception.npy'))
                                            or not os.path.exists(os.path.join(val_dir, 'val_inception.npy'))))
    if needs_fid:
        missing.append(f'FID reference stats / Inception features in {data_dir}' + (' and val/' if diag else ''))
        steps.append(f'python fid.py --data_dir={data_dir}' + ('   # after the val split exists, so its features are written too' if diag else ''))
    if missing:
        raise FileNotFoundError('dataset preparation incomplete. missing:\n  ' + '\n  '.join(missing) +
                                '\nrun:\n  ' + '\n  '.join(steps) + ('\nor pass --diag=False to train without diagnostics (needs only fid.py)' if diag else ''))

check_prepared(data_dir, diag)
model_name, arch = resolve_arch(model, **{k: (None if v < 0 else v) for k, v in dict(
    depth=depth, hidden_size=hidden_size, num_heads=num_heads, patch_size=patch_size, bottleneck_dim=bottleneck_dim,
    in_context_len=in_context_len, in_context_start=in_context_start, mlp_ratio=mlp_ratio).items()},
    attn_res=None if attn_res == 'none' else attn_res,
    attn_res_block_size=attn_res_block_size if (attn_res == 'block' and attn_res_block_size > 0) else None,
    adaln_single=adaln_single or None, attn_res_adaln=attn_res_adaln or None, gated_bottleneck=gated_bottleneck or None,
    skip_pathway=skip_pathway or None, skip_head_blocks=skip_head_blocks or None, skip_inject_block=skip_inject_block or None,
    skip_final_layer=skip_final_layer or None, skip_residual=skip_residual or None,
    dcae_stem=dcae_stem or None, dcae_head=dcae_head or None, patch_overlap=patch_overlap or None, layernorm=layernorm or None,
    dcae_width=dcae_width if (dcae_stem or dcae_head) and dcae_width != 128 else None,
    dcae_kernel2=dcae_kernel2 if (dcae_stem or dcae_head) and dcae_kernel2 != 1 else None)
if not out_dir:
    out_dir = run_dir(runs_root, dataset, model, arch, config, _defaults, run_name)
    legacy = legacy_run_dir(runs_root, dataset, model, arch, config, _defaults, run_name)
    if init_from == 'resume' and not os.path.exists(os.path.join(out_dir, 'ckpt.pt')) and os.path.exists(os.path.join(legacy, 'ckpt.pt')):
        print(f"resuming a run saved under the old directory name {legacy}\n  (rename it to keep things tidy: mv '{legacy}' '{out_dir}')")
        out_dir = legacy
wandb_run_name = wandb_run_name or (os.path.relpath(out_dir, runs_root) if not out_dir.startswith('..') else out_dir)
print(f"data: {data_dir}\nrun:  {out_dir}")

if not compile:
    torch._dynamo.config.disable = True
torch._dynamo.config.cache_size_limit = 128
os.makedirs(out_dir, exist_ok=True)
if init_from == 'scratch':
    # a fresh run in a directory that already holds one: move the old run's outputs aside instead of mixing them (its
    # diag_*.json would otherwise feed this run's trend plots, and train_log.jsonl would be appended to)
    old = [n for n in sorted(os.listdir(out_dir)) if n in ('train_log.jsonl', 'config.json', 'ckpt.pt', 'diagnostics', 'samples', 'trends', 'eval')
           or re.fullmatch(r'ckpt_\d+\.pt', n)]
    if old:
        prev = os.path.join(out_dir, 'previous', time.strftime('%Y%m%d-%H%M%S'))
        os.makedirs(prev)
        for n in old:
            os.replace(os.path.join(out_dir, n), os.path.join(prev, n))
        print(f"{out_dir} already had a run (init_from='scratch'): moved {', '.join(old)} to {prev}")
samples_dir = os.path.join(out_dir, 'samples') # every generated sample grid
os.makedirs(samples_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' or dtype == 'float32') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data
data = MemmapImages(data_dir)
img_size, num_classes = data.img_size, data.num_classes
class_names = data.class_names()
noise_scale = noise_scale or img_size / 256
sampler = EpochSampler(len(data), batch_size)
effective_batch_size = batch_size * gradient_accumulation_steps
iters_per_epoch = len(data) // effective_batch_size
assert iters_per_epoch > 0, f'effective batch {effective_batch_size} is larger than the dataset ({len(data):,} images); lower batch_size / gradient_accumulation_steps'
max_iters = epochs * iters_per_epoch
lr = blr * effective_batch_size / 1024
print(f"{data_dir}: {len(data):,} images, {num_classes} classes, {img_size}px | effective batch {effective_batch_size}, "
      f"{iters_per_epoch:,} iters/epoch, {max_iters:,} total iters, lr {lr:.2e}, noise scale {noise_scale}")

def get_batch():
    x, y = data.get(sampler.next())
    return to_model_input(x, flip=True, device=device), torch.from_numpy(y).to(device, non_blocking=True)

# model
assert img_size % arch['patch_size'] == 0, f"img_size {img_size} not divisible by patch_size {arch['patch_size']}"
denoiser_args = dict(
    model=model_name, arch=arch, img_size=img_size, num_classes=num_classes, attn_dropout=attn_dropout, proj_dropout=proj_dropout,
    label_drop_prob=label_drop_prob, P_mean=P_mean, P_std=P_std, t_eps=t_eps, noise_scale=noise_scale,
    ema_decay1=ema_decay1, ema_decay2=ema_decay2, sampling_method=sampling_method,
    num_sampling_steps=num_sampling_steps, cfg=fid_cfg, interval_min=fid_interval_min, interval_max=fid_interval_max,
)
denoiser = build_denoiser(**denoiser_args).to(device)
denoiser.ema_warmup = ema_warmup
assert skip_pathway_drop_prob == 0 or skip_pathway, 'skip_pathway_drop_prob needs skip_pathway'
assert 0 <= skip_pathway_drop_prob < 1, 'skip_pathway_drop_prob must be in [0, 1)'
assert not skip_stop_grad or skip_pathway, 'skip_stop_grad needs skip_pathway'
assert not skip_final_layer or skip_stop_grad, 'skip_final_layer needs skip_stop_grad (otherwise the original final layer is never used)'
assert not skip_residual or skip_final_layer, 'skip_residual needs skip_final_layer (and skip_stop_grad)'
if skip_pathway:
    denoiser.net.skip_drop_prob = skip_pathway_drop_prob # training only; evaluation and sampling always use the skip
    denoiser.net.skip_stop_grad = skip_stop_grad
assert time_schedule in ('uniform', 'shift', 'train_quantile'), f'unknown time_schedule {time_schedule}'
denoiser.time_schedule = time_schedule
summary(denoiser)
print(describe_arch(model, arch))
print(f"model {model_name}: {sum(p.numel() for p in denoiser.net.parameters()) / 1e6:.1f}M params, "
      f"{(img_size // arch['patch_size']) ** 2} tokens of {arch['patch_size'] ** 2 * 3}-d patches")
with open(os.path.join(out_dir, 'config.json'), 'w') as f:
    json.dump({**config, 'resolved': {'data_dir': data_dir, 'out_dir': out_dir, 'model_name': model_name, 'arch': arch, 'noise_scale': noise_scale,
                                       'effective_batch_size': effective_batch_size, 'lr': lr, 'max_iters': max_iters,
                                       'iters_per_epoch': iters_per_epoch, 'train_images': len(data)}}, f, indent=2)
grid = img_size // denoiser.net.patch_size

encoder, projector, capture = None, None, None
if repa_encoder:
    assert repa_adapter in ('mlp', 'mta'), f"repa_adapter must be 'mlp' or 'mta', got {repa_adapter}"
    repa_coeff = repa_coeff if repa_coeff >= 0 else (0.5 if repa_adapter == 'mlp' else 0.1)
    encoder = FrozenEncoder(repa_encoder, img_size=img_size, patch_size=denoiser.net.patch_size).to(device)
    if repa_adapter == 'mlp':
        projector = Projector(denoiser.net.hidden_size, encoder.embed_dim, repa_proj_dim).to(device)
    else: # 'projector' holds the adapter too, so checkpointing / optimizer setup treat both the same way
        projector = MaskedTransformerAdapter(img_size, denoiser.net.patch_size, encoder.embed_dim, denoiser.net.hidden_size,
                                             denoiser.net.num_heads, num_classes, depth=repa_adapter_depth, mask_ratio=repa_mask_ratio).to(device)
    capture = BlockCapture(denoiser.net, repa_depth)
    with open(os.path.join(out_dir, 'config.json')) as f:
        run_config = json.load(f)
    run_config['resolved']['repa_coeff'] = repa_coeff # the published weight when repa_coeff = -1
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"alignment ({'REPA, MLP projector' if repa_adapter == 'mlp' else f'PixelREPA, {repa_adapter_depth}-block adapter, {repa_mask_ratio:.0%} masking'}): "
          f"block {repa_depth} -> {repa_encoder} ({encoder.embed_dim}-d, {grid}x{grid} grid), coeff {repa_coeff}")

# same weight decay grouping as JiT (no decay on biases / norms)
decay, no_decay = [], []
masker = None
if mask_ratio > 0:
    assert mask_weighting in ('effective_t', 'jit'), "mask_weighting must be 'effective_t' or 'jit'"
    masker = TokenMasker(denoiser.net).to(device)
    if mask_prob == 0:
        print(f"warning: mask_ratio={mask_ratio} but mask_prob=0, so no sample is ever masked; set --mask_prob (e.g. 0.5) to enable masking")
    print(f"token masking: {mask_ratio:.0%} of patches in {mask_prob:.0%} of samples, weighting {mask_weighting}")
trainable = list(denoiser.named_parameters()) + (list(projector.named_parameters()) if projector is not None else []) \
    + (list(masker.named_parameters()) if masker is not None else [])
assert optimizer_type in ('adamw', 'muon'), f"optimizer_type must be 'adamw' or 'muon', got {optimizer_type}"
# with Muon: hidden matrices of the transformer blocks (attention, MLP, adaLN projections) go to Muon; patch embedding,
# time / class embeddings, output layer, gains, biases (and REPA projector / mask token) stay on AdamW
def muon_split(name):
    # number of logical matrices fused into one weight, orthogonalized separately when muon_split_fused
    if not muon_split_fused:
        return 1
    if name.endswith('attn.qkv.weight'):
        return 3
    if name.endswith('mlp.w12.weight'):
        return 2
    if name.endswith('adaLN_modulation.1.weight') or name.endswith('t_block.1.weight'):
        return 6
    if name.endswith('attn_res_modulation.1.weight'):
        return 2
    return 1

# (adaln_single's per-block scale_shift_table is 2D but is 6 per-channel vectors, like gains / biases: AdamW)
muon_named = {id(p): (n, p) for n, p in denoiser.net.blocks.named_parameters()
              if optimizer_type == 'muon' and p.ndim >= 2 and p.requires_grad and (muon_adaln or 'adaLN_modulation' not in n)
              and 'scale_shift_table' not in n}
if optimizer_type == 'muon' and skip_pathway: # the head blocks' hidden matrices, like the trunk's; the gate like adaLN
    muon_named.update({id(p): ('skip_head.' + n, p) for n, p in denoiser.net.skip_head.named_parameters()
                       if p.ndim >= 2 and p.requires_grad and (muon_adaln or 'adaLN_modulation' not in n) and 'scale_shift_table' not in n})
    if muon_adaln:
        muon_named[id(denoiser.net.skip_gate[1].weight)] = ('skip_gate.1.weight', denoiser.net.skip_gate[1].weight)
if optimizer_type == 'muon' and muon_adaln and adaln_single: # the shared adaLN projection is the blocks' adaLN matrix
    muon_named[id(denoiser.net.t_block[1].weight)] = ('t_block.1.weight', denoiser.net.t_block[1].weight)
if optimizer_type == 'muon' and muon_adaln and gated_bottleneck: # an adaLN-style gate projection
    muon_named[id(denoiser.net.bottleneck_gate[1].weight)] = ('bottleneck_gate.1.weight', denoiser.net.bottleneck_gate[1].weight)
if optimizer_type == 'muon' and muon_adaln and attn_res_adaln: # an adaLN projection like the blocks'
    muon_named[id(denoiser.net.attn_res_modulation[1].weight)] = ('attn_res_modulation.1.weight', denoiser.net.attn_res_modulation[1].weight)
muon_params = {} # split -> params
for name, p in trainable:
    if not p.requires_grad:
        continue
    if id(p) in muon_named:
        muon_params.setdefault(muon_split(muon_named[id(p)][0]), []).append(p)
    else:
        (no_decay if p.ndim == 1 or name.endswith('.bias') or 'scale_shift_table' in name else decay).append(p)
# every group stores its peak lr; the schedule scales all of them by the same factor
adamw = torch.optim.AdamW(
    [{'params': no_decay, 'weight_decay': 0.0, 'peak_lr': lr}, {'params': decay, 'weight_decay': weight_decay, 'peak_lr': lr}],
    lr=lr, betas=(beta1, beta2), fused=device_type == 'cuda',
)
if optimizer_type == 'muon':
    assert muon_adjust_lr in ('match_rms_adamw', 'original'), f"muon_adjust_lr must be 'match_rms_adamw' or 'original', got {muon_adjust_lr}"
    # Moonshot's scaling reuses the AdamW lr and weight decay; the original recipe has its own
    peak_muon_lr, muon_wd = (lr, weight_decay) if muon_adjust_lr == 'match_rms_adamw' else (muon_lr, muon_weight_decay)
    muon = Muon([{'params': ps, 'split': k, 'peak_lr': peak_muon_lr} for k, ps in sorted(muon_params.items())],
                lr=peak_muon_lr, momentum=muon_momentum, weight_decay=muon_wd, ns_steps=muon_ns_steps, adjust_lr=muon_adjust_lr)
    optimizer = CombinedOptimizer([muon, adamw])
    flat = [p for ps in muon_params.values() for p in ps]
    print(f"optimizer: Muon ({muon_adjust_lr}, lr {peak_muon_lr:.2e}, wd {muon_wd}) on {len(flat)} block matrices "
          f"({sum(p.numel() for p in flat) / 1e6:.1f}M params; matrices per split {dict((k, len(v)) for k, v in sorted(muon_params.items()))}), "
          f"AdamW on the rest ({sum(p.numel() for p in no_decay + decay) / 1e6:.1f}M params, lr {lr:.2e})")
else:
    optimizer = adamw

# init these up here, can override if init_from='resume'
iter_num = 0
best_fid = float('inf')
wandb_run_id = None
train_seconds = 0.0

if init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
    denoiser.load_state_dict(checkpoint['model'])
    denoiser.ema_params1 = [checkpoint['model_ema1'][name].to(device) for name, _ in denoiser.named_parameters()]
    denoiser.ema_params2 = [checkpoint['model_ema2'][name].to(device) for name, _ in denoiser.named_parameters()]
    if projector is not None:
        projector.load_state_dict(checkpoint['projector'])
    if masker is not None:
        masker.load_state_dict(checkpoint['masker'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iter_num = checkpoint['iter_num']
    best_fid = checkpoint['best_fid']
    wandb_run_id = checkpoint.get('wandb_run_id')
    train_seconds = checkpoint.get('train_seconds', 0.0)
    sampler.epoch = iter_num // iters_per_epoch
    checkpoint = None # free up memory
else:
    denoiser.ema_params1 = [p.detach().clone() for p in denoiser.parameters()]
    denoiser.ema_params2 = [p.detach().clone() for p in denoiser.parameters()]

trainable_params = [p for _, p in trainable if p.requires_grad]
if compile:
    # whole network, after every hook (REPA capture, token masking) is registered so they're traced in;
    # Module.compile keeps the parameter names, so checkpoints and EMA are unaffected
    print("compiling the model... (the first iteration and the first eval take a few minutes)")
    denoiser.net.compile()
ref_stats_path = os.path.join(data_dir, 'fid_stats_jit.npz')
if not os.path.exists(ref_stats_path):
    ref_stats_path = os.path.join(data_dir, 'fid_stats.npz')
ref_stats = load_ref_stats(ref_stats_path)
metrics = InceptionMetrics(device)
print(f"FID reference: {ref_stats_path}")

# fixed subsets for the eval train / val loss (uint8 on CPU)
def _spaced(n, k):
    return np.linspace(0, n - 1, min(k, n)).round().astype(np.int64)
loss_sets = {'train': data.get(_spaced(len(data), eval_loss_images))}
if os.path.exists(os.path.join(data_dir, 'val', 'meta.json')):
    _val = MemmapImages(os.path.join(data_dir, 'val'))
    loss_sets['val'] = _val.get(_spaced(len(_val), eval_loss_images))

@torch.no_grad()
def estimate_loss(images, labels, chunk=256, seed=0):
    """
    The training objective (flow-matching v-loss, no label dropout, no masking) on fixed images with fixed noise and t,
    using the raw weights. Same seed every eval, so values are comparable across evals and between train / val.
    """
    was_training = denoiser.training
    denoiser.eval()
    total = 0.0
    with torch.random.fork_rng(devices=[device] if device_type == 'cuda' else []):
        torch.manual_seed(seed)
        for i in range(0, len(images), chunk):
            x = to_model_input(images[i:i + chunk], flip=False, device=device)
            y = torch.from_numpy(labels[i:i + chunk]).to(device)
            with ctx:
                total += flow_loss(denoiser, x, y).item() * len(x)
    denoiser.train(was_training)
    return total / len(images)

def spaced(n, k):
    # k evenly spaced indices; the memmaps are sorted by class, so this is roughly class-stratified
    return np.linspace(0, n - 1, min(k, n)).round().astype(np.int64)

if diag:
    val_dir = os.path.join(data_dir, 'val') # presence checked by check_prepared
    val_data = MemmapImages(val_dir)
    # 1/4/6: fixed batches, no flips
    xd, yd = data.get(spaced(len(data), diag_images))
    diag_train = (to_model_input(xd, flip=False, device=device), torch.from_numpy(yd).to(device))
    xd, yd = val_data.get(spaced(len(val_data), diag_images))
    diag_val = (to_model_input(xd, flip=False, device=device), torch.from_numpy(yd).to(device))
    diag_ts = diag_lib.t_grid(diag_t_bins)
    band_masks = diag_lib.radial_band_masks(img_size, diag_freq_bands, device)
    # 5: probe sets (uint8 on CPU)
    n_blocks = len(denoiser.net.blocks)
    probe_blocks = tuple(int(b) for b in diag_probe_blocks.split(',')) if diag_probe_blocks else \
        tuple(sorted({max(1, n_blocks // 3), max(1, 2 * n_blocks // 3), n_blocks}))
    assert all(1 <= b <= n_blocks for b in probe_blocks), f'diag_probe_blocks {probe_blocks} outside 1..{n_blocks}'
    probe_ts = [float(t) for t in diag_probe_ts.split(',')]
    window_tiles = [k for k in (int(v) for v in diag_window_tiles.split(',') if v) if k < grid]
    probe_train_x, probe_train_y = data.get(spaced(len(data), diag_probe_train_images))
    probe_val_x, probe_val_y = val_data.get(spaced(len(val_data), diag_probe_val_images))
    probe_train_y, probe_val_y = torch.from_numpy(probe_train_y).to(device), torch.from_numpy(probe_val_y).to(device)
    # 2/7: Inception features of real images
    train_bank = np.load(os.path.join(data_dir, 'train_inception.npy'), mmap_mode='r')
    pr_idx = np.sort(np.random.default_rng(0).choice(len(train_bank), min(fid_num_images, len(train_bank)), replace=False))
    pr_real = torch.from_numpy(train_bank[pr_idx].astype(np.float32)).to(device)
    val_feats = torch.from_numpy(np.load(os.path.join(val_dir, 'val_inception.npy')).astype(np.float32)).to(device)
    # 6: spectrum of real held-out images, on the same [0, 1] scale as the generated uint8 samples
    real_spectrum = 0
    for i in range(0, min(diag_spectrum_images, len(val_data)), 256):
        xs, _ = val_data.get(np.arange(i, min(i + 256, diag_spectrum_images, len(val_data))))
        real_spectrum = real_spectrum + diag_lib.band_power(torch.from_numpy(xs).to(device).permute(0, 3, 1, 2).float() / 255, band_masks)
    real_spectrum = real_spectrum / min(diag_spectrum_images, len(val_data))
    os.makedirs(os.path.join(out_dir, 'diagnostics'), exist_ok=True)
    print(f"diagnostics on: {len(val_data):,} held-out images, probes on blocks {probe_blocks} at t={probe_ts}, attention tiles {window_tiles}")

def get_lr(it):
    # per-iteration version of JiT's schedule: linear warmup, then constant or half-cycle cosine
    epoch = it / iters_per_epoch
    if epoch < warmup_epochs:
        return lr * epoch / warmup_epochs
    if lr_schedule == 'constant':
        return lr
    return min_lr + (lr - min_lr) * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

class SpectrumAccumulator:
    # radially binned power of the first n generated samples
    def __init__(self, n):
        self.n, self.count, self.power = n, 0, 0

    def __call__(self, x_uint8, labels=None):
        if self.count < self.n:
            x = x_uint8[:self.n - self.count]
            self.power = self.power + diag_lib.band_power(x.float() / 255, band_masks)
            self.count += len(x)

    def result(self):
        return self.power / self.count

@torch.no_grad()
def generate(spectrum=None):
    return generate_and_score(
        denoiser, metrics, ref_stats, fid_num_images, num_classes, device, ctx,
        cfg=fid_cfg, interval=(fid_interval_min, fid_interval_max), steps=num_sampling_steps,
        method=sampling_method, batch_size=gen_batch_size, on_batch=spectrum,
    )

@torch.no_grad()
def evaluate(step):
    """
    FID / IS of the EMA weights, plus the diagnostics if enabled. Returns (fid, dict of scalars for logging, dict of images).
    """
    t_start = time.time()
    log, images, record = {}, {}, {}
    for split, (xs, ys) in loss_sets.items():
        log[f'eval/{split}_loss'] = estimate_loss(xs, ys)
    spectrum = SpectrumAccumulator(diag_spectrum_images) if diag else None
    with ema_weights(denoiser, 1):
        out = generate(spectrum)
        if diag:
            # 5. probes on the EMA weights (the model that is sampled from)
            denoiser.eval()
            tr = diag_lib.probe_features(denoiser.net, probe_train_x, probe_ts, probe_blocks, num_classes, ctx, device, noise_scale)
            va = diag_lib.probe_features(denoiser.net, probe_val_x, probe_ts, probe_blocks, num_classes, ctx, device, noise_scale)
            denoiser.train()
            probes = diag_lib.run_probes(tr, probe_train_y, va, probe_val_y, num_classes)
    log['val/fid'], log['val/is'] = out['fid'], out['is']
    if not diag:
        return out['fid'], log, images

    gen_feats = out['features'].to(device)
    # 2. precision / recall
    log['val/precision'], log['val/recall'] = precision_recall(pr_real, gen_feats)
    # 7. memorization
    for k, v in diag_lib.memorization(gen_feats, train_bank, val_feats).items():
        log[f'mem/{k}'] = v
    # 3. raw weights
    if diag_raw:
        raw = generate()
        log['val/fid_raw'], log['val/is_raw'] = raw['fid'], raw['is']
        log['val/precision_raw'], log['val/recall_raw'] = precision_recall(pr_real, raw['features'].to(device))
    # 5. probes
    for (b, t), r in probes.items():
        log[f'probe/knn_block{b:02d}_t{t:.2f}'] = r['knn']
        log[f'probe/linear_block{b:02d}_t{t:.2f}'] = r['linear']
    record['probes'] = {f'block{b}_t{t}': r for (b, t), r in probes.items()}
    images['probes'] = diag_lib.probe_heatmap(probes, probe_blocks, probe_ts, f'probes on held-out images, null label (iter {step})',
                                              os.path.join(out_dir, 'diagnostics', f'probes_{step}.png'))
    # 1 + 4 + 6. per-noise-level pass on the raw weights (the ones being optimized)
    denoiser.eval()
    passes = {split: diag_lib.noise_level_pass(denoiser.net, xb, yb, diag_ts, noise_scale, t_eps, ctx, band_masks)
              for split, (xb, yb) in (('train', diag_train), ('val', diag_val))}
    # 4. locality ablation on the held-out batch: same images / noise / t, attention confined to tiles
    window_passes = {k: diag_lib.noise_level_pass(denoiser.net, *diag_val, diag_ts, noise_scale, t_eps, ctx, band_masks, tile=k)
                     for k in window_tiles}
    denoiser.train()
    t_labels = [f'{t:.3f}' for t in diag_ts.tolist()]
    for i, tl in enumerate(t_labels):
        for split in ('train', 'val'):
            log[f'noise_loss/{split}_t{tl}'] = passes[split]['v_loss'][i].item()
            log[f'noise_xmse/{split}_t{tl}'] = passes[split]['x_mse'][i].item()
        log[f'noise_gap/t{tl}'] = (passes['val']['v_loss'][i] - passes['train']['v_loss'][i]).item()
    pv = passes['val']
    for b in range(pv['attn_entropy'].shape[1]):
        log[f'attn_entropy/block{b + 1:02d}'] = pv['attn_entropy'][:, b].mean().item()
        log[f'gate_msa/block{b + 1:02d}'] = pv['gate_msa'][:, b].mean().item()
        log[f'gate_mlp/block{b + 1:02d}'] = pv['gate_mlp'][:, b].mean().item()
        log[f'attn_dist/block{b + 1:02d}'] = pv['attn_dist'][:, b].mean().item()
        if arch['in_context_len'] > 0 and b >= arch['in_context_start']: # only blocks that see in-context class tokens
            log[f'attn_prefix_mass/block{b + 1:02d}'] = pv['attn_prefix_mass'][:, b].mean().item()
    # relative held-out loss increase when attention can't leave a tile: ~0 means the model doesn't use longer-range context there
    window_increase = {k: (wp['v_loss'] / pv['v_loss'] - 1) for k, wp in window_passes.items()}
    for k, inc in window_increase.items():
        for i, tl in enumerate(t_labels):
            log[f'window_loss_increase/tile{k}_t{tl}'] = inc[i].item()
    if 'attn_res_alpha' in pv:
        # AttnRes: the final layer's mixing weight on the patch embedding per noise level (how much it reads the input
        # directly), and the full layer x source weights at the noisiest, middle and cleanest noise level
        alpha = pv['attn_res_alpha'] # [t, sublayers + final, sources]
        for i, tl in enumerate(t_labels):
            log[f'attn_res_emb_weight/t{tl}'] = alpha[i, -1, 0].item()
        rows, cols = diag_lib.attn_res_labels(arch['depth'], alpha.shape[2])
        images['attn_res_alpha'] = diag_lib.attn_res_figure(alpha.numpy(), rows, cols, t_labels,
                                                            os.path.join(out_dir, 'diagnostics', f'attn_res_alpha_{step}.png'), step)
    if 'skip_ratio' in pv: # skip pathway: RMS of the gated input patches / the trunk output, per noise level
        for i, tl in enumerate(t_labels):
            log[f'skip_ratio/t{tl}'] = pv['skip_ratio'][i, 0].item()
    if 'bypass_ratio' in pv: # gated bottleneck: RMS of the gated full-rank path / the bottleneck path, per noise level
        for i, tl in enumerate(t_labels):
            log[f'bypass_ratio/t{tl}'] = pv['bypass_ratio'][i, 0].item()
    band_labels = [f'band{k}' for k in range(diag_freq_bands)]
    for k, bl in enumerate(band_labels):
        log[f'freq_err/{bl}'] = pv['freq_rel_err'][:, k].mean().item()
        # at the cleanest noise level: model error relative to passing the input through / to the Wiener filter
        # (> 1 = the model loses detail it was given)
        log[f'freq_vs_copy/{bl}'] = (pv['freq_rel_err'][-1, k] / pv['freq_rel_err_copy'][-1, k]).item()
        log[f'freq_vs_wiener/{bl}'] = (pv['freq_rel_err'][-1, k] / pv['freq_rel_err_wiener'][-1, k]).item()
    # 6. sample vs real spectrum, log10 ratio per band (0 = matches real images)
    spec_ratio = torch.log10(spectrum.result() / real_spectrum).cpu()
    for k, bl in enumerate(band_labels):
        log[f'spectrum_log_ratio/{bl}'] = spec_ratio[k].item()
    d = os.path.join(out_dir, 'diagnostics')
    images['freq_err'] = diag_lib.heatmap(pv['freq_rel_err'].numpy(), [f't={tl}' for tl in t_labels], band_labels,
                                          f'held-out relative prediction error by frequency band (iter {step})', os.path.join(d, f'freq_err_{step}.png'))
    images['attn_entropy'] = diag_lib.heatmap(pv['attn_entropy'].numpy(), [f't={tl}' for tl in t_labels], [f'b{b + 1}' for b in range(pv['attn_entropy'].shape[1])],
                                              f'attention entropy, normalized (iter {step})', os.path.join(d, f'attn_entropy_{step}.png'), log=False)
    images['attn_dist'] = diag_lib.heatmap(pv['attn_dist'].numpy(), [f't={tl}' for tl in t_labels], [f'b{b + 1}' for b in range(pv['attn_dist'].shape[1])],
                                           f'mean attention distance / uniform (iter {step})', os.path.join(d, f'attn_dist_{step}.png'), log=False)
    if window_increase:
        images['window_loss_increase'] = diag_lib.heatmap(
            torch.stack([window_increase[k] for k in window_tiles]).numpy(), [f'tile {k}x{k} ({k * arch["patch_size"]}px)' for k in window_tiles],
            [f't={tl}' for tl in t_labels], f'held-out loss increase with attention confined to tiles (iter {step})',
            os.path.join(d, f'window_loss_increase_{step}.png'), log=False)
    record.update({'iter': step, 'scalars': log, 't': diag_ts.tolist(),
                   'passes': {split: {k: v.tolist() for k, v in p.items()} for split, p in passes.items()},
                   'window_loss_increase': {f'tile{k}': v.tolist() for k, v in window_increase.items()}})
    with open(os.path.join(d, f'diag_{step}.json'), 'w') as f:
        json.dump(record, f)
    log['diag_minutes'] = (time.time() - t_start) / 60
    return out['fid'], log, images

@torch.no_grad()
def save_samples(step):
    # one sample for each of 64 evenly spaced classes, fixed noise so grids are comparable over training and across CFG scales:
    # {out_dir}/samples/samples_{step}.png at fid_cfg, samples/samples_{step}_cfg{g}.png at sample_cfg
    labels = torch.linspace(0, num_classes - 1, 64).long().to(device)
    settings = [(fid_cfg, (fid_interval_min, fid_interval_max), '')]
    if sample_cfg > 0:
        settings.append((sample_cfg, (sample_interval_min, sample_interval_max), f'_cfg{sample_cfg:g}'))
    names = [class_names[i] for i in labels.tolist()]
    with ema_weights(denoiser, 1):
        denoiser.eval()
        for cfg, interval, suffix in settings:
            denoiser.cfg_scale, denoiser.cfg_interval = cfg, interval
            denoiser.steps, denoiser.method = num_sampling_steps, sampling_method
            with torch.random.fork_rng(devices=[device] if device_type == 'cuda' else []):
                torch.manual_seed(0)
                with ctx:
                    x = denoiser.generate(labels)
            save_labeled_grid(x, names, os.path.join(samples_dir, f'samples_{step}{suffix}.png'))
        denoiser.train()

def save_checkpoint():
    ema1, ema2 = ema_state_dicts(denoiser)
    checkpoint = {
        'model': denoiser.state_dict(),
        'model_ema1': ema1,
        'model_ema2': ema2,
        'projector': projector.state_dict() if projector is not None else None,
        'masker': masker.state_dict() if masker is not None else None,
        'optimizer': optimizer.state_dict(),
        'denoiser_args': denoiser_args,
        'iter_num': iter_num,
        'epoch': iter_num / iters_per_epoch,
        'best_fid': best_fid,
        'config': config,
        'wandb_run_id': wandb_run_id,
        'train_seconds': train_seconds,
    }
    # write then rename, so a preemption mid-save can't corrupt the only checkpoint
    t_save = time.time()
    tmp_path = os.path.join(out_dir, 'ckpt.pt.tmp')
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, os.path.join(out_dir, 'ckpt.pt'))
    if keep_iters and iter_num % keep_iters == 0:
        # milestone for eval_jit.py / eval_linear.py: the EMA1 weights and what's needed to rebuild the model
        torch.save({'model_ema1': ema1, 'denoiser_args': denoiser_args, 'iter_num': iter_num, 'epoch': iter_num / iters_per_epoch,
                    'config': config}, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
    print(f"saved checkpoint to {out_dir} at iter {iter_num} (epoch {iter_num / iters_per_epoch:.2f}) in {time.time() - t_save:.1f}s")

# logging
if wandb_log:
    import wandb
    if wandb_run_id is None:
        wandb_run_id = uuid.uuid4().hex[:8] # stored in the checkpoint so a resumed run continues the same wandb run
    wandb.init(project=wandb_project, name=wandb_run_name, group=dataset, config=config, id=wandb_run_id, resume='allow')

eval_iters = eval_every_epochs * iters_per_epoch
save_iters = max(1, save_every_epochs * iters_per_epoch)
keep_iters = keep_ckpt_every_epochs * iters_per_epoch

# training loop
X, Y = get_batch() # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
while True:

    # determine and set the learning rate for this iteration
    lr_now = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['peak_lr'] * lr_now / lr

    # FID eval of EMA weights
    if (eval_only or (iter_num > 0 and local_iter_num > 0 and (iter_num % eval_iters == 0 or iter_num == max_iters))):
        fid, eval_log, eval_images = evaluate(iter_num)
        save_samples(iter_num)
        progress_gifs = {}
        if progress_gif:
            try: # animation must never stop training
                progress_gifs = make_progress_gifs(out_dir)
            except Exception as e:
                print(f"warning: progress gif failed ({type(e).__name__}: {e}); training continues")
        trend_images = {}
        if trend_plots and diag:
            try: # plotting must never stop training
                plot_trends.VERBOSE = False
                trend_images = plot_trends.make_run_figures(out_dir)
            except Exception as e:
                print(f"warning: trend plots failed ({type(e).__name__}: {e}); training continues")
        best_fid = min(best_fid, fid)
        extra = ''.join(f", {k.split('/')[-1]} {eval_log[k]:.3f}" for k in ('val/precision', 'val/recall', 'val/fid_raw', 'mem/nn_ratio') if k in eval_log)
        loss_str = ', '.join(f"{k.split('/')[-1].replace('_', ' ')} {eval_log[k]:.4f}" for k in ('eval/train_loss', 'eval/val_loss') if k in eval_log)
        print(f"iter {iter_num} (epoch {iter_num / iters_per_epoch:.2f}): {loss_str}, FID-{fid_num_images / 1000:g}k {fid:.2f}, IS {eval_log['val/is']:.1f} (cfg {fid_cfg}){extra}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "epoch": iter_num / iters_per_epoch,
                "train_hours": train_seconds / 3600,
                **eval_log,
                **{f"diag_img/{k}": wandb.Image(v) for k, v in eval_images.items()},
                **{f"trends/{k}": wandb.Image(v) for k, v in trend_images.items()},
                **{f"samples/progress{k}": wandb.Video(v, format='gif') for k, v in progress_gifs.items()},
            }, step=iter_num)
        t0 = time.time() # don't count eval time in the next iteration's timing

    if eval_only:
        break

    if iter_num > 0 and local_iter_num > 0 and (iter_num % save_iters == 0 or iter_num == max_iters or (keep_iters and iter_num % keep_iters == 0)):
        save_checkpoint()
        t0 = time.time() # don't count the save in the next iteration's timing

    # termination conditions
    if iter_num >= max_iters:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            loss_parts = {}
            if masker is None and not (encoder is not None and repa_adapter == 'mta') and not skip_stop_grad:
                loss_fm = denoiser(X, Y) # JiT's own loss
                loss_parts['skip'] = loss_fm
            else:
                # same loss, but exposing this forward's t / dropped labels (the PixelREPA adapter is conditioned on them)
                # and, with skip_stop_grad, adding the trunk-only prediction's loss
                token_mask = masker.sample(len(X), mask_ratio, mask_prob, device) if masker is not None else None
                loss_fm, t_used, y_used = flow_loss(denoiser, X, Y, masker, token_mask, mask_weighting, return_aux=True, parts=loss_parts)
            loss = loss_fm
            if encoder is not None:
                target = encoder(X, grid)
                if repa_adapter == 'mlp':
                    loss_repa = repa_loss(capture.out, target, projector)
                else:
                    loss_repa = mta_loss(capture.out, target, projector, t_used, y_used)
                loss = loss + repa_coeff * loss_repa
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately fetch the next batch while the GPU works on the forward pass
        X, Y = get_batch()
        loss.backward()
    # 4. optimization health: grad norms and update/weight ratios per block (only on logging iterations)
    log_opt = diag_opt_stats and iter_num % log_interval == 0
    if log_opt:
        named = [(n, p) for n, p in denoiser.named_parameters() if p.requires_grad]
        grad_norms = diag_lib.grouped_norms((n, p.grad) for n, p in named)
        weights_before = [(n, p.detach().clone()) for n, p in named]
    # clip the gradient
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
    optimizer.step()
    if log_opt:
        update_norms = diag_lib.grouped_norms((n, p.detach() - w) for (n, p), (_, w) in zip(named, weights_before))
        weight_norms = diag_lib.grouped_norms(weights_before)
        del weights_before
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    denoiser.update_ema(iter_num)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    train_seconds += dt
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # with skip_stop_grad, the logged loss is the with-skip prediction's (comparable to other runs); the trunk's separately
        lossf = loss_parts['skip'].item()
        loss_trunkf = loss_parts['trunk'].item() if 'trunk' in loss_parts else None
        if not math.isfinite(lossf):
            raise RuntimeError(f"loss is {lossf} at iter {iter_num}, stopping")
        imgs_per_sec = effective_batch_size / dt
        repa_str = f", repa {loss_repa.item():.4f}" if encoder is not None else ""
        grad_str = f", grad norm {math.sqrt(sum(v ** 2 for v in grad_norms.values())):.4f}" if log_opt else ""
        trunk_str = f", trunk-only loss {loss_trunkf:.4f}" if loss_trunkf is not None else ""
        print(f"iter {iter_num} (epoch {iter_num / iters_per_epoch:.2f}): loss {lossf:.4f}{trunk_str}{repa_str}{grad_str}, time {dt*1000:.2f}ms, {imgs_per_sec:.0f} img/s")
        if wandb_log:
            log = {
                "iter": iter_num,
                "epoch": iter_num / iters_per_epoch,
                "train_hours": train_seconds / 3600,
                "train/loss": lossf,
                **({"train/loss_trunk": loss_trunkf} if loss_trunkf is not None else {}),
                "lr": lr_now,
                "imgs_per_sec": imgs_per_sec,
            }
            if optimizer_type == 'muon':
                log["lr_muon"] = peak_muon_lr * lr_now / lr
            if encoder is not None:
                log["train/repa_loss"] = loss_repa.item()
            if log_opt:
                log["grad_norm/total"] = math.sqrt(sum(v ** 2 for v in grad_norms.values()))
                log.update({f"grad_norm/{k}": v for k, v in grad_norms.items()})
                log.update({f"update_ratio/{k}": update_norms[k] / (weight_norms[k] + 1e-12) for k in update_norms})
                log.update({f"weight_norm/{k}": v for k, v in weight_norms.items()})
            wandb.log(log, step=iter_num)
        # same stats to a local file, for plot_trends.py (independent of wandb)
        record = {'iter': iter_num, 'epoch': iter_num / iters_per_epoch, 'images': iter_num * effective_batch_size,
                  'loss': lossf, 'lr': lr_now, 'imgs_per_sec': imgs_per_sec}
        if loss_trunkf is not None:
            record['loss_trunk'] = loss_trunkf
        if encoder is not None:
            record['repa_loss'] = loss_repa.item()
        if optimizer_type == 'muon':
            record['lr_muon'] = peak_muon_lr * lr_now / lr
        if log_opt:
            record['grad_norm'] = grad_norms
            record['update_ratio'] = {k: update_norms[k] / (weight_norms[k] + 1e-12) for k in update_norms}
            record['weight_norm'] = weight_norms # the ratio's denominator: a large ratio can just mean small weights
        with open(os.path.join(out_dir, 'train_log.jsonl'), 'a') as f:
            f.write(json.dumps(record) + '\n')
    iter_num += 1
    local_iter_num += 1

print(f"done. best FID-{fid_num_images / 1000:g}k {best_fid:.2f}. final eval with:\n$ python eval_jit.py --ckpt={os.path.join(out_dir, 'ckpt.pt')} --data_dir={data_dir}")
