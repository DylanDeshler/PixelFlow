"""
Trains a DiTo tokenizer (dito_model.py: the paper's f8c4 diffusion autoencoder, from the official code) on a prepared pixel
dataset, for the pixels-vs-latents allocation study. The objective, prediction target, latent LayerNorm, noise
synchronization and decoder sampler are the paper's, unchanged; what differs is the study's setup:
  data       the study's 128px center crops with random horizontal flips (data_jit.MemmapImages), optionally a nested class-
             stratified subset (--train_frac, subsets.py), so a tokenizer is trained on exactly its generator's images
  optimizer  Muon (match_rms_adamw) on the hidden conv filters and linear weights of encoder and decoder, AdamW on the rest:
             the first and last conv of each network, the 1x1 latent conv, the time embedding, norms and biases (Muon's
             guidance: input / output layers and gains on AdamW). --lr is the shared peak lr, --aux_lr_mult scales AdamW's.
             The paper: AdamW 1e-4, batch 64, no EMA (kept: batch 64 default, no EMA)
  schedule   linear warm-up over --warmup_updates, then constant; run length by --max_updates (or converged by rFID)
Evaluation every --eval_every updates on balanced held-out images: rFID (reconstructions decoded with the paper's 50 Euler
steps, against the training-set reference stats, as latent_rfid.py), PSNR and the decoder's loss; a checkpoint is kept at
every eval (ckpt_{iter}.pt: the tokenizer investment axis) and ckpt.pt is refreshed for resuming. The training compute so far
(updates x batch x 3 x (encoder + UNet forward), measured) is logged and stored in every checkpoint (train_pflops), which
latent pipelines read as the tokenizer's cost.

$ python train_dito.py --data_dir=/data/in1k_128 --size=B                       # full data, DiTo-B
$ python train_dito.py --data_dir=/data/in1k_128 --size=S --train_frac=0.125     # DiTo-S on N/8
"""

import os
import sys
import json
import math
import time
from contextlib import nullcontext

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore', message=r'.*upsample_nearest is deprecated.*')

from data_jit import MemmapImages, EpochSampler
from dito_model import DiTo
from muon import Muon, CombinedOptimizer
from fid import InceptionMetrics, load_ref_stats, frechet_distance, to_uint8
from util.progress import bar

sys.stdout.reconfigure(line_buffering=True)

# -----------------------------------------------------------------------------
data_dir = ''
runs_root = 'runs'
out_dir = '' # default {runs_root}/{dataset}/dito-{size}/{changes}
init_from = 'scratch' # or 'resume'
size = 'B' # 'B' (paper), 'L' (paper), 'S' (half of B's channels; not in the paper)
train_frac = 1.0
batch_size = 64 # the paper's
gradient_accumulation_steps = 1
optimizer_type = 'muon' # 'muon' (study) or 'adamw' (the paper's: lr 1e-4, betas (0.9, 0.999), weight decay 0.01)
lr = 1e-4
aux_lr_mult = 1.0
weight_decay = 0.01 # the paper's AdamW weight decay (AdamW groups except norms / biases; Muon's decoupled decay)
beta1 = 0.9
beta2 = 0.999
muon_momentum = 0.95
warmup_updates = 500
max_updates = 100000
zaug_p = 0.1 # noise synchronization probability (the paper's noise-sync configs)
# evaluation
eval_every = 10000 # updates (0 = no evals, e.g. CPU smoke tests; the final checkpoint is still kept)
eval_images = 5000 # rFID@5K as the paper (balanced classes; multiple of the class count)
decode_steps = 50 # the paper's Euler steps
converge_tol = 0.0 # > 0: stop when rFID improved by < this over the last third of training (jit_common.converged), as the generators
converge_patience = 2
converge_min_updates = 20000
settle_tol = 0.0 # > 0 (an lr candidate): stop once the predicted rFID at settle_target_updates (predict_fid on the rFID curve)
                 # changed < settle_tol between consecutive evals settle_patience times, after settle_min_updates; or at settle_max_updates
settle_target_updates = 300000
settle_patience = 2
settle_min_updates = 30000
settle_max_updates = 100000
log_interval = 100
seed = 0
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
_defaults = {k: globals()[k] for k in config_keys}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

assert data_dir, 'pass --data_dir'
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' or dtype == 'float32') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data = MemmapImages(data_dir)
val = MemmapImages(os.path.join(data_dir, 'val'))
img_size = data.img_size
dataset = os.path.basename(os.path.normpath(data_dir))
if not out_dir:
    keep = ('train_frac', 'optimizer_type', 'lr', 'aux_lr_mult', 'weight_decay', 'warmup_updates', 'zaug_p', 'batch_size')
    changed = [f'{k}{config[k]:g}' if isinstance(config[k], float) else f'{k}{config[k]}' for k in keep if config[k] != _defaults[k]]
    out_dir = os.path.join(runs_root, dataset, f'dito-{size}', '_'.join(changed) or 'base')
os.makedirs(out_dir, exist_ok=True)
print(f"data: {data_dir} ({img_size}px)\nrun:  {out_dir}")

train_rows = None
if train_frac < 1:
    from subsets import subset_rows
    train_rows = subset_rows(data_dir, data.labels, train_frac)
    print(f"training subset: {train_frac:g} of the data, {len(train_rows):,} images (nested, class-stratified: {data_dir}/subsets/)")
sampler = EpochSampler(len(data), batch_size, seed=seed, indices=train_rows)

model = DiTo(size, img_size=img_size, zaug_p=zaug_p).to(device)
n_enc = sum(p.numel() for p in model.encoder.parameters())
n_dec = sum(p.numel() for p in model.renderer.parameters())

# measured compute (FLOP counter, one image; FLOPs = 2 x multiply-adds)
from torch.utils.flop_counter import FlopCounterMode
def _flops(fn):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops() / 1e9
_x = torch.zeros(1, 3, img_size, img_size, device=device)
_z = torch.zeros(1, *model.z_shape, device=device)
enc_gflops = _flops(lambda: model.encode(_x))
unet_gflops = _flops(lambda: model.renderer(_x, t=torch.full((1,), 500.0, device=device), z_dec=_z))
train_gflops = 3 * (enc_gflops + unet_gflops) # per training image (forward + backward ~ 3x forward)
decode_gflops = decode_steps * unet_gflops
print(f"DiTo-{size}: encoder {n_enc / 1e6:.1f}M, decoder UNet {n_dec / 1e6:.1f}M params; latent {model.z_shape}; "
      f"per image: encode {enc_gflops:.1f} GF, UNet forward {unet_gflops:.1f} GF -> training {train_gflops:.0f} GF, "
      f"{decode_steps}-step decode {decode_gflops:.0f} GF")

# optimizer groups
aux_names = ('encoder.0.conv_in.', 'encoder.0.conv_out.', 'encoder.1.', 'renderer.embed_image.', 'renderer.output.', 'renderer.embed_time.')
muon_params, decay, no_decay = [], [], []
for n, p in model.named_parameters():
    if optimizer_type == 'muon' and p.ndim >= 2 and not n.startswith(aux_names):
        muon_params.append(p)
    elif p.ndim == 1 or n.endswith('.bias'):
        no_decay.append(p)
    else:
        decay.append(p)
aux_lr = lr * (aux_lr_mult if optimizer_type == 'muon' else 1.0)
adamw = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0.0, 'peak_lr': aux_lr, 'lr': aux_lr},
                           {'params': decay, 'weight_decay': weight_decay, 'peak_lr': aux_lr, 'lr': aux_lr}],
                          lr=aux_lr, betas=(beta1, beta2), fused=device_type == 'cuda')
if optimizer_type == 'muon':
    muon = Muon([{'params': muon_params, 'peak_lr': lr}], lr=lr, momentum=muon_momentum, weight_decay=weight_decay, adjust_lr='match_rms_adamw')
    optimizer = CombinedOptimizer([muon, adamw])
    print(f"optimizer: Muon on {len(muon_params)} hidden conv / linear weights ({sum(p.numel() for p in muon_params) / 1e6:.1f}M params, lr {lr:.2e}), "
          f"AdamW on the rest ({sum(p.numel() for p in no_decay + decay) / 1e6:.1f}M params, lr {aux_lr:.2e})")
else:
    optimizer = adamw
    print(f"optimizer: AdamW lr {lr:.2e} (the paper's)")

iter_num, train_seconds, fid_hist = 0, 0.0, []
if init_from == 'resume':
    ck = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    iter_num, train_seconds, fid_hist = ck['iter_num'], ck.get('train_seconds', 0.0), ck.get('fid_hist', [])
    sampler.epoch = iter_num * batch_size // sampler.n
    print(f"resumed at update {iter_num}")
    del ck

with open(os.path.join(out_dir, 'config.json'), 'w') as f:
    json.dump({**config, 'resolved': {'size': size, 'img_size': img_size, 'z_shape': list(model.z_shape), 'encoder_params': n_enc,
                                       'decoder_params': n_dec, 'encode_gflops': enc_gflops, 'unet_gflops': unet_gflops,
                                       'train_gflops_per_image': train_gflops, 'decode_gflops': decode_gflops,
                                       'decode_steps': decode_steps, 'train_images': sampler.n, 'out_dir': out_dir}}, f, indent=2)

if compile and device_type == 'cuda': # in place, so checkpoint keys stay the plain module's
    model.encoder.compile()
    model.renderer.compile()

# held-out evaluation set: balanced classes, the first eval_images / classes images of each
labels = np.asarray(val.labels)
C = int(labels.max()) + 1
per = eval_images // C
eval_idx = np.sort(np.concatenate([np.flatnonzero(labels == c)[:per] for c in range(C)]))
ref_path = os.path.join(data_dir, 'fid_stats_jit.npz')
ref_path = ref_path if os.path.exists(ref_path) else os.path.join(data_dir, 'fid_stats.npz')
ref_stats = load_ref_stats(ref_path) if eval_every > 0 else None
metrics = None # loaded at the first eval

def batch_pixels(images_u8):
    return torch.from_numpy(np.asarray(images_u8)).to(device).permute(0, 3, 1, 2).float().div_(127.5).sub_(1)

@torch.no_grad()
def evaluate():
    global metrics
    metrics = metrics or InceptionMetrics(device)
    model.eval()
    feats, mse, losses = [], 0.0, []
    g = torch.Generator(device=device).manual_seed(seed)
    for s in bar(range(0, len(eval_idx), 100), f'rFID-{len(eval_idx) / 1000:g}k: encode + {decode_steps}-step decode'):
        x = batch_pixels(val.images[eval_idx[s:s + 100]])
        with ctx:
            z = model.encode(x)
            rec = model.decode(z, noise=torch.randn(x.shape, generator=g, device=device)).float().clamp(-1, 1)
            losses.append(model.loss(x)[0].item())
        mse += ((rec - x) / 2).pow(2).mean(dim=(1, 2, 3)).sum().item()
        feats.append(metrics.features(to_uint8(rec))[0])
    model.train()
    rfid = frechet_distance(torch.cat(feats), ref_stats, device)
    psnr = 10 * math.log10(1 / (mse / len(eval_idx)))
    return {'rfid': rfid, 'psnr': psnr, 'val_loss': float(np.mean(losses))}

def save(kept):
    state = {'model': model.state_dict(), 'iter_num': iter_num, 'train_seconds': train_seconds, 'fid_hist': fid_hist, 'config': config,
             'size': size, 'img_size': img_size, 'zaug_p': zaug_p, 'decode_steps': decode_steps,
             'train_pflops': iter_num * batch_size * train_gflops / 1e6, 'train_gflops_per_image': train_gflops,
             'encode_gflops': enc_gflops, 'decode_gflops': decode_gflops, 'encoder_params': n_enc, 'decoder_params': n_dec,
             'train_images': sampler.n, 'train_frac': train_frac}
    if kept: # the tokenizer at this investment (no optimizer state), for prepare_latents / latent_vae ('dito:<path>')
        torch.save({k: v for k, v in state.items()}, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
    torch.save({**state, 'optimizer': optimizer.state_dict()}, os.path.join(out_dir, 'ckpt.pt.tmp'))
    os.replace(os.path.join(out_dir, 'ckpt.pt.tmp'), os.path.join(out_dir, 'ckpt.pt'))

from jit_common import converged
model.train()
t_last = time.time()
while True:
    if iter_num > 0 and eval_every <= 0 and iter_num == max_updates:
        save(kept=True)
    if iter_num > 0 and eval_every > 0 and (iter_num % eval_every == 0 or iter_num == max_updates):
        r = evaluate()
        fid_hist.append((iter_num, r['rfid']))
        pf = iter_num * batch_size * train_gflops / 1e6
        print(f"update {iter_num}: rFID-{len(eval_idx) / 1000:g}k {r['rfid']:.2f}, PSNR {r['psnr']:.2f} dB, val loss {r['val_loss']:.4f}; "
              f"training {pf:,.1f} PFLOPs, {train_seconds / 3600:.2f} h")
        with open(os.path.join(out_dir, 'eval_log.jsonl'), 'a') as f:
            f.write(json.dumps({'iter': iter_num, 'train_pflops': pf, 'train_hours': train_seconds / 3600, **r}) + '\n')
        if settle_tol > 0:
            import predict_fid
            preds = [predict_fid.predict(fid_hist[:i], settle_target_updates) for i in range(1, len(fid_hist) + 1)]
            why = 'reached settle_max_updates' if iter_num >= settle_max_updates else \
                  ('rFID prediction settled' if iter_num >= settle_min_updates and predict_fid.settled(preds, settle_tol, settle_patience) else None)
            print(f"settle check: predicted rFID at {settle_target_updates:,} updates: {preds[-1]}" + (f" -> stopping: {why}" if why else ""))
            if why:
                max_updates = iter_num
                with open(os.path.join(out_dir, 'settled.json'), 'w') as f:
                    json.dump({'iter': iter_num, 'reason': why, 'pred_rfid': preds[-1], 'preds': preds}, f, indent=1)
        if converge_tol > 0:
            conv, detail = converged(fid_hist, converge_tol, patience=converge_patience, min_iter=converge_min_updates)
            if detail:
                print(f"convergence check (rFID): {detail}" + (" -> converged, stopping" if conv else ""))
            if conv:
                max_updates = iter_num
        save(kept=True)
        t_last = time.time()
    if iter_num >= max_updates:
        break
    scale = min(1.0, iter_num / warmup_updates) if warmup_updates > 0 else 1.0
    for opt in (optimizer.optimizers if optimizer_type == 'muon' else [optimizer]):
        for gr in opt.param_groups:
            gr['lr'] = gr['peak_lr'] * scale
    idx = np.sort(sampler.next())
    x_all = batch_pixels(data.images[idx])
    flip = torch.rand(len(x_all), device=device) < 0.5
    x_all = torch.where(flip.view(-1, 1, 1, 1), x_all.flip(-1), x_all)
    tot = 0.0
    for x in x_all.chunk(gradient_accumulation_steps):
        with ctx:
            loss = model.loss(x)[0]
        (loss / gradient_accumulation_steps).backward()
        tot += loss.item() / gradient_accumulation_steps
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    iter_num += 1
    if iter_num % log_interval == 0:
        now = time.time()
        dt = (now - t_last) / log_interval
        train_seconds += now - t_last
        t_last = now
        if not math.isfinite(tot):
            raise RuntimeError(f'loss is {tot} at update {iter_num}')
        print(f"update {iter_num}: loss {tot:.4f}, {batch_size / dt:.0f} img/s, lr x{scale:.2f}")
        with open(os.path.join(out_dir, 'train_log.jsonl'), 'a') as f:
            f.write(json.dumps({'iter': iter_num, 'loss': tot, 'imgs_per_sec': batch_size / dt, 'train_hours': train_seconds / 3600}) + '\n')
print(f"done: {out_dir} at update {iter_num}")
