"""
How a trained model uses its KV prefix (kv_prefix_len slots, class + learned vector, read as extra keys / values from
kv_prefix_start on): which slots are read, at which noise levels, in which blocks, by which classes and where in the
image. Inference only (the instrumented forward: the model's own math with explicit attention weights).

On held-out images (per_class per class, true labels: the prefix is class-conditional), noised to each t with fixed noise:
  read(slot)   attention weight from patch queries to that slot, averaged over images, heads and patches. 'gated' weights
               each head by its attention gate (2 sigmoid, 1 = open), i.e. how much of the slot's value reaches the output
Figure {out}/kvdiag_{run}.png:
  A  prefix mass (all slots) per reading block x t
  B  each slot's share of the prefix reads per t (rows: slots sorted by the t where they peak; each row normalized),
     i.e. do slots specialize by noise level
  C  each slot's share per reading block
  D  slot usage per class (a subset of classes, rows normalized) and the class-vs-class similarity of slot usage
  E  where patches read the prefix: per-patch prefix mass (mean over reading blocks and heads) overlaid on example images at a
     few t
plus summary numbers (slot-usage entropy overall / per t / per class, effective number of slots) in {out}/kvdiag_{run}.json.

$ python kv_prefix_diag.py --ckpt=runs/in1k_128/jit-b16_icl0/<run> --data_dir=/data/in1k_128
"""

import os
import json
import math

import numpy as np
import torch

from jit_common import load_denoiser
from data_jit import MemmapImages, to_model_input
import diagnostics as diag_lib
from util.progress import bar

# -----------------------------------------------------------------------------
ckpt = ''
data_dir = ''
ema = 1
per_class = 2 # held-out images per class
ts = '0.05,0.1,0.2,0.3,0.45,0.6,0.75,0.9'
show_ts = '0.1,0.3,0.6' # noise levels for the spatial overlays
show_images = 6
batch = 100
seed = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out = 'kvdiag'
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

assert ckpt and data_dir, 'pass --ckpt and --data_dir'
path = ckpt if ckpt.endswith('.pt') else os.path.join(ckpt, 'ckpt.pt')
run_name = os.path.basename(os.path.dirname(path) if ckpt.endswith('.pt') else ckpt.rstrip('/'))
ctx = torch.autocast('cuda', dtype=torch.bfloat16) if 'cuda' in device else torch.autocast('cpu', enabled=False)
denoiser, meta = load_denoiser(path, ema=ema, device=device)
net = denoiser.net
P = getattr(net, 'kv_prefix_len', 0)
assert P > 0, 'this checkpoint has no kv prefix'
s = meta['denoiser_args'].get('noise_scale', 1.0)
val = MemmapImages(os.path.join(data_dir, 'val'))
C = val.num_classes
rng = np.random.default_rng(seed)
idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(val.labels == c), per_class, replace=False) for c in range(C)]))
labels_all = val.labels[idx]
t_list = [float(v) for v in ts.split(',')]
show_t = [float(v) for v in show_ts.split(',')]
g = int(round(net.x_embedder.num_patches ** 0.5))
try:
    names = val.class_names()
except Exception:
    names = [str(i) for i in range(C)]

# accumulators: per t, per reading block: slot reads (raw and gated) summed over images / heads / patches; per class slot
# reads (gated, summed over blocks); per-patch prefix mass for the overlay images
read_raw, read_gated, blocks_read = {}, {}, None
class_reads = np.zeros((len(t_list), C, P))
overlay = {} # t -> [show_images, g, g]
show_idx = np.linspace(0, len(idx) - 1, show_images).round().astype(int)
for ti, t in enumerate(bar(t_list, f'prefix reads at {len(t_list)} noise levels x {len(idx)} images')):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    acc_raw, acc_gated = None, None
    for s0 in range(0, len(idx), batch):
        b = np.arange(s0, min(s0 + batch, len(idx)))
        x = to_model_input(val.images[idx[b]], flip=False, device=device)
        y = torch.from_numpy(labels_all[b]).to(device)
        e = torch.randn(x.shape, generator=gen).to(device) * s
        z = t * x + (1 - t) * e
        diag_lib.PREFIX_CAPTURE = []
        with torch.no_grad(), ctx:
            diag_lib.instrumented_forward(net, z, torch.full((len(b),), t, device=device), y)
        caps = diag_lib.PREFIX_CAPTURE
        diag_lib.PREFIX_CAPTURE = None
        if blocks_read is None:
            blocks_read = [i + 1 for i, c_ in enumerate(caps) if c_ is not None]
        cs = [c_ for c_ in caps if c_ is not None]
        raw = torch.stack([c_['w'].sum((0, 1, 2)) for c_ in cs]).cpu().numpy() # [blocks, P] summed over B, H, N
        gated = torch.stack([(c_['w'] * c_['gate'].unsqueeze(-1)).sum((0, 1, 2)) for c_ in cs]).cpu().numpy()
        acc_raw = raw if acc_raw is None else acc_raw + raw
        acc_gated = gated if acc_gated is None else acc_gated + gated
        per_img = sum((c_['w'] * c_['gate'].unsqueeze(-1)).sum((1, 2)) for c_ in cs).cpu().numpy() # [B, P] over blocks
        np.add.at(class_reads[ti], labels_all[b], per_img)
        if t in show_t:
            sel = [j for j, k in enumerate(b) if k in show_idx]
            if sel:
                mass = sum(c_['w'].sum(-1).mean(1) for c_ in cs) / len(cs) # [B, N] prefix mass per patch, mean over heads and blocks
                for j in sel:
                    overlay.setdefault(t, {})[int(b[j])] = mass[j].reshape(g, g).cpu().numpy()
    H_, Npatch = net.blocks[0].attn.num_heads, g * g
    norm = len(idx) * H_ * Npatch
    read_raw[t], read_gated[t] = acc_raw / norm, acc_gated / norm # [blocks, P]: mean attention per patch query and head

T, Bk = len(t_list), len(blocks_read)
R = np.stack([read_gated[t] for t in t_list]) # [T, blocks, P]
Rraw = np.stack([read_raw[t] for t in t_list])
mass_bt = Rraw.sum(-1) # [T, blocks] prefix mass
slot_t = R.sum(1) # [T, P]
slot_b = R.sum(0) # [blocks, P]

def entropy(p, axis=-1):
    p = p / np.clip(p.sum(axis, keepdims=True), 1e-12, None)
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis)

summary = {
    'blocks_read': blocks_read, 't': t_list,
    'prefix_mass_mean_by_t': mass_bt.mean(1).tolist(), 'prefix_mass_by_block': mass_bt.mean(0).tolist(),
    'effective_slots_overall': float(np.exp(entropy(slot_t.sum(0)))), # exp(entropy) of the overall slot distribution
    'effective_slots_by_t': np.exp(entropy(slot_t)).tolist(),
    'effective_slots_per_class_median': float(np.median(np.exp(entropy(class_reads.sum(0) + 1e-12)))),
    'slot_peak_t': [t_list[int(i)] for i in (slot_t / np.clip(slot_t.sum(0, keepdims=True), 1e-12, None)).argmax(0)],
}
# how much slot usage depends on t: mutual information between slot and t (bits) under the gated read distribution
joint = slot_t / slot_t.sum()
pt, ps = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
summary['mi_slot_t_bits'] = float((joint * np.log2(np.clip(joint / (pt * ps), 1e-12, None))).sum())
summary['max_mi_slot_t_bits'] = float(min(math.log2(T), math.log2(P)))
cj = class_reads.sum(0); cj = cj / cj.sum()
pc, ps2 = cj.sum(1, keepdims=True), cj.sum(0, keepdims=True)
summary['mi_slot_class_bits'] = float((cj * np.log2(np.clip(cj / (pc * ps2), 1e-12, None))).sum())
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, f'kvdiag_{run_name}.json'), 'w') as f:
    json.dump({'summary': summary, 'settings': {k: globals()[k] for k in config_keys}, 'read_gated': R.tolist(),
               'read_raw': Rraw.tolist(), 'class_reads': class_reads.tolist()}, f)
print(f"{run_name}: {P} prefix slots read in blocks {blocks_read}")
print(f"  prefix mass (raw attention on all slots) by t: " + ', '.join(f'{t:g}: {m:.3f}' for t, m in zip(t_list, summary['prefix_mass_mean_by_t'])))
print(f"  effective slots (exp entropy of gated reads): overall {summary['effective_slots_overall']:.1f} of {P}; per t "
      + ', '.join(f'{v:.1f}' for v in summary['effective_slots_by_t']) + f"; per class (median) {summary['effective_slots_per_class_median']:.1f}")
print(f"  slot <-> t mutual information {summary['mi_slot_t_bits']:.3f} bits (max {summary['max_mi_slot_t_bits']:.2f}); "
      f"slot <-> class {summary['mi_slot_class_bits']:.3f} bits")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
# A: mass per block x t
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(mass_bt.T, aspect='auto', cmap='viridis', origin='lower')
ax.set_xticks(range(T)); ax.set_xticklabels([f'{t:g}' for t in t_list], fontsize=7)
ax.set_yticks(range(Bk)); ax.set_yticklabels([f'b{b}' for b in blocks_read], fontsize=7)
ax.set_xlabel('t'); ax.set_title('A  prefix attention mass (block x t)', fontsize=9); plt.colorbar(im, ax=ax, fraction=0.046)
# B: slot x t, rows normalized, sorted by peak t
ax = fig.add_subplot(gs[0, 1])
st = (slot_t / np.clip(slot_t.sum(0, keepdims=True), 1e-12, None)).T # [P, T] each slot's profile over t
order = np.argsort(np.argmax(st, 1) + 1e-3 * np.arange(P))
im = ax.imshow(st[order], aspect='auto', cmap='magma')
ax.set_xticks(range(T)); ax.set_xticklabels([f'{t:g}' for t in t_list], fontsize=7)
ax.set_yticks(range(P)); ax.set_yticklabels([str(o) for o in order], fontsize=5)
ax.set_xlabel('t'); ax.set_ylabel('slot (sorted by peak t)', fontsize=8)
ax.set_title(f"B  each slot's reads over t (row-normalized)\nslot-t MI {summary['mi_slot_t_bits']:.2f} of {summary['max_mi_slot_t_bits']:.2f} bits", fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046)
# C: slot x block, rows normalized
ax = fig.add_subplot(gs[0, 2])
sb = (slot_b / np.clip(slot_b.sum(0, keepdims=True), 1e-12, None)).T
im = ax.imshow(sb[order], aspect='auto', cmap='magma')
ax.set_xticks(range(Bk)); ax.set_xticklabels([f'b{b}' for b in blocks_read], fontsize=7)
ax.set_yticks(range(P)); ax.set_yticklabels([str(o) for o in order], fontsize=5)
ax.set_title("C  each slot's reads over blocks (row-normalized)", fontsize=9); plt.colorbar(im, ax=ax, fraction=0.046)
# D: class x slot usage for a class subset, and the overall slot distribution
ax = fig.add_subplot(gs[0, 3])
cr = class_reads.sum(0)
cr = cr / np.clip(cr.sum(1, keepdims=True), 1e-12, None)
sub = np.linspace(0, C - 1, min(40, C)).round().astype(int)
im = ax.imshow(cr[sub][:, order], aspect='auto', cmap='viridis')
ax.set_yticks(range(len(sub))); ax.set_yticklabels([names[c][:14] for c in sub], fontsize=5)
ax.set_xticks(range(P)); ax.set_xticklabels([str(o) for o in order], fontsize=4, rotation=90)
ax.set_title(f"D  slot usage per class (rows normalized)\nslot-class MI {summary['mi_slot_class_bits']:.2f} bits", fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046)
# E: overlays
sub_gs = gs[1, :].subgridspec(len(show_t) + 1, show_images, hspace=0.25, wspace=0.05)
for j, k in enumerate(show_idx):
    img = val.images[idx[k]]
    ax = fig.add_subplot(sub_gs[0, j]); ax.imshow(img); ax.axis('off'); ax.set_title(names[labels_all[k]][:18], fontsize=7)
    for r, t in enumerate(show_t):
        ax = fig.add_subplot(sub_gs[r + 1, j]); ax.imshow(img, alpha=0.45)
        m = overlay.get(t, {}).get(int(k))
        if m is not None:
            ax.imshow(np.kron(m, np.ones((img.shape[0] // g, img.shape[0] // g))), cmap='inferno', alpha=0.6)
        ax.axis('off')
        if j == 0:
            ax.set_title(f't={t:g}: prefix mass per patch', fontsize=7, loc='left')
fig.suptitle(f'{run_name}: KV-prefix usage ({len(idx)} held-out images, gated reads)', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(out, f'kvdiag_{run_name}.png'), dpi=140)
print(f'wrote {out}/kvdiag_{run_name}.png and .json')
