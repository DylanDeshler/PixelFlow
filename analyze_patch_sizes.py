"""
A no-training estimate of what fixed-budget quadtree tokenization (model_quadtree.py) could gain, from three uniform
plain-JiT checkpoints trained alike at 32 / 16 / 8 px patches (same steps): generation where every step's clean-image
estimate is stitched region by region from the three models according to a router's layout, scored by FID-10k against
the models alone, and where that gain comes from (t, K, patch size, which regions).

$ python analyze_patch_sizes.py --ckpts=<32px ckpt>,<16px ckpt>,<8px ckpt> --data_dir=/data/in1k_128

Parts (every generation uses the same seed, so the same noise and labels: samples are comparable image by image):
  anchors      FID of each uniform model alone. The 16px model is the K=64 baseline, the 8px model the K=256 ceiling.
  stitched     FID per router x budget K. A uniform layout equal to an anchor (K=16 / 64 / 256) reuses it.
               Routers:
                 loss     the no-external-representation router (proposal Section 7): the hierarchical greedy on the
                          largest expected loss reductions. With the 8px estimate standing in for E[x | z_t], the
                          expected error of the level-l model over a region is ||x_l - x_8||^2 + the posterior variance
                          (the same for every model), so the expected split gain is the measured gain with x_8 in place
                          of x. This is what a split-value regression head converges to (the conditional mean of the
                          realized gain), here without training but borrowing the 8px model's estimate: optimistic.
                 dino / pixel  on the 16px model's null-label estimate, as QuadJiT training routes
                 disagree ||x_coarse - x_fine||^2 per region (= loss for 16px parents; ignores the 16px model's own
                          error for 32px parents)
                 random, uniform (coarse-first, centre-out)
                 X_exact  router X's scores with the exact allocation (allocate_exact) instead of the greedy: the
                          layout maximizing the summed scores, where the greedy never splits a node with a small gain to
                          reach large gains among its children. loss_exact: the loss router's own objective, optimized.
               Flicker: the fraction of the image whose patch size changes from one layout to the next within a
               trajectory (every re-route), from the sample pass.
               "gap closed" = (FID_16 - FID) / (FID_16 - FID_8): the share of the 16px -> 8px improvement reached.
  attribution  For each router in --attr_routers at --attr_tokens:
                 by t           stitching only for t in each window of --attr_t, the 16px model elsewhere
                 by patch size  the same layouts with the 8px regions rendered by the 16px model (what splitting buys)
                                and with the 32px regions rendered by the 16px model (what merging costs)
               The by-K view is the stitched part's K sweep.
  regions      (--regions, default on; minutes) Which regions benefit, per t: on held-out images noised to each t in --ts,
               the split gain of every candidate patch (the error the finer model removes over its region: e32 - e16,
               e16 - e8; split exactly into structure (8px block means) + detail), and every heuristic's rank
               correlation with it per level: dino (as routed: foreground gate x child divergence), dino_gate, dino_div,
               pixel, loss (expected gains: how much of the realized gain is predictable from z_t), disagree,
               centre, random. The same in Inception feature space (FID's) on a subset: a patch's gain
               is how much closer the image's pool features get to the real image's when that region is upgraded to the
               finer model. Plus the heuristics' correlations with each other, and the measured error each allocation
               removes vs the uniform layout at the same K: greedy vs exact, on the measured (oracle) and expected
               (loss) gains.
  recon        (--recon, opt-in) Reconstruction FID: held-out images noised to t0 and sampled from there with stitched
               layouts, including the loss-oracle (layout re-chosen every step from the measured gains against the true
               image). A router beating the loss-oracle on FID means loss is the wrong target for splitting.

Caveats.
  * The three models differ in compute per pixel and bottleneck compression as well as patch size; a budget-K QuadJiT
    is one network, so this is a guide to the gains, not a prediction of a trained model.
  * Stitched sampling can show seams between the models' predictions, which a single network wouldn't have. Measured
    by the seam ratio (mean |neighbour difference| across token boundaries / inside tokens) of every configuration's
    samples with their final layouts, against the same layouts on the 16px / 8px models' samples of the same noise
    (placebos: no stitching). "seams x1.00" = no more boundary contrast than the placebo. The samples figures show
    every image clean and with its layout, to judge by eye.
  * A stitched configuration runs the models its layout needs (all three: ~1.3x the 8px model's cost), plus the 16px
    routing estimate (dino / pixel: ~0.25x) and DINOv2 at 224px (dino: ~1x). --route_every reuses a layout for that many
    network calls of a trajectory.
  * Results are written after every generation; rerunning with the same --out and FID settings skips finished ones.
    --fid=False: samples, seams and figures only (~a minute per configuration), e.g. on top of a finished FID run.

Outputs: {out}.json (every number), {out}_samples.npz (--n_samples images per configuration, with final layouts), and
  {out}_fid.png          FID vs K per router, with the anchors
  {out}_attribution.png  FID change vs the 16px model: full stitching, per t window, without splits / merges
  {out}_samples_K{K}.png the same noise through the 16px / 8px models and every router at K: each image, then with
                         its final layout drawn; FID and seams in the titles
  {out}_heuristics.png   rank correlation of each heuristic with the loss / feature split gains vs t, per level, and
                         the heuristics' correlations with each other at --viz_t
  {out}_gains.png        gain maps per region with the loss-oracle's, feature-oracle's and routers' layouts at --viz_t
"""

import os
import sys
import json
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_jit import MemmapImages, to_model_input
from jit_common import load_denoiser, generate_and_score
from fid import InceptionMetrics, load_ref_stats, to_uint8, precision_recall
from diagnostics import seam_ratio
from model_quadtree import QuadGeometry, QuadRouter, allocate, cell_maps, layout_agreement, draw_layout, LEVEL_COLORS

# -----------------------------------------------------------------------------
ckpts = '' # comma list of three plain-JiT checkpoints (32 / 16 / 8 px patches, any order), trained alike to the same step
data_dir = '' # the prepared dataset they trained on (fid_stats.npz, train_inception.npy, val/)
ema = 1
seed = 0
# stitched sampling FID
num_images = 10000
tokens = '64,100,160' # budgets K (16 + 3m at 128px)
routers = 'loss,loss_exact,dino,pixel,random,uniform' # + disagree; any router + _exact: exact allocation
attr_routers = 'loss,dino'
attr_tokens = 64
attr_t = '0,0.5,0.8,1' # t window edges for the by-t attribution
fid = True # FID per configuration (False: samples, seams and figures only, ~a minute per configuration)
n_samples = 32 # samples per configuration (same seed and labels, spread over the classes) for seams and figures
show_samples = 6 # of those, in each samples figure
route_every = 1 # reuse a layout for this many network calls of a trajectory (1 = re-route every call)
cfg = 1.0
num_sampling_steps = 50
sampling_method = 'heun'
gen_batch_size = 128
compile = False # torch.compile every network (one-time compile per batch shape; outputs match to bf16 rounding)
dino = 'dinov2_vitb14_reg'
gate_tau = 1.0
# regions: which regions benefit, and the heuristics' correlation with that
regions = True
n_images = 256
ts = '0.3,0.5,0.7,0.85,0.95'
batch_size = 64
feature_images = 32 # Inception-feature gains (1 + candidates Inception passes per image and t)
feature_ts = '0.5,0.85'
# reconstruction FID from t0 (with the loss-oracle)
recon = False
recon_num_images = 10000
recon_t0 = '0.3'
recon_routers = 'oracle,loss,dino,random,uniform'
# figures
viz_t = 0.85 # one of --ts (and --feature_ts for the feature-oracle column)
viz_images = 8
out = '' # output prefix; default {data_dir name}_patch_sizes
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

assert ckpts and data_dir, 'pass --ckpts (three plain-JiT checkpoints) and --data_dir'
sys.stdout.reconfigure(line_buffering=True) # progress lines reach a redirected log (nohup ... > log) as they happen
out = out or f"{os.path.basename(os.path.normpath(data_dir))}_patch_sizes"
ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if 'cuda' in device else nullcontext()

# models: coarsest first
loaded = sorted((load_denoiser(p, ema, device) for p in ckpts.split(',')), key=lambda dm: -dm[0].net.patch_size)
assert len(loaded) == 3, 'pass exactly three checkpoints (32 / 16 / 8 px)'
models, steps = [d for d, _ in loaded], [m.get('iter_num') for _, m in loaded]
print('checkpoints at iteration ' + ', '.join(f'{d.net.patch_size}px {s}' for d, s in zip(models, steps)))
if len(set(steps)) > 1:
    print('WARNING: the checkpoints are at different iterations, so the comparison mixes patch size with training length')
patches = [d.net.patch_size for d in models]
if compile:
    for d in models:
        d.net = torch.compile(d.net)
nets = [d.net for d in models]
img, num_classes = models[0].img_size, models[0].num_classes
assert all(d.img_size == img and d.num_classes == num_classes for d in models), 'the checkpoints disagree on resolution / classes'
assert len({d.noise_scale for d in models}) == 1, 'the checkpoints disagree on noise_scale'
noise_scale = models[0].noise_scale
geo = QuadGeometry(img, patches[-1], 3)
assert geo.patches == patches, f'patch sizes must be p, 2p, 4p (got {patches})'
G, f_cell = geo.grids[-1], img // geo.grids[-1]
budgets = [int(k) for k in tokens.split(',')]
for K in budgets + [attr_tokens]:
    geo.check_budget(K)
router_list = [r for r in routers.split(',') if r]
attr_list = [r for r in attr_routers.split(',') if r]
edges = [float(v) for v in attr_t.split(',')]
windows = list(zip(edges[:-1], edges[1:]))
router = QuadRouter(geo, geo.n0, 'uniform', gate_tau=gate_tau, num_classes=num_classes, img=img)
if regions or 'dino' in set(router_list) | set(attr_list) | (set(recon_routers.split(',')) if recon else set()):
    router.load_dino(dino)
router.to(device)
LEVEL_TOKENS = {geo.counts[l]: l for l in range(geo.L)} # uniform layouts of one level: K = 16 / 64 / 256 at 128px

data = MemmapImages(os.path.join(data_dir, 'val') if os.path.isdir(os.path.join(data_dir, 'val')) else data_dir)
names_all = data.class_names()

# results so far: finished generations are reused if they were made with the same models and sampling settings
FID_KEYS = ('ckpts', 'ema', 'seed', 'num_images', 'cfg', 'num_sampling_steps', 'sampling_method', 'route_every', 'dino',
            'gate_tau', 'recon_num_images', 'gen_batch_size') # gen_batch_size: which noise each image gets
results = {'patches': patches, 'fid': {}, 'seams': {}, 'flicker': {}, 'regions': {}, 'recon': {}}
samples = {}
if os.path.exists(out + '.json'):
    with open(out + '.json') as fh:
        prev = json.load(fh)
    if all(prev.get('config', {}).get(k) == config[k] for k in FID_KEYS):
        for part in ('fid', 'recon', 'seams', 'flicker'):
            results[part] = prev.get(part, {})
        if os.path.exists(out + '_samples.npz'):
            samples = dict(np.load(out + '_samples.npz'))
        print(f"resuming {out}.json: {len(results['fid'])} FIDs done")
    else:
        print(f'{out}.json was made with different FID settings: starting over (pass another --out to keep it)')
results['config'] = config

def save():
    with open(out + '.json', 'w') as fh:
        json.dump(results, fh, indent=1)
    np.savez(out + '_samples.npz', **samples)

# -----------------------------------------------------------------------------
# measurement helpers

def cell_err(pred, x):
    # summed squared difference (mean over channels) over every finest cell [B, G, G]
    return F.avg_pool2d((pred.float() - x.float()).pow(2).mean(1, keepdim=True), f_cell)[:, 0] * f_cell ** 2

def lowpass(x):
    # 8px block means, upsampled: the error splits exactly into this part + the part within blocks
    return F.interpolate(F.avg_pool2d(x.float(), 8), scale_factor=8, mode='nearest')

def node_sums(E, level):
    # sums of a cell map over every candidate patch at `level` [B, count], row-major (candidate order)
    k = G // geo.grids[level]
    return (F.avg_pool2d(E.unsqueeze(1), k)[:, 0] * k * k).flatten(1)

def split_gains(E):
    # E: per-level cell error maps [L x [B, G, G]] -> split gain of every splittable candidate [B, n_split]
    return torch.cat([node_sums(E[l], l) - node_sums(E[l + 1], l) for l in range(geo.L - 1)], dim=1)

def disagreement(preds):
    # where the finer model's prediction differs from the coarser one's, per candidate [B, n_split]
    return torch.cat([node_sums(cell_err(preds[l], preds[l + 1]), l) for l in range(geo.L - 1)], dim=1)

def spearman(a, b):
    # rank correlation along dim 1 (per image) [B]
    ra = a.float().argsort(1).argsort(1).float()
    rb = b.float().argsort(1).argsort(1).float()
    ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp_min(1e-12)

def per_level(s):
    return [s[:, geo.offsets[l]:geo.offsets[l] + geo.counts[l]] for l in range(geo.L - 1)]

def null_estimate(z, t):
    # the 16px model's null-label estimate: what dino / pixel route on, as QuadJiT training routes on its baseline
    with ctx:
        return nets[1](z, t, torch.full((z.shape[0],), num_classes, device=z.device, dtype=torch.long)).float()

@torch.no_grad()
def router_scores(name, z, t, preds=None, x_true=None):
    if name == 'uniform':
        return router.neutral.expand(z.shape[0], -1)
    if name == 'random':
        return router.random_scores(z)
    if name == 'disagree':
        return disagreement(preds)
    if name == 'loss': # expected gains, the 8px estimate standing in for the truth
        return split_gains([cell_err(p, preds[-1]) for p in preds])
    if name == 'oracle': # measured gains against the true image (reconstruction only)
        return split_gains([cell_err(p, x_true) for p in preds])
    est = null_estimate(z, t)
    with ctx:
        return router.dino_scores(est).float() if name == 'dino' else router.pixel_scores(est).float()

def maxplus(a, b):
    # max-plus convolution over the last dim: out[k] = max_j a[k - j] + b[j], and the maximizing j (b's share)
    la, lb = a.shape[-1], b.shape[-1]
    k = torch.arange(la + lb - 1, device=a.device)[:, None]
    j = torch.arange(lb, device=a.device)[None, :]
    i = k - j
    m = a[..., i.clamp(0, la - 1)] + b[..., None, :]
    m = m.masked_fill(~((i >= 0) & (i < la)), float('-inf'))
    return m.max(-1)

def allocate_exact(scores, K):
    """
    The layout maximizing the summed scores of the splits made, under the same rules as the greedy (a node splits only if
    its parent does; exactly m = (K - N0) / 3 splits). Exact for additive scores (the loss router's expected gains), where
    the greedy is myopic: it never splits a node with a small gain to reach large gains among its children.
    Dynamic program: f_node[s] = best total with s splits in the node's subtree (0 unsplit; else its own score + the best
    share of s - 1 among its children, combined by max-plus convolution), then the same across the N0 roots; the stored
    maximizers give the splits back.
    """
    B, dev = scores.shape[0], scores.device
    m = geo.check_budget(K)
    scores = scores.float()
    children = router.children
    f = torch.zeros(B, geo.counts[-1], 1, device=dev) # finest level: no splits
    kid_choices = {} # level -> list of argmax tensors combining children 0..c
    for l in range(geo.L - 2, -1, -1):
        sl = scores[:, geo.offsets[l]:geo.offsets[l] + geo.counts[l]]
        kids = children[geo.offsets[l]:geo.offsets[l] + geo.counts[l]] - geo.offsets[l + 1] # [n_l, 4] within level l + 1
        g = f[:, kids] # [B, n_l, 4, S + 1]
        acc, kid_choices[l] = g[:, :, 0], []
        for c in range(1, 4):
            acc, arg = maxplus(acc, g[:, :, c])
            kid_choices[l].append(arg)
        f = torch.cat([torch.zeros(B, geo.counts[l], 1, device=dev), sl[..., None] + acc], dim=-1)
    acc, root_choices = f[:, 0], []
    for r in range(1, geo.n0):
        acc, arg = maxplus(acc, f[:, r])
        root_choices.append(arg)
    # back out the splits per root, then per level
    rows = torch.arange(B, device=dev)
    k = torch.full((B,), m, dtype=torch.long, device=dev)
    s_nodes = torch.zeros(B, geo.n0, dtype=torch.long, device=dev)
    for r in range(geo.n0 - 1, 0, -1):
        s_nodes[:, r] = root_choices[r - 1][rows, k]
        k = k - s_nodes[:, r]
    s_nodes[:, 0] = k
    split = torch.zeros(B, geo.n, dtype=torch.bool, device=dev)
    for l in range(geo.L - 1):
        split[:, geo.offsets[l]:geo.offsets[l] + geo.counts[l]] = s_nodes > 0
        if l == geo.L - 2:
            break
        rem = (s_nodes - 1).clamp_min(0) # splits shared among the children
        nl = geo.counts[l]
        kid_s = torch.zeros(B, nl, 4, dtype=torch.long, device=dev)
        idx = torch.arange(nl, device=dev)[None, :]
        for c in range(3, 0, -1):
            kid_s[:, :, c] = kid_choices[l][c - 1][rows[:, None], idx, rem]
            rem = rem - kid_s[:, :, c]
        kid_s[:, :, 0] = rem
        kid_s = kid_s * (s_nodes > 0)[..., None]
        kids = children[geo.offsets[l]:geo.offsets[l] + nl] - geo.offsets[l + 1]
        s_next = torch.zeros(B, geo.counts[l + 1], dtype=torch.long, device=dev)
        s_next[:, kids.flatten()] = kid_s.flatten(1)
        s_nodes = s_next
    in_tree = torch.ones(B, geo.n, dtype=torch.bool, device=dev)
    in_tree[:, geo.n0:] = split[:, router.parent[geo.n0:]]
    leaves = in_tree & ~split
    return torch.arange(geo.n, device=dev).expand(B, -1)[leaves].view(B, K)

def layout_of(scores, K, exact=False):
    return allocate_exact(scores, K) if exact else allocate(scores, K, geo, router.children, router.parent)

def base_router(name):
    # 'loss_exact' -> ('loss', True)
    return (name[:-len('_exact')], True) if name.endswith('_exact') else (name, False)

def stitched_error(layout, E):
    # a layout's error: every finest cell's error from the model at its token's size [B]
    lev = cell_maps(layout, geo)[1]
    return torch.stack(E).gather(0, lev.unsqueeze(0))[0].flatten(1).sum(1)

def level_area(lev):
    # fraction of the image at each level [L] (a device tensor: no sync) from a cell level map [B, G, G]
    return torch.stack([(lev == l).float().mean() for l in range(geo.L)])

# -----------------------------------------------------------------------------
# stitched sampling

class StitchedNet(nn.Module):
    """
    The three uniform models as one network: a router's layout picks which model's prediction each region takes. Outside
    `window` (t) the 16px model alone; `render` maps a level to the level whose model renders its regions (attribution).
    mode 'X_exact': router X's scores, exact allocation. Under CFG, loss / disagree / oracle route each half on its own
    predictions. Logs the area per level and the flicker (fraction of the image whose patch size changed since the previous
    layout of the trajectory) per call as device tensors, read once by summary(); after a trajectory, cache holds its final
    layout. No host syncs per call (except to test t against a window), so the GPU never waits on Python between models.
    """
    def __init__(self, mode, K, window=(0.0, 1.0), render=None):
        super().__init__()
        self.mode, self.K, self.window, self.render = mode, K, window, render or {}
        self.base, self.exact = base_router(mode)
        self.x_true, self.cache, self.calls, self.prev = None, None, 0, None
        self.log = [] # (t, area [L], flicker or None) per call
        # the models this configuration can use, known up front (the levels a layout can contain, after render)
        levels = set(range(geo.L))
        if self.base == 'uniform': # one fixed layout
            levels = set(geo.level[allocate(router.neutral[None].cpu(), K, geo, geo.children, geo.parent)[0]].tolist())
        self.used = sorted({self.render.get(l, l) for l in levels})

    def begin_sampling(self): # called by the sampler at the start of every trajectory
        self.cache, self.calls, self.prev = None, 0, None

    def end_sampling(self):
        pass

    def summary(self):
        # {t: mean area per level}, {t: mean flicker} over every call logged so far; clears the log
        if not self.log:
            return {}, {}
        ts = torch.stack([l[0] for l in self.log]).float().cpu().numpy().round(4)
        areas = torch.stack([l[1] for l in self.log]).cpu().numpy()
        fl = [(t, f) for t, (_, _, f) in zip(ts, self.log) if f is not None]
        flv = torch.stack([f for _, f in fl]).cpu().numpy() if fl else []
        area_t = {f'{t:g}': areas[ts == t].mean(0).tolist() for t in np.unique(ts)}
        flick_t = {}
        for (t, _), v in zip(fl, flv):
            flick_t.setdefault(f'{t:g}', []).append(float(v))
        self.log = []
        return area_t, {k: float(np.mean(v)) for k, v in flick_t.items()}

    @torch.no_grad()
    def forward(self, z, t, y):
        B, t0 = z.shape[0], t.flatten()[0]
        if self.window != (0.0, 1.0) and not (self.window[0] <= t0.item() < self.window[1]):
            self.log.append((t0, torch.tensor([float(l == 1) for l in range(geo.L)], device=z.device), None))
            return nets[1](z, t, y)
        preds = [None] * geo.L
        if self.base in ('loss', 'disagree', 'oracle'):
            with ctx:
                preds = [net(z, t, y).float() for net in nets]
        flick = None
        if self.cache is None or self.cache.shape[0] != B or self.calls % route_every == 0:
            layout = layout_of(router_scores(self.base, z, t, preds=preds, x_true=self.x_true), self.K, self.exact)
            if self.prev is not None and self.prev.shape == layout.shape: # flicker: area re-tokenized since the last layout
                flick = 1 - layout_agreement(self.prev, layout, geo).mean()
            self.cache = self.prev = layout
        self.calls += 1
        lev = cell_maps(self.cache, geo)[1]
        for src, dst in self.render.items():
            lev = torch.where(lev == src, torch.full_like(lev, dst), lev)
        self.log.append((t0, level_area(lev), flick))
        lev_px = lev.repeat_interleave(f_cell, 1).repeat_interleave(f_cell, 2).unsqueeze(1)
        x_hat = torch.zeros(z.shape, device=z.device, dtype=torch.float32)
        for l in self.used:
            if preds[l] is None:
                with ctx:
                    preds[l] = nets[l](z, t, y).float()
            x_hat = torch.where(lev_px == l, preds[l], x_hat)
        return x_hat.to(z.dtype)

def with_net(net, fn):
    # run fn with the 16px denoiser's sampler around a stitched network
    den = models[1]
    orig = den.net
    den.net = net
    try:
        return fn(den)
    finally:
        den.net = orig

metrics = InceptionMetrics(device)
ref_stats = load_ref_stats(os.path.join(data_dir, 'fid_stats.npz'))
bank_path = os.path.join(data_dir, 'train_inception.npy')
pr_real = None
if os.path.exists(bank_path): # precision / recall against train features, as train_jit.py
    bank = np.load(bank_path, mmap_mode='r')
    pr_idx = np.sort(np.random.default_rng(0).choice(len(bank), min(num_images, len(bank)), replace=False))
    pr_real = torch.from_numpy(bank[pr_idx].astype(np.float32)).to(device)

# -----------------------------------------------------------------------------
# figures (redrawn after every new result, so a partial run has them too)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plot_style import SURFACE, INK, INK_2, CATEGORICAL, MARKERS, style_axis, line, heatmap_style

ORDER = ['uniform', 'random', 'pixel', 'dino', 'loss', 'disagree', 'dino_gate', 'dino_div', 'centre', 'loss_exact'] # fixed slots
COLORS = dict(zip(ORDER, CATEGORICAL[:6] + ['#f3c45e', '#a87200', INK_2, '#a8336b'])) # dino's terms: lighter / darker dino; exact: darker
color = lambda name: COLORS.get(name, INK_2)
marker = lambda name: MARKERS[ORDER.index(name) % len(MARKERS)] if name in ORDER else 'o'
fid_of = lambda k: results['fid'][k]['fid'] if k in results['fid'] else float('nan')
legend_levels = lambda fig, title: fig.legend(handles=[Patch(color=LEVEL_COLORS[l], label=f'{p}px') for l, p in enumerate(patches)],
                                              loc='upper right', ncol=geo.L, frameon=False, fontsize=8, title=title, title_fontsize=8)
to_np = lambda t: ((t.float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()

def blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)

def fig_fid():
    fig, ax = plt.subplots(figsize=(6.5, 4.4), facecolor=SURFACE)
    Ks_all = sorted(set(budgets) | set(LEVEL_TOKENS))
    for r in router_list:
        Ks = Ks_all if r == 'uniform' else budgets
        line(ax, Ks, [fid_of(stitched_key(r, K)) for K in Ks], color(r), marker(r), r)
    for l, ls in ((1, '--'), (2, ':')):
        ax.axhline(fid_of(f'uniform_{patches[l]}px'), color=INK_2, linewidth=1, linestyle=ls, label=f'{patches[l]}px model alone')
    ax.set_xscale('log', base=2)
    ax.set_xticks(Ks_all, [str(k) for k in Ks_all])
    ax.minorticks_off()
    style_axis(ax, 'tokens per image K', f'FID-{num_images / 1000:g}k', 'stitched sampling from the three uniform models')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out + '_fid.png', dpi=140, facecolor=SURFACE)
    plt.close(fig)

def fig_attribution():
    F16, F8 = fid_of(f'uniform_{patches[1]}px'), fid_of(f'uniform_{patches[2]}px')
    rows = ['full'] + [f't {lo:g}-{hi:g} only' for lo, hi in windows] + ['without splits', 'without merges']
    suffix = [''] + [f'_t{lo:g}-{hi:g}' for lo, hi in windows] + ['_nosplit', '_nomerge']
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(rows) * len(attr_list) + 1.6), facecolor=SURFACE)
    h = 0.8 / len(attr_list)
    for j, r in enumerate(attr_list):
        vals = [fid_of(stitched_key(r, attr_tokens) if s == '' else f'{r}_K{attr_tokens}{s}') - F16 for s in suffix]
        ax.barh(np.arange(len(rows)) + (j - (len(attr_list) - 1) / 2) * h, vals, height=h * 0.9, color=color(r), label=r)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.axvline(F8 - F16, color=INK_2, linewidth=1, linestyle=':', label=f'{patches[2]}px model alone')
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.invert_yaxis()
    style_axis(ax, f'FID-{num_images / 1000:g}k change vs the {patches[1]}px model (negative = better)', '',
               f'where the gain comes from, K={attr_tokens} ({patches[1]}px model outside the window / for the removed level)')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out + '_attribution.png', dpi=140, facecolor=SURFACE)
    plt.close(fig)

def fig_samples(K):
    # the same noise and labels through the 16px / 8px models and every router at K: each image clean, then with its
    # final layout drawn
    cols = list(dict.fromkeys([f'uniform_{patches[1]}px', f'uniform_{patches[2]}px'] + [stitched_key(r, K) for r in router_list]))
    cols = [c for c in cols if c + '__images' in samples]
    if len(cols) < 2:
        return
    nv = min(show_samples, len(samples[cols[0] + '__images']))
    fig, axes = plt.subplots(2 * nv, len(cols), figsize=(1.9 * len(cols), 1.95 * 2 * nv + 0.8), facecolor=SURFACE, squeeze=False)
    for j, c in enumerate(cols):
        ims = torch.from_numpy(samples[c + '__images'][:nv]).float().div(127.5).sub(1)
        lay = samples.get(c + '__final')
        drawn = draw_layout(ims, torch.from_numpy(lay[:nv]), geo) if lay is not None else ims
        for i in range(nv):
            blank(axes[2 * i, j]); blank(axes[2 * i + 1, j])
            axes[2 * i, j].imshow(to_np(ims[i]))
            axes[2 * i + 1, j].imshow(to_np(drawn[i]))
        title = c.replace(f'_K{K}', '') if c.endswith(f'_K{K}') else c.replace('uniform_', '') + ' alone'
        if c in results['fid']:
            title += f"\nFID {fid_of(c):.1f}"
        if c in results['seams']:
            title += f"\nseams x{seam_x(c):.2f}"
        axes[0, j].set_title(title, color=INK, fontsize=8)
    legend_levels(fig, 'final layout (finer side)')
    fig.suptitle(f'K={K}: same noise and labels; rows: sample, then its final layout\n'
                 f'seams = seam ratio / the {patches[1]}px placebo\'s (1 = none)', color=INK, fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / (1.95 * 2 * nv + 0.8)))
    fig.savefig(out + f'_samples_K{K}.png', dpi=140, facecolor=SURFACE)
    plt.close(fig)

seam_x = lambda k: results['seams'][k]['stitched'][0] / results['seams'][k]['placebo_16px'][0] if k in results['seams'] else float('nan')

def draw_progress():
    if f'uniform_{patches[1]}px' in results['fid']:
        fig_fid()
        if attr_list:
            fig_attribution()
    for K in sorted(set(budgets) | {attr_tokens}):
        fig_samples(K)

SAMPLE_LABELS = torch.from_numpy(np.linspace(0, num_classes - 1, n_samples).round().astype(np.int64)).to(device)

def sample_batch(net=None, level=None):
    # n_samples images (uint8) from the uniform model at `level` or a stitched net: the same seed and labels for every one
    def go(den):
        den.eval()
        den.cfg_scale, den.cfg_interval, den.steps, den.method = cfg, (0.0, 1.0), num_sampling_steps, sampling_method
        den.eta, den.fixed_shape = 1.0, False
        with torch.random.fork_rng(devices=[device] if 'cuda' in str(device) else []), torch.no_grad(), ctx:
            torch.manual_seed(seed)
            return den.generate(SAMPLE_LABELS)
    return to_uint8(go(models[level]) if net is None else with_net(net, go)).cpu()

def seams(key):
    """
    Seam ratio of a stitched configuration's samples (mean |neighbour difference| across token boundaries / inside tokens,
    over all boundaries and over boundaries between levels; diagnostics.seam_ratio) with their final layouts, next to the
    same layouts drawn on the 16px and 8px models' samples of the same noise (placebos: no stitching, so no seams).
    """
    a16, a8 = f'uniform_{patches[1]}px__images', f'uniform_{patches[2]}px__images'
    if key + '__final' not in samples or a16 not in samples or a8 not in samples:
        return
    lay = torch.from_numpy(samples[key + '__final']).to(device)
    ims = lambda k: torch.from_numpy(samples[k][:len(lay)]).to(device).float().div(127.5).sub(1)
    r = {name: seam_ratio(ims(k), lay, geo) for name, k in (('stitched', key + '__images'), ('placebo_16px', a16), ('placebo_8px', a8))}
    results['seams'][key] = r
    print(f"{key:<28} seams (across / inside tokens; between levels): {r['stitched'][0]:.3f}; {r['stitched'][1]:.3f} | "
          f"same layout on the {patches[1]}px model's samples {r['placebo_16px'][0]:.3f}; {r['placebo_16px'][1]:.3f}, "
          f"{patches[2]}px {r['placebo_8px'][0]:.3f}; {r['placebo_8px'][1]:.3f}")

def run(key, net=None, level=None):
    """
    One configuration (a stitched net, or the uniform model at `level`) stored under key: its samples and seams, then its
    FID-{num_images} (with --fid). Finished parts are skipped (resume); the figures are redrawn after every new result.
    """
    stitched = net is not None
    todo = [w for w, need in (('samples', key + '__images' not in samples or len(samples[key + '__images']) < n_samples
                              or (stitched and (key + '__final' not in samples or key not in results['flicker']))),
                              (f'FID-{num_images / 1000:g}k', fid and key not in results['fid'])) if need]
    if todo:
        print(f"[{key}] {' + '.join(todo)}")
    if key + '__images' not in samples or len(samples[key + '__images']) < n_samples or \
            (stitched and (key + '__final' not in samples or key not in results['flicker'])):
        t_start = time.time()
        samples[key + '__images'] = sample_batch(net, level).numpy()
        print(f"[{key}] {n_samples} samples in {time.time() - t_start:.0f}s")
        if stitched:
            # a t window the trajectory never reached leaves the 16px model's layout
            final = net.cache if net.cache is not None else layout_of(router.neutral.expand(n_samples, -1), geo.counts[1])
            samples[key + '__final'] = final[:n_samples].cpu().numpy()
            by_t = net.summary()[1]
            results['flicker'][key] = {'mean': float(np.mean(list(by_t.values()))) if by_t else 0.0, 'by_t': by_t}
            print(f"{key:<28} flicker (image re-tokenized per re-route): {results['flicker'][key]['mean']:.1%}")
        seams(key)
        save(); draw_progress()
    elif stitched and key not in results['seams']:
        seams(key)
    if not fid or key in results['fid']:
        return results['fid'].get(key)
    kw = dict(cfg=cfg, interval=(0.0, 1.0), steps=num_sampling_steps, method=sampling_method, batch_size=gen_batch_size, seed=seed)
    t_start = time.time()
    if net is None:
        o = generate_and_score(models[level], metrics, ref_stats, num_images, num_classes, device, ctx, **kw)
        area, area_t = [float(l == level) for l in range(geo.L)], {}
    else:
        o = with_net(net, lambda den: generate_and_score(den, metrics, ref_stats, num_images, num_classes, device, ctx, **kw))
        area_t = net.summary()[0]
        area = np.mean(list(area_t.values()), axis=0).tolist()
    r = {'fid': o['fid'], 'is': o['is'], 'area': area, 'area_by_t': area_t}
    if pr_real is not None:
        r['precision'], r['recall'] = precision_recall(pr_real, o['features'].to(device))
    results['fid'][key] = r
    save(); draw_progress()
    pr = f", P {r['precision']:.3f}, R {r['recall']:.3f}" if 'precision' in r else ''
    print(f"{key:<28} FID-{num_images / 1000:g}k {r['fid']:7.2f}, IS {r['is']:6.1f}{pr} | {(time.time() - t_start) / 60:.1f} min | area "
          + ' '.join(f'{p}px {a:.0%}' for p, a in zip(patches, area)))
    return r

def stitched_key(r, K):
    # a uniform layout of one level is that level's model alone
    return f'uniform_{patches[LEVEL_TOKENS[K]]}px' if r == 'uniform' and K in LEVEL_TOKENS else f'{r}_K{K}'

print(f"models: {', '.join(f'{p}px' for p in patches)} at {img}px | FID-{num_images / 1000:g}k, {sampling_method} "
      f"{num_sampling_steps} steps, cfg {cfg} | routers {router_list} at K {budgets} | attribution {attr_list} at K={attr_tokens}")
for l in range(geo.L):
    run(f'uniform_{patches[l]}px', level=l)
for K in budgets:
    for r in router_list:
        if not (r == 'uniform' and K in LEVEL_TOKENS): # that level's model alone: an anchor
            run(stitched_key(r, K), StitchedNet(r, K))
for r in attr_list:
    if not (r == 'uniform' and attr_tokens in LEVEL_TOKENS):
        run(stitched_key(r, attr_tokens), StitchedNet(r, attr_tokens))
    for lo, hi in windows:
        run(f'{r}_K{attr_tokens}_t{lo:g}-{hi:g}', StitchedNet(r, attr_tokens, window=(lo, hi)))
    run(f'{r}_K{attr_tokens}_nosplit', StitchedNet(r, attr_tokens, render={geo.L - 1: 1}))
    run(f'{r}_K{attr_tokens}_nomerge', StitchedNet(r, attr_tokens, render={0: 1}))

if f'uniform_{patches[1]}px' in results['fid'] and f'uniform_{patches[2]}px' in results['fid']:
    F16, F8 = fid_of(f'uniform_{patches[1]}px'), fid_of(f'uniform_{patches[2]}px')
    gap = lambda f: (F16 - f) / (F16 - F8) if F16 != F8 else float('nan')
    print(f"\nFID-{num_images / 1000:g}k vs the {patches[1]}px model ({F16:.2f}); gap closed = share of the {patches[1]}px -> "
          f"{patches[2]}px improvement ({F16:.2f} -> {F8:.2f}) reached; seams = seam ratio / the {patches[1]}px placebo's (1 = none)")
    print(f"  {'configuration':<28} {'FID':>7} {'vs 16px':>8} {'gap':>6} {'seams':>6} {'flicker':>8}")
    for k, v in results['fid'].items():
        fl = results['flicker'][k]['mean'] if k in results['flicker'] else float('nan')
        print(f"  {k:<28} {v['fid']:7.2f} {v['fid'] - F16:+8.2f} {gap(v['fid']):6.0%} {seam_x(k):6.2f} {fl:8.1%}")

# -----------------------------------------------------------------------------
# regions: which regions benefit, and how each heuristic correlates with that

HEURISTICS = ['loss', 'dino', 'dino_gate', 'dino_div', 'pixel', 'disagree', 'centre', 'random']
viz = {}
if regions:
    rng_idx = np.sort(np.random.default_rng(seed).choice(len(data), n_images, replace=False))
    images, labels = data.get(rng_idx)
    X, Y = to_model_input(images, flip=False, device=device), torch.from_numpy(labels).to(device)
    names = [names_all[i] for i in labels.tolist()]
    NOISE = torch.randn(X.shape, generator=torch.Generator(device=device).manual_seed(seed), device=device) * noise_scale

    @torch.no_grad()
    def region_pass(x, y, e, t_val):
        # per-level predictions and every heuristic's scores [B, n_split] for one batch noised to t
        t = torch.full((len(x),), t_val, device=device)
        z = t_val * x + (1 - t_val) * e
        est = null_estimate(z, t)
        with ctx:
            preds = [net(z, t, y).float() for net in nets]
            dino_s, parts = router.dino_scores(est, parts=True)
        h = {'loss': split_gains([cell_err(p, preds[-1]) for p in preds]), 'dino': dino_s.float(), 'dino_gate': parts['gate'].float(), 'dino_div': parts['div'].float(),
             'pixel': router.pixel_scores(est), 'disagree': disagreement(preds), 'centre': router.neutral.expand(len(x), -1),
             'random': router.random_scores(z)}
        return preds, h

    for t_val in [float(v) for v in ts.split(',')]:
        tot = {b: [0.0] * (geo.L - 1) for b in ('total', 'structure', 'detail')}
        corr = {h: [[] for _ in range(geo.L - 1)] for h in HEURISTICS}
        names_x = ['gain'] + HEURISTICS
        cross = {(a, b): [] for a in names_x for b in names_x}
        alloc = {(name, K): [] for name in ('oracle', 'oracle_exact', 'loss', 'loss_exact') for K in budgets}
        for i in range(0, n_images, batch_size):
            x, y, e = X[i:i + batch_size], Y[i:i + batch_size], NOISE[i:i + batch_size]
            preds, h = region_pass(x, y, e, t_val)
            gains = split_gains([cell_err(p, x) for p in preds])
            bands = {'total': gains, 'structure': split_gains([cell_err(lowpass(p), lowpass(x)) for p in preds]),
                     'detail': split_gains([cell_err(p - lowpass(p), x - lowpass(x)) for p in preds])}
            for b, s in bands.items():
                for l, sl in enumerate(per_level(s)):
                    tot[b][l] += sl.sum().item()
            for name, s in h.items():
                for l, (sl, gl) in enumerate(zip(per_level(s), per_level(gains))):
                    corr[name][l].append(spearman(sl, gl).cpu())
            E = [cell_err(p, x) for p in preds]
            for K in budgets: # error removed vs the coarse-first uniform layout at the same K (measured against x)
                e_uni = stitched_error(layout_of(router.neutral.expand(len(x), -1), K), E)
                for name in ('oracle', 'oracle_exact', 'loss', 'loss_exact'):
                    b, ex = base_router(name)
                    alloc[(name, K)].append((e_uni - stitched_error(layout_of(gains if b == 'oracle' else h['loss'], K, ex), E)).cpu())
            allv = dict(h, gain=gains)
            for (a, b) in cross:
                cross[(a, b)].append(torch.stack([spearman(sa, sb) for sa, sb in zip(per_level(allv[a]), per_level(allv[b]))]).mean(0).cpu())
            if abs(t_val - viz_t) < 1e-6 and i == 0:
                n = min(viz_images, len(x))
                viz['gains'] = gains[:n].cpu()
                viz['loss-oracle'] = layout_of(gains, attr_tokens)[:n].cpu()
                for r in router_list:
                    b, ex = base_router(r)
                    viz[r] = layout_of(router.neutral.expand(len(x), -1) if b == 'uniform' else h[b], attr_tokens, ex)[:n].cpu()
        per_patch = lambda v, l: v / (n_images * geo.counts[l])
        results['regions'][f'{t_val:g}'] = {
            'gain_per_patch': {f'{patches[l]}->{patches[l + 1]}px': {b: per_patch(tot[b][l], l) for b in tot} for l in range(geo.L - 1)},
            'loss_corr': {name: [torch.cat(c).nanmean().item() for c in corr[name]] for name in HEURISTICS},
            'cross_corr': {f'{a}|{b}': torch.cat(v).nanmean().item() for (a, b), v in cross.items()},
            'allocation_gain_vs_uniform': {f'{name}_K{K}': torch.cat(v).mean().item() for (name, K), v in alloc.items()}}

    # Inception-feature split gains on a subset
    for t_val in [float(v) for v in feature_ts.split(',')]:
        n = min(feature_images, n_images)
        fg, lg, hs = [], [], {name: [] for name in HEURISTICS}
        for j in range(n):
            x, y, e = X[j:j + 1], Y[j:j + 1], NOISE[j:j + 1]
            preds, h = region_pass(x, y, e, t_val)
            real_f = metrics.features(to_uint8(x))[0]
            g_j = []
            for l in range(geo.L - 1):
                p, gl = geo.patches[l], geo.grids[l]
                variants = [preds[l][0]]
                for c in range(geo.counts[l]):
                    r0, c0 = divmod(c, gl)
                    v = preds[l][0].clone()
                    v[:, r0 * p:(r0 + 1) * p, c0 * p:(c0 + 1) * p] = preds[l + 1][0, :, r0 * p:(r0 + 1) * p, c0 * p:(c0 + 1) * p]
                    variants.append(v)
                d = (metrics.features(to_uint8(torch.stack(variants)))[0] - real_f).norm(dim=1)
                g_j.append(d[0] - d[1:])
            fg.append(torch.cat(g_j))
            lg.append(split_gains([cell_err(pr, x) for pr in preds])[0].cpu())
            for name in HEURISTICS:
                hs[name].append(h[name][0].cpu())
        fg, lg = torch.stack(fg), torch.stack(lg)
        row = {'loss_vs_feature': [spearman(a, b).nanmean().item() for a, b in zip(per_level(lg), per_level(fg))],
               'feature_corr': {name: [spearman(a, b).nanmean().item() for a, b in zip(per_level(torch.stack(hs[name])), per_level(fg))] for name in HEURISTICS},
               'oracle_agreement': layout_agreement(layout_of(lg.to(device), attr_tokens), layout_of(fg.to(device), attr_tokens), geo).mean().item()}
        results['regions'].setdefault(f'{t_val:g}', {})['features'] = row
        if abs(t_val - viz_t) < 1e-6:
            viz['feature-oracle'] = layout_of(fg.to(device), attr_tokens)[:viz_images].cpu()
    save()

    print('\nsplit gains (error the finer model removes), mean per patch = structure (8px block means) + detail:')
    for tk, rr in results['regions'].items():
        if 'gain_per_patch' in rr:
            print(f'  t={tk:<5} ' + '   '.join(f"{k}: {v['total']:+.3g} = {v['structure']:+.3g} + {v['detail']:+.3g}" for k, v in rr['gain_per_patch'].items()))
    print('\nmeasured error removed vs the uniform layout at the same K (per image): greedy / exact allocation, '
          'on the measured gains (oracle) and on the expected ones (loss)')
    for tk, rr in results['regions'].items():
        if 'allocation_gain_vs_uniform' in rr:
            a = rr['allocation_gain_vs_uniform']
            print(f'  t={tk:<5} ' + '   '.join(f"K={K}: oracle {a[f'oracle_K{K}']:+.3g} / {a[f'oracle_exact_K{K}']:+.3g}, "
                                            f"loss {a[f'loss_K{K}']:+.3g} / {a[f'loss_exact_K{K}']:+.3g}" for K in budgets))
    print(f'\nrank correlation with the split gain per region, {patches[0]}px | {patches[1]}px parents (feat: Inception-feature gains):')
    print('  ' + ' ' * 7 + ''.join(f'{name:>14}' for name in HEURISTICS))
    for tk, rr in results['regions'].items():
        if 'loss_corr' in rr:
            print(f'  t={tk:<5}' + ''.join(f"{rr['loss_corr'][name][0]:>+7.2f}|{rr['loss_corr'][name][1]:<+6.2f}" for name in HEURISTICS))
        if 'features' in rr:
            fr = rr['features']
            print(f'  {"  feat":<7}' + ''.join(f"{fr['feature_corr'][name][0]:>+7.2f}|{fr['feature_corr'][name][1]:<+6.2f}" for name in HEURISTICS)
                  + f"   loss vs feature gains {fr['loss_vs_feature'][0]:+.2f}|{fr['loss_vs_feature'][1]:+.2f}, oracle layouts agree {fr['oracle_agreement']:.0%}")

# -----------------------------------------------------------------------------
# reconstruction FID from t0

if recon:
    ridx = np.sort(np.random.default_rng(seed + 1).choice(len(data), min(recon_num_images, len(data)), replace=False))

    def recon_fid(net, t0):
        # FID of the held-out images noised to t0 and sampled back with net (None: the clean images themselves)
        feats, logits = [], []
        for i in range(0, len(ridx), gen_batch_size):
            xb, yb = data.get(ridx[i:i + gen_batch_size])
            xb, yb = to_model_input(xb, flip=False, device=device), torch.from_numpy(yb).to(device)
            if net is None:
                xr = xb
            else:
                e = torch.randn(xb.shape, generator=torch.Generator(device=device).manual_seed(seed + i), device=device)
                z0 = t0 * xb + (1 - t0) * e * noise_scale
                net.x_true = xb
                def go(den):
                    den.cfg_scale, den.cfg_interval, den.steps, den.method = 1.0, (0.0, 1.0), num_sampling_steps, sampling_method
                    den.eta, den.fixed_shape = 1.0, False
                    with torch.no_grad(), ctx:
                        return den.sample_from(z0, t0, yb)
                xr = with_net(net, go)
            f_, l_ = metrics.features(to_uint8(xr))
            feats.append(f_); logits.append(l_)
        return metrics.compute(torch.cat(feats), torch.cat(logits), ref_stats)[0]

    if 'clean_images' not in results['recon']:
        results['recon']['clean_images'] = recon_fid(None, None); save()
    print(f"\nreconstruction FID-{len(ridx) / 1000:g}k of the clean held-out images themselves: {results['recon']['clean_images']:.2f}")
    for t0 in [float(v) for v in recon_t0.split(',')]:
        for r in recon_routers.split(','):
            key = f't0{t0:g}_{r}_K{attr_tokens}'
            if key not in results['recon']:
                results['recon'][key] = recon_fid(StitchedNet(r, attr_tokens), t0); save()
            print(f'reconstruction from t0={t0:g}, {r} K={attr_tokens}: FID {results["recon"][key]:.2f}')

print(f'\nwrote {out}.json, {out}_samples.npz')
draw_progress()

if regions:
    # heuristics vs gains, and with each other
    reg = results['regions']
    t_loss = [float(k) for k, v in reg.items() if 'loss_corr' in v]
    t_feat = [float(k) for k, v in reg.items() if 'features' in v]
    fig = plt.figure(figsize=(16, 4.6), facecolor=SURFACE)
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
    for l in range(geo.L - 1):
        ax = axes[l]
        for name in HEURISTICS:
            line(ax, t_loss, [reg[f'{t:g}']['loss_corr'][name][l] for t in t_loss], color(name), marker(name), name)
            ax.plot(t_feat, [reg[f'{t:g}']['features']['feature_corr'][name][l] for t in t_feat], marker(name), color=color(name),
                    markerfacecolor='none', markersize=9, linestyle='none')
        ax.axhline(0, color=INK_2, linewidth=1, linestyle=':')
        style_axis(ax, 't (1 = clean)', 'rank correlation with the split gain',
                   f'{patches[l]}px -> {patches[l + 1]}px (hollow: feature-space gains)')
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK_2)
    tk = f'{viz_t:g}'
    if 'cross_corr' in reg.get(tk, {}):
        lab = ['gain'] + HEURISTICS
        M = np.array([[reg[tk]['cross_corr'][f'{a}|{b}'] for b in lab] for a in lab])
        im = axes[2].imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
        heatmap_style(axes[2], im, fig, lab, lab, f'rank correlations at t={viz_t:g} (mean over levels)', 'Spearman')
        for i in range(len(lab)):
            for j in range(len(lab)):
                axes[2].text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6, color=INK if abs(M[i, j]) < 0.6 else SURFACE)
    else:
        blank(axes[2])
    fig.tight_layout()
    fig.savefig(out + '_heuristics.png', dpi=140, facecolor=SURFACE)
    plt.close(fig)

    # gain maps and layouts
    if 'gains' in viz:
        nv = len(viz['gains'])
        cols = ['image', f'gain {patches[0]}->{patches[1]}px', f'gain {patches[1]}->{patches[2]}px'] \
            + [c for c in ['loss-oracle', 'feature-oracle'] + router_list if c in viz]
        fig, axes = plt.subplots(nv, len(cols), figsize=(1.7 * len(cols), 1.8 * nv + 0.6), facecolor=SURFACE, squeeze=False)
        gmax = [per_level(viz['gains'])[l].abs().max().item() or 1.0 for l in range(geo.L - 1)]
        for i in range(nv):
            for j, c in enumerate(cols):
                ax = axes[i, j]
                blank(ax)
                if c == 'image':
                    ax.imshow(to_np(X[i]))
                    ax.set_ylabel(names[i][:16], color=INK_2, fontsize=7)
                elif c.startswith('gain'):
                    l = j - 1
                    ax.imshow(per_level(viz['gains'])[l][i].view(geo.grids[l], geo.grids[l]).numpy(), cmap='RdBu_r',
                              vmin=-gmax[l], vmax=gmax[l], interpolation='nearest')
                elif i < len(viz[c]):
                    ax.imshow(to_np(draw_layout(X[i:i + 1].cpu(), viz[c][i:i + 1], geo)[0]))
                if i == 0:
                    ax.set_title(c, color=INK, fontsize=8)
        legend_levels(fig, 'token boundaries (finer side)')
        fig.suptitle(f'split gains per region (red: the finer model helps) and layouts at t={viz_t:g}, K={attr_tokens}',
                     color=INK, fontsize=10, x=0.01, ha='left')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out + '_gains.png', dpi=140, facecolor=SURFACE)
        plt.close(fig)

print(f"wrote {', '.join(out + s for s in ['_fid.png', '_attribution.png'] + [f'_samples_K{K}.png' for K in sorted(set(budgets) | {attr_tokens})] + (['_heuristics.png', '_gains.png'] if regions else []))}")
