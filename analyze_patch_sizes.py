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
               finer model. Plus the heuristics' correlations with each other.
  recon        (--recon, opt-in) Reconstruction FID: held-out images noised to t0 and sampled from there with stitched
               layouts, including the loss-oracle (layout re-chosen every step from the measured gains against the true
               image). A router beating the loss-oracle on FID means loss is the wrong target for splitting.

Caveats.
  * The three models differ in compute per pixel and bottleneck compression as well as patch size; a budget-K QuadJiT
    is one network, so this is a guide to the gains, not a prediction of a trained model.
  * Stitched sampling can show seams between the models' predictions, which a single network wouldn't have.
  * A stitched configuration runs the models its layout needs (all three: ~1.3x the 8px model's cost), plus the 16px
    routing estimate (dino / pixel: ~0.25x) and DINOv2 at 224px (dino: ~1x). --route_every reuses a layout for that many
    network calls of a trajectory.
  * Results are written after every generation; rerunning with the same --out and FID settings skips finished ones.

Outputs: {out}.json (every number), {out}_samples.npz (the first images of every configuration, with layouts), and
  {out}_fid.png          FID vs K per router, with the anchors
  {out}_attribution.png  FID change vs the 16px model: full stitching, per t window, without splits / merges
  {out}_samples.png      the same noise through the 16px / 8px models and every router at --attr_tokens, layouts drawn
  {out}_heuristics.png   rank correlation of each heuristic with the loss / feature split gains vs t, per level, and
                         the heuristics' correlations with each other at --viz_t
  {out}_gains.png        gain maps per region with the loss-oracle's, feature-oracle's and routers' layouts at --viz_t
"""

import os
import json
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_jit import MemmapImages, to_model_input
from jit_common import load_denoiser, generate_and_score
from fid import InceptionMetrics, load_ref_stats, to_uint8, precision_recall
from model_quadtree import QuadGeometry, QuadRouter, allocate, cell_maps, layout_agreement, draw_layout, LEVEL_COLORS

# -----------------------------------------------------------------------------
ckpts = '' # comma list of three plain-JiT checkpoints (32 / 16 / 8 px patches, any order), trained alike to the same step
data_dir = '' # the prepared dataset they trained on (fid_stats.npz, train_inception.npy, val/)
ema = 1
seed = 0
# stitched sampling FID
num_images = 10000
tokens = '64,100,160' # budgets K (16 + 3m at 128px)
routers = 'loss,dino,pixel,random,uniform' # + disagree
attr_routers = 'loss,dino'
attr_tokens = 64
attr_t = '0,0.5,0.8,1' # t window edges for the by-t attribution
route_every = 1 # reuse a layout for this many network calls of a trajectory (1 = re-route every call)
cfg = 1.0
num_sampling_steps = 50
sampling_method = 'heun'
gen_batch_size = 128
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
out = out or f"{os.path.basename(os.path.normpath(data_dir))}_patch_sizes"
ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if 'cuda' in device else nullcontext()

# models: coarsest first
loaded = sorted((load_denoiser(p, ema, device) for p in ckpts.split(',')), key=lambda dm: -dm[0].net.patch_size)
assert len(loaded) == 3, 'pass exactly three checkpoints (32 / 16 / 8 px)'
models, steps = [d for d, _ in loaded], [m.get('iter_num') for _, m in loaded]
print('checkpoints at iteration ' + ', '.join(f'{d.net.patch_size}px {s}' for d, s in zip(models, steps)))
if len(set(steps)) > 1:
    print('WARNING: the checkpoints are at different iterations, so the comparison mixes patch size with training length')
nets = [d.net for d in models]
patches = [n.patch_size for n in nets]
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
            'gate_tau', 'viz_t', 'viz_images', 'recon_num_images')
results = {'patches': patches, 'fid': {}, 'regions': {}, 'recon': {}}
samples = {}
if os.path.exists(out + '.json'):
    with open(out + '.json') as fh:
        prev = json.load(fh)
    if all(prev.get('config', {}).get(k) == config[k] for k in FID_KEYS):
        results['fid'], results['recon'] = prev.get('fid', {}), prev.get('recon', {})
        if os.path.exists(out + '_samples.npz'):
            samples = dict(np.load(out + '_samples.npz'))
        print(f"resuming {out}.json: {len(results['fid'])} generations done")
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

def layout_of(scores, K):
    return allocate(scores, K, geo, router.children, router.parent)

def level_area(lev):
    # fraction of the image at each level [L] from a cell level map [B, G, G]
    return [(lev == l).float().mean().item() for l in range(geo.L)]

# -----------------------------------------------------------------------------
# stitched sampling

class StitchedNet(nn.Module):
    """
    The three uniform models as one network: a router's layout picks which model's prediction each region takes. Only the
    models a layout uses are run (loss / disagree / oracle need all three). Outside `window` (t) the 16px model alone; `render`
    maps a level to the level whose model renders its regions (attribution). Records the area per level per t and the
    first batch's layout at viz_t. Under CFG, loss / disagree / oracle route each half on its own predictions.
    """
    def __init__(self, mode, K, window=(0.0, 1.0), render=None):
        super().__init__()
        self.mode, self.K, self.window, self.render = mode, K, window, render or {}
        self.x_true, self.cache, self.calls = None, None, 0
        self.area, self.viz_layout = {}, None

    def begin_sampling(self): # called by the sampler at the start of every trajectory
        self.cache, self.calls = None, 0

    def end_sampling(self):
        pass

    @torch.no_grad()
    def forward(self, z, t, y):
        B, tv = z.shape[0], t.flatten()[0].item()
        if not (self.window[0] <= tv < self.window[1]):
            self.area.setdefault(round(tv, 4), []).append([float(l == 1) for l in range(geo.L)])
            return nets[1](z, t, y)
        preds = [None] * geo.L
        if self.mode in ('loss', 'disagree', 'oracle'):
            with ctx:
                preds = [net(z, t, y).float() for net in nets]
        if self.cache is None or self.cache.shape[0] != B or self.calls % route_every == 0:
            self.cache = layout_of(router_scores(self.mode, z, t, preds=preds, x_true=self.x_true), self.K)
        self.calls += 1
        lev = cell_maps(self.cache, geo)[1]
        if self.viz_layout is None and tv >= viz_t:
            self.viz_layout = self.cache[:viz_images].cpu()
        for src, dst in self.render.items():
            lev = torch.where(lev == src, torch.full_like(lev, dst), lev)
        self.area.setdefault(round(tv, 4), []).append(level_area(lev))
        lev_px = lev.repeat_interleave(f_cell, 1).repeat_interleave(f_cell, 2).unsqueeze(1)
        x_hat = torch.zeros(z.shape, device=z.device, dtype=torch.float32)
        for l in range(geo.L):
            m = lev_px == l
            if not m.any():
                continue
            if preds[l] is None:
                with ctx:
                    preds[l] = nets[l](z, t, y).float()
            x_hat = torch.where(m, preds[l], x_hat)
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

def run(key, net=None, level=None):
    """
    FID-{num_images} of one configuration (a stitched net, or the uniform model at `level`), stored under key; skipped
    if already done (resume).
    """
    if key in results['fid']:
        return results['fid'][key]
    first = []
    keep = lambda x, labels: first.append(x[:viz_images].cpu()) if not first else None
    kw = dict(cfg=cfg, interval=(0.0, 1.0), steps=num_sampling_steps, method=sampling_method, batch_size=gen_batch_size,
              seed=seed, on_batch=keep)
    if net is None:
        o = generate_and_score(models[level], metrics, ref_stats, num_images, num_classes, device, ctx, **kw)
        area, area_t = [float(l == level) for l in range(geo.L)], {}
    else:
        o = with_net(net, lambda den: generate_and_score(den, metrics, ref_stats, num_images, num_classes, device, ctx, **kw))
        area_t = {f'{tv:g}': np.mean(v, axis=0).tolist() for tv, v in sorted(net.area.items())}
        area = np.mean(list(area_t.values()), axis=0).tolist()
        if net.viz_layout is not None:
            samples[key + '__layout'] = net.viz_layout.numpy()
    r = {'fid': o['fid'], 'is': o['is'], 'area': area, 'area_by_t': area_t}
    if pr_real is not None:
        r['precision'], r['recall'] = precision_recall(pr_real, o['features'].to(device))
    results['fid'][key] = r
    samples[key + '__images'] = first[0].numpy()
    save()
    pr = f", P {r['precision']:.3f}, R {r['recall']:.3f}" if 'precision' in r else ''
    print(f"{key:<28} FID-{num_images / 1000:g}k {r['fid']:7.2f}, IS {r['is']:6.1f}{pr} | area "
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
        k = stitched_key(r, K)
        if k not in results['fid']:
            run(k, StitchedNet(r, K))
for r in attr_list:
    run(stitched_key(r, attr_tokens), StitchedNet(r, attr_tokens))
    for lo, hi in windows:
        run(f'{r}_K{attr_tokens}_t{lo:g}-{hi:g}', StitchedNet(r, attr_tokens, window=(lo, hi)))
    run(f'{r}_K{attr_tokens}_nosplit', StitchedNet(r, attr_tokens, render={geo.L - 1: 1}))
    run(f'{r}_K{attr_tokens}_nomerge', StitchedNet(r, attr_tokens, render={0: 1}))

F16, F8 = results['fid'][f'uniform_{patches[1]}px']['fid'], results['fid'][f'uniform_{patches[2]}px']['fid']
gap = lambda f: (F16 - f) / (F16 - F8) if F16 != F8 else float('nan')
print(f"\nFID-{num_images / 1000:g}k vs the {patches[1]}px model ({F16:.2f}); gap closed = share of the {patches[1]}px -> "
      f"{patches[2]}px improvement ({F16:.2f} -> {F8:.2f}) reached")
print(f"  {'configuration':<28} {'FID':>7} {'vs 16px':>8} {'gap':>6}")
for k, v in results['fid'].items():
    print(f"  {k:<28} {v['fid']:7.2f} {v['fid'] - F16:+8.2f} {gap(v['fid']):6.0%}")

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
            allv = dict(h, gain=gains)
            for (a, b) in cross:
                cross[(a, b)].append(torch.stack([spearman(sa, sb) for sa, sb in zip(per_level(allv[a]), per_level(allv[b]))]).mean(0).cpu())
            if abs(t_val - viz_t) < 1e-6 and i == 0:
                n = min(viz_images, len(x))
                viz['gains'] = gains[:n].cpu()
                viz['loss-oracle'] = layout_of(gains, attr_tokens)[:n].cpu()
                for r in router_list:
                    viz[r] = layout_of(router.neutral.expand(len(x), -1) if r == 'uniform' else h[r], attr_tokens)[:n].cpu()
        per_patch = lambda v, l: v / (n_images * geo.counts[l])
        results['regions'][f'{t_val:g}'] = {
            'gain_per_patch': {f'{patches[l]}->{patches[l + 1]}px': {b: per_patch(tot[b][l], l) for b in tot} for l in range(geo.L - 1)},
            'loss_corr': {name: [torch.cat(c).nanmean().item() for c in corr[name]] for name in HEURISTICS},
            'cross_corr': {f'{a}|{b}': torch.cat(v).nanmean().item() for (a, b), v in cross.items()}}

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

# -----------------------------------------------------------------------------
# figures

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plot_style import SURFACE, INK, INK_2, CATEGORICAL, MARKERS, style_axis, line, heatmap_style

ORDER = ['uniform', 'random', 'pixel', 'dino', 'loss', 'disagree', 'dino_gate', 'dino_div', 'centre'] # fixed slots
COLORS = dict(zip(ORDER, CATEGORICAL[:6] + ['#f3c45e', '#a87200', INK_2])) # dino's terms: lighter / darker dino
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

# FID vs K
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

# attribution
if attr_list:
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

# samples: the same noise through the anchors and every router at attr_tokens
cols = list(dict.fromkeys([f'uniform_{patches[1]}px', f'uniform_{patches[2]}px'] + [stitched_key(r, attr_tokens) for r in router_list]))
cols = [c for c in cols if c + '__images' in samples]
if cols:
    nv = min(len(samples[cols[0] + '__images']), viz_images)
    fig, axes = plt.subplots(nv, len(cols), figsize=(1.7 * len(cols), 1.8 * nv + 0.8), facecolor=SURFACE, squeeze=False)
    for j, c in enumerate(cols):
        ims = torch.from_numpy(samples[c + '__images'][:nv]).float().div(127.5).sub(1)
        if c + '__layout' in samples:
            ims = draw_layout(ims, torch.from_numpy(samples[c + '__layout'][:nv]), geo)
        for i in range(nv):
            blank(axes[i, j])
            axes[i, j].imshow(to_np(ims[i]))
        axes[0, j].set_title(f"{c.replace('uniform_', '')}\nFID {fid_of(c):.1f}", color=INK, fontsize=8)
    legend_levels(fig, f'layout at t={viz_t:g} (finer side)')
    fig.suptitle('same noise and labels through each configuration', color=INK, fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out + '_samples.png', dpi=140, facecolor=SURFACE)
    plt.close(fig)

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

print(f"wrote {', '.join(out + s for s in ['_fid.png', '_attribution.png', '_samples.png'] + (['_heuristics.png', '_gains.png'] if regions else []))}")
