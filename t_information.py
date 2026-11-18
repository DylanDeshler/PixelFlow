"""
Semantic information gain over noise levels: at which t does a noisy image z_t = t x + (1 - t) s e reveal the class, its
coarse category, and each patch's local content? A candidate training t distribution is then "sample t in proportion to
the information being revealed there". Inference only, on a trained JiT checkpoint.

For a variable Y (class, a coarse level of it, or a patch's content code) the information the noisy image carries is
  I(Y; z_t) >= H(Y) - CE_t,   CE_t = cross-entropy of a classifier q applied to the model's prediction x_hat(z_t)
(any q gives this lower bound; the better q, the tighter). x_hat comes from the model with the null label (the class
label would otherwise leak in), and q reads it with a frozen DINOv2: the prediction is an image at every t, so no decoder
is needed and image-space readers apply directly (no attention confound of hidden states).
  class       linear head on DINOv2's CLS feature, fit on clean training images; temperature refit per t (one scalar)
  coarse      levels '--levels' of a hierarchy built from the data: k-means of the class-mean DINOv2 features, each class in
              one cluster (so I(class) = I(coarse) + I(class | coarse): the coarse curve is part of the class curve)
  patch       each DINOv2 patch of the clean image gets one of --patch_codes content codes (k-means of clean patch
              features); q = softmax(kappa * cosine to the codes) of the same patch of x_hat, kappa refit per t
I_Y(t) is made non-decreasing in t (the true one is) and normalized by its value at the cleanest measured t (what the
model's prediction reveals at most; the reader on the clean image is reported alongside but not used: the gap is the
prediction's floor near t = 1, not information revealed at t = 1). Its increments over the sampler's uniform steps give each variable's information-gain density over t; the
proposed training distribution mixes the class and patch densities, (1 - patch_weight) x class + patch_weight x patch,
and the closest split logit-normal (m, sl, sr) to it is reported (train_jit.py --P_mean / --P_std / --P_std_right).

Every level (coarse ones, class, patch) gets its own curve, density and fitted split logit-normal. Per class: each level's
curve over that class's images (pointwise information about the true value) and the t where half of it is revealed, to
see how class-dependent the schedule is (more images per class = less noisy per-class curves: --num_images=50000 uses
all of ImageNet val).

Variability per label at every t (every level; patch grouped by the image's class): mean pointwise information and
top-1 accuracy with standard errors across the label's images, the between-label spread, and between-label variance /
within-label noise (~1: label differences are sampling noise; >> 1: real), plus each label's 25-75% transition width.

Writes {out}/tinfo_{run}.json (all curves, per-class curves, variability, fits), {out}/tinfo_{run}_variability.png and
{out}/tinfo_{run}.png: information vs t per level; per-sampler-step shares of every level (and the mix) against logitnorm(-0.8, 0.8) and splitlogitnorm(-0.8, 0.8,
1.2); the spread of per-class half-information points per level.

$ python t_information.py --ckpt=runs/in1k_128/jit-b16_icl0/<run> --data_dir=/data/in1k_128
"""

import os
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from jit_common import load_denoiser
from data_jit import MemmapImages, normalize, denormalize
from util.progress import bar

# -----------------------------------------------------------------------------
ckpt = '' # run directory (its ckpt.pt) or a checkpoint path
data_dir = ''
ema = 1 # 1 / 2 = that EMA's weights, 0 = raw
num_images = 10000 # held-out images (val split, balanced classes)
head_images_per_class = 50 # clean training images per class for the class head and the patch codes
levels = '10,100' # coarse levels of the class hierarchy (cluster counts)
patch_codes = 256
encoder = 'dinov2_vitb14' # torch.hub DINOv2 reader
grid = 16 # reader patch grid (16 x 14 = 224 px input)
ts = '' # comma-separated t values; '' = 31 logit-spaced from sigmoid(-4) to sigmoid(4.5)
patch_weight = 0.5 # share of the patch density in the mixed proposal
sampling_steps = 50
batch = 250
seed = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out = 'tinfo'
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

assert ckpt and data_dir, 'pass --ckpt and --data_dir'
path = ckpt if ckpt.endswith('.pt') else os.path.join(ckpt, 'ckpt.pt')
run_name = os.path.basename(os.path.dirname(path) if ckpt.endswith('.pt') else ckpt.rstrip('/'))
torch.manual_seed(seed)
ctx = torch.autocast('cuda', dtype=torch.bfloat16) if 'cuda' in device else torch.autocast('cpu', enabled=False)

denoiser, meta = load_denoiser(path, ema=ema, device=device)
net = denoiser.net
s = meta['denoiser_args'].get('noise_scale', 1.0)
train, val = MemmapImages(data_dir), MemmapImages(os.path.join(data_dir, 'val'))
C = train.num_classes
null = C # label-dropout index = unconditional

# reader: DINOv2 CLS + patch features of [-1, 1] images
reader = torch.hub.load('facebookresearch/dinov2', encoder).to(device).eval().requires_grad_(False)
MEAN = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
@torch.no_grad()
def read(x):
    x = F.interpolate(((x.float() + 1) / 2 - MEAN) / STD, size=(grid * 14, grid * 14), mode='bicubic', align_corners=False)
    with ctx:
        f = reader.forward_features(x)
    # (clone: the CLS feature is a view into the whole token tensor, which a stored view would keep alive)
    return f['x_norm_clstoken'].float().clone(), F.normalize(f['x_norm_patchtokens'].float(), dim=-1) # [B, D], [B, grid^2, D]

def to_x(u8):
    return torch.from_numpy(np.asarray(u8)).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1

def balanced(labels, per_class, rng):
    idx = [rng.choice(np.flatnonzero(labels == c), per_class, replace=False) for c in range(C)]
    return np.sort(np.concatenate(idx))

rng = np.random.default_rng(seed)
tr_idx = balanced(train.labels, head_images_per_class, rng)
va_idx = balanced(val.labels, num_images // C, rng)

# 1. clean training features: class head, coarse hierarchy, patch codes
cls_tr, patch_sample = [], []
for i in bar(range(0, len(tr_idx), batch), f'clean training features ({len(tr_idx)} images)'):
    c, p = read(to_x(train.images[tr_idx[i:i + batch]]))
    cls_tr.append(c)
    patch_sample.append(p[:, torch.randperm(p.shape[1], device=device)[:8]].reshape(-1, p.shape[-1])) # 8 patches per image
cls_tr = torch.cat(cls_tr)
y_tr = torch.from_numpy(train.labels[tr_idx]).to(device)
mu, sd = cls_tr.mean(0, keepdim=True), cls_tr.std(0, keepdim=True) + 1e-6
head = torch.nn.Linear(cls_tr.shape[1], C).to(device)
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
xs = (cls_tr - mu) / sd
for step in bar(range(2000), 'class head'):
    j = torch.randint(len(xs), (1024,), device=device)
    loss = F.cross_entropy(head(xs[j]), y_tr[j])
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
head.requires_grad_(False)

def kmeans(x, k, iters=30):
    # cosine k-means (x rows unit-norm), k-means++-style random init
    g = torch.Generator(device=x.device).manual_seed(seed)
    cent = x[torch.randperm(len(x), generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        a = (x @ cent.T).argmax(1)
        for j in range(k):
            m = a == j
            if m.any():
                cent[j] = F.normalize(x[m].mean(0), dim=0)
    return cent, (x @ cent.T).argmax(1)

class_means = F.normalize(torch.stack([F.normalize(cls_tr[y_tr == c], dim=-1).mean(0) for c in range(C)]), dim=-1)
level_k = [int(v) for v in levels.split(',') if v.strip()]
level_maps = {k: kmeans(class_means, k)[1] for k in level_k} # class -> cluster
codes, _ = kmeans(torch.cat(patch_sample), patch_codes)

# 2. held-out targets: class, coarse labels, clean patch codes
y_va = torch.from_numpy(val.labels[va_idx]).to(device)
code_va, cls_clean = [], []
for i in range(0, len(va_idx), batch):
    c, p = read(to_x(val.images[va_idx[i:i + batch]]))
    cls_clean.append(c)
    code_va.append((p @ codes.T).argmax(-1))
code_va = torch.cat(code_va) # [N, grid^2]

def entropy_bits(labels, k):
    p = torch.bincount(labels.flatten(), minlength=k).float()
    p = p / p.sum()
    return float(-(p[p > 0] * p[p > 0].log2()).sum())

H = {'class': entropy_bits(y_va, C), 'patch': entropy_bits(code_va, patch_codes)}
H.update({f'coarse{k}': entropy_bits(level_maps[k][y_va], k) for k in level_k})

TEMPS = np.exp(np.linspace(math.log(0.25), math.log(8), 15)).tolist()
KAPPAS = torch.exp(torch.linspace(math.log(2), math.log(200), 21)).tolist()
LEVELS = [f'coarse{k}' for k in level_k] + ['class', 'patch'] # coarse -> fine, then local content

def patch_ce_per_image(patch_f, targets):
    # mean cross-entropy (nats) over each image's patches at every sharpness kappa [B, len(KAPPAS)], and each image's share
    # of patches whose nearest code is the clean patch's code [B] (appended as the last column)
    sims = (patch_f @ codes.T) # [B, P, codes]
    ce = torch.stack([F.cross_entropy((k * sims).flatten(0, 1), targets.flatten(), reduction='none').view(targets.shape).mean(1)
                      for k in KAPPAS], 1)
    acc = (sims.argmax(-1) == targets).float().mean(1, keepdim=True)
    return torch.cat([ce, acc], 1).cpu()

def per_image_ce(cls_f, patch_ce):
    # per-image cross-entropies (bits) of every level at its best scalar temperature / sharpness (chosen over all images),
    # and per-image top-1 correctness (patch: share of patches with the right code) in ACC[level]
    logits = head((cls_f - mu) / sd)
    pred = logits.argmax(-1)
    ACC['class'].append((pred == y_va).float().cpu())
    for k in level_k:
        ACC[f'coarse{k}'].append((level_maps[k][pred] == level_maps[k][y_va]).float().cpu())
    ACC['patch'].append(patch_ce[:, -1])
    patch_ce = patch_ce[:, :-1]
    def best(fn, targets):
        ces = [F.cross_entropy(fn(v), targets, reduction='none') for v in TEMPS]
        j = int(torch.stack([c.mean() for c in ces]).argmin())
        return (ces[j] / math.log(2)).cpu()
    out = {'class': best(lambda T: logits / T, y_va)}
    for k in level_k: # coarse probabilities = sums of class probabilities within a cluster
        member = [level_maps[k] == j for j in range(k)]
        def coarse_logits(T, member=member):
            lp = F.log_softmax(logits / T, -1)
            return torch.stack([torch.logsumexp(lp[:, m_], -1) if m_.any() else torch.full_like(lp[:, 0], -1e9) for m_ in member], -1)
        out[f'coarse{k}'] = best(coarse_logits, level_maps[k][y_va])
    out['patch'] = patch_ce[:, int(patch_ce.mean(0).argmin())] / math.log(2)
    return out

# 3. per-image cross-entropies at each noise level, from the model's null-label prediction
t_list = [float(v) for v in ts.split(',') if v.strip()] if ts else (1 / (1 + np.exp(-np.linspace(-4, 4.5, 31)))).tolist()
def noise_for(n_, size):
    # the same noise for batch n_ at every t, regenerated per batch instead of held for all images
    g = torch.Generator(device=device).manual_seed(seed * 100003 + n_)
    return torch.randn((size, 3, val.img_size, val.img_size), generator=g, device=device)
ce_t = {k: [] for k in LEVELS} # per level: [len(t_list)] tensors of [N] bits
ACC = {k: [] for k in LEVELS} # per level: per-image top-1 correctness at each t (then the clean images last)
for t in bar(t_list, f'information at {len(t_list)} noise levels x {len(va_idx)} images'):
    cf, pce = [], []
    for n_, i in enumerate(range(0, len(va_idx), batch)):
        x = to_x(val.images[va_idx[i:i + batch]])
        z = t * normalize(x) + (1 - t) * s * noise_for(n_, len(x)) # noised in the model's data space (checkpoint's data_norm)
        with torch.no_grad(), ctx:
            x_hat = net(z, torch.full((len(x),), t, device=device), torch.full((len(x),), null, device=device, dtype=torch.long))
        c, p = read(denormalize(x_hat.float()).clamp(-1, 1)) # the reader sees [-1, 1] images
        cf.append(c)
        pce.append(patch_ce_per_image(p, code_va[i:i + batch]))
    for k, v in per_image_ce(torch.cat(cf), torch.cat(pce)).items():
        ce_t[k].append(v)
# t = 1: the reader on the clean images themselves
pce = [patch_ce_per_image(read(to_x(val.images[va_idx[i:i + batch]]))[1], code_va[i:i + batch]) for i in range(0, len(va_idx), batch)]
ce_clean = per_image_ce(torch.cat(cls_clean), torch.cat(pce))

# 4. information curves: overall and per class, normalized by the clean value, monotone in t
T = np.array([0.0] + t_list + [1.0])
edges = np.linspace(0, 1, sampling_steps + 1)
y_cpu = y_va.cpu().numpy()

def curve(ce_list, ce_cl, Hk, mask=None):
    # bits revealed at each t (mean over the masked images): H - CE, clipped at 0. Normalized by the most the model's
    # prediction reveals (its value at the cleanest measured t), not by the reader on the clean image: the clean image is
    # not a noise level the model sees, and the gap between the two (the prediction never reproducing the clean patch
    # codes exactly: the bottleneck floor near t = 1) would otherwise show up as a spike of 'information' at t = 1.
    # The clean value is returned separately (third value) for reference.
    sel = (lambda v: v[mask]) if mask is not None else (lambda v: v)
    bits = [max(Hk - float(sel(v).mean()), 0.0) for v in ce_list]
    top = max(max(bits), 1e-9)
    clean_bits = Hk - float(sel(ce_cl).mean())
    return np.maximum.accumulate(np.array([0.0] + bits + [top]) / top), bits, clean_bits

def step_shares(I):
    d = np.diff(np.interp(edges, T, I))
    return d / max(d.sum(), 1e-12)

def half_t(I):
    return float(np.interp(0.5, I + np.arange(len(I)) * 1e-9, T)) # (tiny ramp: ties at 0 / 1)

norm, bits, clean, shares = {}, {}, {}, {}
for k in LEVELS:
    norm[k], bits[k], clean[k] = curve(ce_t[k], ce_clean[k], H[k])
    shares[k] = step_shares(norm[k])
mix = (1 - patch_weight) * shares['class'] + patch_weight * shares['patch']

# per class: each level's curve over that class's images (I_c = H - CE_c, the pointwise information about the true value)
per_class = {k: {} for k in LEVELS}
for c in range(C):
    m_ = y_cpu == c
    for k in LEVELS:
        I_c = curve(ce_t[k], ce_clean[k], H[k], m_)[0]
        per_class[k][c] = {'half_t': half_t(I_c), 'normalized': I_c.tolist()}

# variability per label at every t: for each level, its labels (coarse clusters, classes; patch: the image's class), each
# label's mean pointwise information (normalized by that label's clean value) and accuracy with standard errors across the
# label's images, the between-label spread, and the ratio of between-label variance to within-label noise (~1: label
# differences are sampling noise; >> 1: real). Plus each label's transition width (t from 25% to 75% revealed).
def label_stats(k):
    lab = (level_maps[int(k[6:])][y_va] if k.startswith('coarse') else y_va).cpu().numpy()
    labels = np.unique(lab)
    info = np.stack([(H[k] - v.numpy()) for v in ce_t[k]] + [H[k] - ce_clean[k].numpy()]) # [t + clean, N] pointwise bits
    acc = np.stack([a.numpy() for a in ACC[k]]) # [t + clean, N]
    M, SE, A, ASE, width = [], [], [], [], []
    for j in labels:
        m_ = lab == j
        n = int(m_.sum())
        top = max(info[-2, m_].mean(), 1e-9) # the cleanest measured t (as curve())
        mean = info[:, m_].mean(1) / top
        se = info[:, m_].std(1, ddof=1) / math.sqrt(n) / top if n > 1 else np.full(len(mean), np.nan)
        a = acc[:, m_].mean(1)
        M.append(mean); SE.append(se); A.append(a); ASE.append(np.sqrt(a * (1 - a) / n))
        mono = np.maximum.accumulate(np.concatenate([[0.0], np.clip(mean[:-1], 0, None), [1.0]]))
        width.append(float(np.interp(0.75, mono + np.arange(len(mono)) * 1e-9, T) - np.interp(0.25, mono + np.arange(len(mono)) * 1e-9, T)))
    M, SE, A, ASE = (np.stack(v) for v in (M, SE, A, ASE)) # [labels, t + clean]
    ratio = M.var(0, ddof=1) / np.maximum(np.nanmean(SE ** 2, 0), 1e-6) # between-label variance / mean within-label noise (floored)
    return {'labels': labels.tolist(), 'mean': M, 'se': SE, 'acc': A, 'acc_se': ASE, 'between_within': ratio, 'width': np.array(width)}

var_stats = {k: label_stats(k) for k in LEVELS}

def split_shares(m, sl, sr):
    # per-step mass of a split logit-normal on the sampler grid
    e = torch.tensor(edges[1:-1], dtype=torch.float64)
    z = torch.log(e / (1 - e))
    wl = sl / (sl + sr)
    cdf = torch.where(z <= m, 2 * wl * 0.5 * (1 + torch.special.erf((z - m) / (sl * math.sqrt(2)))),
                      wl + 2 * (1 - wl) * 0.5 * torch.special.erf((z - m) / (sr * math.sqrt(2))))
    return np.diff(np.concatenate([[0.0], cdf.numpy(), [1.0]]))

GRID = [(m, sl, sr) for m in np.linspace(-2.5, 1.5, 41) for sl in np.linspace(0.3, 2.5, 23) for sr in np.linspace(0.3, 3.0, 28)]
GRID_SHARES = np.stack([split_shares(*p_) for p_ in bar(GRID, 'split logit-normal table')])
def fit_split(target):
    j = int(((GRID_SHARES - target) ** 2).sum(1).argmin())
    return tuple(round(float(v), 2) for v in GRID[j])
fits = {k: fit_split(shares[k]) for k in LEVELS}
fits['mix'] = fit_split(mix)

# 5. report
try:
    names = val.class_names()
except Exception:
    names = [str(i) for i in range(C)]
os.makedirs(out, exist_ok=True)
res = {'ckpt': path, 'iter': meta.get('iter_num'), 'noise_scale': s, 't': t_list, 't_grid': T.tolist(), 'levels': LEVELS,
       'entropy_bits': H, 'clean_bits': clean, 'bits': bits, 'normalized': {k: v.tolist() for k, v in norm.items()},
       'half_t': {k: half_t(norm[k]) for k in LEVELS}, 'step_shares': {k: v.tolist() for k, v in shares.items()},
       'mix_shares': mix.tolist(), 'patch_weight': patch_weight, 'fit_splitlogitnorm': fits,
       'level_maps': {str(k): level_maps[k].tolist() for k in level_k},
       'per_class': {k: {names[c]: v for c, v in d.items()} for k, d in per_class.items()},
       'variability': {k: {'t': t_list + [1.0], 'labels': v['labels'], 'mean': v['mean'].tolist(), 'se': v['se'].tolist(),
                           'acc': v['acc'].tolist(), 'acc_se': v['acc_se'].tolist(), 'between_within': v['between_within'].tolist(),
                           'transition_width': v['width'].tolist()} for k, v in var_stats.items()},
       'settings': {k: globals()[k] for k in config_keys}}
with open(os.path.join(out, f'tinfo_{run_name}.json'), 'w') as f:
    json.dump(res, f, indent=1)

print(f'\n{run_name} (iter {meta.get("iter_num")}), {len(va_idx)} held-out images ({len(va_idx) // C} per class):')
print(f"  {'level':<10} {'clean bits':>10} {'of H':>7} {'half at t':>10} {'per-class half t: 10% / 50% / 90%':>36}  closest splitlogitnorm")
for k in LEVELS:
    h = np.array([per_class[k][c]['half_t'] for c in range(C)])
    q = np.percentile(h, [10, 50, 90])
    print(f"  {k:<10} {clean[k]:>10.2f} {H[k]:>7.2f} {half_t(norm[k]):>10.3f} {q[0]:>14.3f} / {q[1]:.3f} / {q[2]:.3f}      {fits[k]}")
print(f"  mixed ({1 - patch_weight:g} class + {patch_weight:g} patch) -> splitlogitnorm{fits['mix']}")
def _named(cs, k):
    return ', '.join('%s (%.2f)' % (names[c], per_class[k][c]['half_t']) for c in cs)
for k in ('class', 'patch'):
    order = sorted(range(C), key=lambda c: per_class[k][c]['half_t'])
    print(f'  {k}: earliest {_named(order[:5], k)}')
    print(f'  {k}: latest   {_named(order[-5:], k)}')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors = {k: col for k, col in zip(LEVELS, ['#7fb3d5', '#2e86c1', '#1b4f72', '#c0392b'][-len(LEVELS):] if len(LEVELS) <= 4 else plt.cm.viridis(np.linspace(0, 1, len(LEVELS))))}
fig, axes = plt.subplots(3, 1, figsize=(10, 11))
a1, a2, a3 = axes
for k in LEVELS:
    a1.plot(T, norm[k], color=colors[k], lw=1.6, label=f'{k}: {clean[k]:.2f} of {H[k]:.2f} bits clean, half at t={half_t(norm[k]):.3f}')
a1.set_xlabel('t (data fraction)'); a1.set_ylabel('I(Y; z_t) / I(Y; x)'); a1.set_xlim(0, 1); a1.grid(alpha=0.3); a1.legend(fontsize=7)
a1.set_title(f'{run_name}: information revealed per hierarchy level (null label, DINOv2 reader on the prediction)', fontsize=9)
cen = (edges[:-1] + edges[1:]) / 2
a2.bar(cen, 100 * mix, width=0.9 / sampling_steps, color='0.85', label=f"mixed ({1 - patch_weight:g} class + {patch_weight:g} patch) -> fit {fits['mix']}")
for k in LEVELS:
    a2.plot(cen, 100 * shares[k], color=colors[k], lw=1.3, label=f'{k} -> fit splitlogitnorm{fits[k]}')
for (m, sl, sr), style, lab in (((-0.8, 0.8, 0.8), ':', 'logitnorm(-0.8, 0.8)'), ((-0.8, 0.8, 1.2), '--', 'splitlogitnorm(-0.8, 0.8, 1.2)')):
    a2.plot(cen, 100 * split_shares(m, sl, sr), color='k', ls=style, lw=1, label=lab)
a2.axhline(100 / sampling_steps, color='0.5', lw=0.6)
a2.set_xlabel(f"t (the sampler's {sampling_steps} uniform steps)"); a2.set_ylabel('% of each level\'s information per step'); a2.set_xlim(0, 1)
a2.legend(fontsize=6.5); a2.grid(alpha=0.3)
bins = np.linspace(0, 1, 51)
for k in LEVELS:
    a3.hist([per_class[k][c]['half_t'] for c in range(C)], bins=bins, histtype='step', color=colors[k], lw=1.4, label=k)
a3.set_xlabel('t where half of a class\'s information is revealed'); a3.set_ylabel('classes'); a3.set_xlim(0, 1)
a3.set_title(f'class dependence: spread of per-class half-information points ({len(va_idx) // C} images per class)', fontsize=9)
a3.legend(fontsize=7); a3.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out, f'tinfo_{run_name}.png'), dpi=150)
print('variability per label (between-label variance / within-label noise: ~1 = no real label differences):')
pick = sorted({int(np.argmin(np.abs(np.array(t_list) - v))) for v in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)})
print(f"  {'level':<10} {'labels':>6} " + ''.join(f"{'t=%.2f' % t_list[i]:>9}" for i in pick) + f"{'25-75% width: 10/50/90%':>28}")
for k in LEVELS:
    v = var_stats[k]
    w = np.percentile(v['width'], [10, 50, 90])
    print(f"  {k:<10} {len(v['labels']):>6} " + ''.join(f"{v['between_within'][i]:>9.3g}" for i in pick) + f"{w[0]:>14.3f} / {w[1]:.3f} / {w[2]:.3f}")

fig, axes = plt.subplots(3, len(LEVELS), figsize=(3.6 * len(LEVELS), 9), squeeze=False)
Tt = np.array(t_list + [1.0])
for col, k in enumerate(LEVELS):
    v = var_stats[k]
    for row, (Y, YSE, lab) in enumerate(((v['mean'], v['se'], 'information / clean'), (v['acc'], v['acc_se'], 'top-1 accuracy' if k != 'patch' else 'patch code accuracy'))):
        ax = axes[row, col]
        q = np.nanpercentile(Y, [10, 25, 50, 75, 90], axis=0)
        ax.fill_between(Tt, q[0], q[4], color=colors[k], alpha=0.15, lw=0, label='labels 10-90%')
        ax.fill_between(Tt, q[1], q[3], color=colors[k], alpha=0.3, lw=0, label='labels 25-75%')
        ax.plot(Tt, q[2], color=colors[k], lw=1.5, label='median label')
        se_med = np.nanmedian(YSE, 0)
        ax.plot(Tt, q[2] + 2 * se_med, color='k', lw=0.6, ls=':', label='median label +- 2 SE (sampling noise)')
        ax.plot(Tt, q[2] - 2 * se_med, color='k', lw=0.6, ls=':')
        ax.set_xlim(0, 1); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel(lab, fontsize=8)
        if row == 0:
            ax.set_title(f'{k} ({len(v["labels"])} labels)', fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=6)
    ax = axes[2, col]
    ax.semilogy(Tt[:-1], np.maximum(v['between_within'][:-1], 1e-3), color=colors[k], lw=1.5) # (t = 1 is 1 for every label by construction)
    ax.axhline(1, color='k', lw=0.8, ls='--')
    ax.set_xlim(0, 1); ax.grid(alpha=0.3, which='both'); ax.tick_params(labelsize=7); ax.set_xlabel('t', fontsize=8)
    if col == 0:
        ax.set_ylabel('between-label var / within-label noise', fontsize=8)
fig.suptitle(f'{run_name}: variability per label at each t ({len(va_idx) // C} images per class)', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(out, f'tinfo_{run_name}_variability.png'), dpi=150)
print(f'wrote {out}/tinfo_{run_name}.json, .png and _variability.png')
