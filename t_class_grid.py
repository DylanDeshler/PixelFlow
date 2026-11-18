"""
Class-specific sampling time grids from t_information.py's per-class information curves: does placing each class's 50
sampler steps where its own information is revealed beat one grid for all classes? Inference only.

Grids (the sampler, steps and CFG otherwise unchanged; every grid runs from t = 0 to t = 1):
  uniform    JiT's linspace(0, 1)
  global     steps at quantiles of a blend of uniform and the overall information curve: CDF(t) = (1 - blend) t + blend I(t)
  class      the same with each class's own curve I_c(t) (per-sample grids in one batch)
I(t) is the normalized, monotone information curve of --level ('class', 'patch', a coarse level, or 'mix' = (1 - w) class
+ w patch with t_information's patch_weight); blend < 1 keeps some steps everywhere (a pure information grid leaves
almost none near t = 0 and t = 1, where the last step would jump).

Scored on class groups, each against reference statistics of the same classes (an all-class reference would penalize
the group's class histogram): the --group_size classes whose information is revealed earliest, the latest, and all
classes. FID (Inception features of the training images, train_inception.npy from fid.py) and FD-DINOv2, with the same
labels and noise for every grid (paired).

Writes {out}/classgrid_{run}.json and .png (the grids of a few early / late classes, and the scores).

$ python t_class_grid.py --ckpt=runs/in1k_128/jit-b16_icl0/<run> --tinfo=tinfo/tinfo_<run>.json --data_dir=/data/in1k_128
"""

import os
import json
from contextlib import nullcontext

import numpy as np
import torch

from jit_common import load_denoiser
from fid import to_uint8
from data_jit import MemmapImages
from fid import InceptionMetrics, frechet_distance
from dino_fd import DinoFD
from util.progress import bar

# -----------------------------------------------------------------------------
ckpt = ''
tinfo = '' # t_information.py JSON (per-class curves)
data_dir = ''
level = 'mix' # 'class', 'patch', 'coarse<k>' or 'mix'
blend = 0.5 # grid CDF = (1 - blend) uniform + blend information
group_size = 200 # classes in the early / late groups
per_class = 50 # samples per class in the early / late groups (all classes: 10 per class)
cfg = 1.0
interval_min = 0.0
interval_max = 1.0
steps = 50
dino = True
dino_encoder = 'dinov2_vitb14'
batch = 250
seed = 0
compile = True
device = 'cuda'
out = 'tclassgrid'
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

assert ckpt and tinfo and data_dir, 'pass --ckpt, --tinfo and --data_dir'
path = ckpt if ckpt.endswith('.pt') else os.path.join(ckpt, 'ckpt.pt')
run_name = os.path.basename(os.path.dirname(path) if ckpt.endswith('.pt') else ckpt.rstrip('/'))
ctx = torch.autocast('cuda', dtype=torch.bfloat16) if 'cuda' in device else nullcontext()
info = json.load(open(tinfo))
data = MemmapImages(data_dir)
C = data.num_classes
names = data.class_names()
T = np.array(info['t_grid'])

def curve_of(name=None):
    # normalized information curve on T: overall (name None) or one class's
    def get(k):
        return np.array(info['normalized'][k] if name is None else info['per_class'][k][name]['normalized'])
    if level == 'mix':
        w = info['patch_weight']
        I = (1 - w) * get('class') + w * get('patch')
    else:
        I = get(level)
    return np.maximum.accumulate(np.clip(I, 0, 1))

def grid_from(I):
    # steps + 1 times at the quantiles of CDF(t) = (1 - blend) t + blend I(t)
    cdf = (1 - blend) * T + blend * I
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    cdf = cdf + np.arange(len(cdf)) * 1e-9 # strictly increasing for the inversion
    g = np.interp(np.linspace(0, 1, steps + 1), cdf, T)
    g[0], g[-1] = 0.0, 1.0
    return g

grids_class = np.stack([grid_from(curve_of(names[c])) for c in range(C)]) # [C, steps + 1]
grid_global = grid_from(curve_of())
grid_uniform = np.linspace(0, 1, steps + 1)

# class groups by the half-information t of the chosen level (mix: average of class and patch)
def half(c):
    if level == 'mix':
        return 0.5 * (info['per_class']['class'][names[c]]['half_t'] + info['per_class']['patch'][names[c]]['half_t'])
    return info['per_class'][level][names[c]]['half_t']
order = sorted(range(C), key=half)
groups = {'early': order[:group_size], 'late': order[-group_size:], 'all': list(range(C))}

denoiser, meta = load_denoiser(path, ema=1, device=device)
if compile:
    denoiser.net.compile()
denoiser.cfg_scale, denoiser.cfg_interval, denoiser.method = cfg, (interval_min, interval_max), 'heun'
metrics = InceptionMetrics(device)
fd = DinoFD(device, encoder=dino_encoder) if dino else None
train_feats = np.load(os.path.join(data_dir, 'train_inception.npy'), mmap_mode='r')

def inception_ref(classes):
    # mean / covariance of the training images' Inception features of these classes, streamed in chunks (float64 sums)
    if len(classes) == C: # all classes: fid.py's reference stats
        from fid import load_ref_stats
        p_ = os.path.join(data_dir, 'fid_stats_jit.npz')
        return load_ref_stats(p_ if os.path.exists(p_) else os.path.join(data_dir, 'fid_stats.npz'))
    idx = np.flatnonzero(np.isin(data.labels, classes))
    s1, s2 = torch.zeros(2048, dtype=torch.float64), torch.zeros(2048, 2048, dtype=torch.float64)
    for i in range(0, len(idx), 20000):
        f = torch.from_numpy(train_feats[idx[i:i + 20000]].astype(np.float32)).double()
        s1 += f.sum(0); s2 += f.T @ f
    n = len(idx)
    mu = s1 / n
    return {'mu': mu.float().numpy(), 'sigma': ((s2 - n * torch.outer(mu, mu)) / (n - 1)).numpy()}

@torch.no_grad()
def sample(labels, grid_rows):
    # Heun with a per-sample time grid (grid_rows [B, steps + 1]), the last step Euler, as JiT's sampler
    B = len(labels)
    z = denoiser.noise_scale * torch.randn(B, 3, data.img_size, data.img_size, device=device)
    ts = torch.from_numpy(grid_rows.T.copy()).float().to(device).view(steps + 1, B, 1, 1, 1)
    with ctx:
        for i in range(steps - 1):
            z = denoiser._heun_step(z, ts[i], ts[i + 1], labels)
        z = denoiser._euler_step(z, ts[-2], ts[-1], labels)
    return z

def score(group, grid_kind):
    classes = groups[group]
    n_per = per_class if group != 'all' else 10
    labels_all = np.repeat(np.array(classes), n_per)
    inc, din = [], []
    with torch.random.fork_rng(devices=[device] if 'cuda' in device else []):
        torch.manual_seed(seed)
        for s in bar(range(0, len(labels_all), batch), f'{group} classes, {grid_kind} grid ({len(labels_all)} samples)'):
            lab = labels_all[s:s + batch]
            rows = {'uniform': np.tile(grid_uniform, (len(lab), 1)), 'global': np.tile(grid_global, (len(lab), 1)),
                    'class': grids_class[lab]}[grid_kind]
            x = to_uint8(sample(torch.from_numpy(lab).long().to(device), rows))
            inc.append(metrics.features(x)[0])
            if dino:
                din.append(fd.features(x))
    r = {'fid': frechet_distance(torch.cat(inc), refs[group]['inception'], device)}
    if dino:
        r['fd_dino'] = fd.distance(torch.cat(din), refs[group]['dino'])
    return r

os.makedirs(out, exist_ok=True)
refs = {}
for g, cl in groups.items():
    refs[g] = {'inception': inception_ref(cl)}
    if dino:
        refs[g]['dino'] = fd.reference(data, n_per_class=50, classes=cl,
                                       cache=os.path.join(out, f'dino_ref_{run_name}_{level}_{g}{group_size}.npz' if g != 'all' else 'dino_ref_train50k.npz'))

res_path = os.path.join(out, f'classgrid_{run_name}_{level}_blend{blend:g}.json')
res = {'settings': {k: globals()[k] for k in config_keys}, 'groups': {g: [names[c] for c in cl] for g, cl in groups.items()},
       'grid_uniform': grid_uniform.tolist(), 'grid_global': grid_global.tolist(), 'scores': {}}
for g in groups:
    for kind in ('uniform', 'global', 'class'):
        res['scores'][f'{g}/{kind}'] = score(g, kind)
        print(f'{g:>5} classes, {kind:>7} grid: {res["scores"][f"{g}/{kind}"]}')
        with open(res_path, 'w') as f:
            json.dump(res, f, indent=1)

print(f'\n{run_name}: class-specific grids ({level}, blend {blend}); lower is better')
for g in groups:
    row = ', '.join(f"{kind} " + ' / '.join(f'{res["scores"][f"{g}/{kind}"][m]:.2f}' for m in res['scores'][f'{g}/{kind}']) for kind in ('uniform', 'global', 'class'))
    print(f'  {g:>5} (FID / FD-DINOv2): {row}')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
u = np.linspace(0, 1, steps + 1)
ax.plot(u, grid_uniform, color='0.6', lw=1, label='uniform')
ax.plot(u, grid_global, color='k', lw=1.6, label='global')
for c in groups['early'][:5]:
    ax.plot(u, grids_class[c], color='#2e86c1', lw=0.9, alpha=0.8)
for c in groups['late'][-5:]:
    ax.plot(u, grids_class[c], color='#c0392b', lw=0.9, alpha=0.8)
ax.plot([], [], color='#2e86c1', label='5 earliest classes'); ax.plot([], [], color='#c0392b', label='5 latest classes')
ax.set_xlabel('sampler step / steps'); ax.set_ylabel('t'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title(f'{run_name}: sampling grids ({level}, blend {blend})', fontsize=9)
fig.tight_layout()
fig.savefig(res_path[:-5] + '.png', dpi=150)
print(f'wrote {res_path} and .png')
