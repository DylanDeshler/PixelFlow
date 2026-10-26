"""
Look at the quadtree router before training anything (research_proposal_fixed_budget_quadtree_patch_allocation.md,
model_quadtree.py). On held-out images of a prepared dataset:

    figure 1  {out}.png        each image, DINOv2's PCA-1 foreground projection and gate (dino router), and the layouts
                               every router in --routers produces at every budget in --tokens, routed on the clean image
    figure 2  {out}_noisy.png  (with --baseline) the layout routed the way training and sampling route it: on a frozen
                               baseline's clean-image estimate from the image noised to each t in --ts, next to the
                               clean-image layout, with the fraction of the image tokenized at the same level ("agree")

Token boundaries are coloured by the finer of the two tokens they separate (32px blue, 16px orange, 8px green at 128px).
Also prints the area per level for every router and budget, and the clean-vs-noisy agreement per t.

$ python viz_quadtree_router.py --data_dir=/data/in1k_128
$ python viz_quadtree_router.py --data_dir=/data/in1k_128 --baseline=runs/in1k_128/jit-b16_icl0/<run>/ckpt_<iter>.pt --ts=0.3,0.5,0.7,0.9
"""

import os
from contextlib import nullcontext

import numpy as np
import torch

from data_jit import MemmapImages, to_model_input
from model_quadtree import QuadGeometry, QuadRouter, allocate, draw_layout, layout_agreement, LEVEL_COLORS

# -----------------------------------------------------------------------------
data_dir = '' # a prepared dataset (its val/ split is used if present)
n_images = 8
seed = 0 # which images (random, spread over the split)
tokens = '64,100,160' # budgets K (16 + 3m at 128px with 3 levels)
routers = 'dino,pixel,random' # dino, pixel, random, uniform
dino = 'dinov2_vitb14_reg'
gate_tau = 1.0
levels = 3
patch_size = 8 # finest level
baseline = '' # frozen uniform JiT checkpoint: also route on its estimate from noised images (figure 2)
ts = '0.3,0.5,0.7,0.9'
noisy_tokens = 64 # budget for figure 2
out = '' # output path prefix; default {data_dir name}_quadtree_router
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

assert data_dir, 'pass --data_dir'
out = out or f"{os.path.basename(os.path.normpath(data_dir))}_quadtree_router"
ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if 'cuda' in device else nullcontext()
split = os.path.join(data_dir, 'val') if os.path.isdir(os.path.join(data_dir, 'val')) else data_dir
data = MemmapImages(split)
idx = np.sort(np.random.default_rng(seed).choice(len(data), n_images, replace=False))
images, labels = data.get(idx)
names = [data.class_names()[i] for i in labels.tolist()]
x = to_model_input(images, flip=False, device=device)
budgets = [int(k) for k in tokens.split(',')]
router_list = [r for r in routers.split(',') if r]
geo = QuadGeometry(data.img_size, patch_size, levels)
for K in budgets + [noisy_tokens]:
    geo.check_budget(K)
router = QuadRouter(geo, geo.n0, 'uniform', gate_tau=gate_tau, img=data.img_size) # scorers and neutral scores; budgets passed to allocate
if 'dino' in router_list or baseline:
    router.load_dino(dino)
router.to(device)

@torch.no_grad()
def scores_for(name, x0):
    if name == 'dino':
        with ctx:
            return router.dino_scores(x0, parts=True)
    if name == 'pixel':
        return router.pixel_scores(x0), None
    if name == 'random':
        return router.random_scores(x0), None
    return router.neutral.expand(x0.shape[0], -1), None

def layout(scores, K):
    return allocate(scores, K, geo, router.children, router.parent)

# routed on the clean images
scores, parts = {}, None
for name in router_list:
    scores[name], p = scores_for(name, x)
    parts = p if name == 'dino' else parts
layouts = {(name, K): layout(scores[name], K) for name in router_list for K in budgets}

print(f'{n_images} images from {split} ({data.img_size}px), levels {geo.patches} px')
print('area per level (coarsest first), mean over images, routed on the clean image:')
for (name, K), lay in layouts.items():
    lvl = geo.level.to(lay.device)[lay]
    w2 = 4.0 ** (geo.L - 1 - lvl).float()
    area = [((w2 * (lvl == l)).sum(1) / geo.counts[-1]).mean().item() for l in range(geo.L)]
    print(f"  {name:>7} K={K:<4} " + '  '.join(f'{p}px {a:5.1%}' for p, a in zip(geo.patches, area)))

# figure 1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plot_style import SURFACE, INK, INK_2

to_np = lambda t: ((t.float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
columns = [('image', None)]
if parts is not None:
    columns += [('DINO PCA-1 (foreground +)', 'proj'), ('foreground gate', 'gate')]
columns += [(f'{name} K={K}', (name, K)) for name in router_list for K in budgets]

def figure(rows, cols, title):
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.8 * rows + 0.6), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    handles = [Patch(color=LEVEL_COLORS[l], label=f'{p}px') for l, p in enumerate(geo.patches)]
    fig.legend(handles=handles, loc='upper right', ncol=len(handles), frameon=False, fontsize=8, title='token boundaries (finer side)', title_fontsize=8)
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha='left')
    return fig, axes

fig, axes = figure(n_images, len(columns), f'quadtree router on clean held-out images ({data.img_size}px)')
for i in range(n_images):
    for j, (title, what) in enumerate(columns):
        ax = axes[i, j]
        if what is None:
            ax.imshow(to_np(x[i]))
        elif what == 'proj':
            m = parts['proj'][i].abs().max().item()
            ax.imshow(parts['proj'][i].cpu().numpy(), cmap='RdBu_r', vmin=-m, vmax=m, interpolation='nearest')
        elif what == 'gate':
            ax.imshow(parts['gate_cells'][i].cpu().numpy(), cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        else:
            ax.imshow(to_np(draw_layout(x[i:i + 1], layouts[what][i:i + 1], geo)[0]))
        if i == 0:
            ax.set_title(title, color=INK, fontsize=8)
        if j == 0:
            ax.set_ylabel(names[i][:16], color=INK_2, fontsize=7)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out + '.png', dpi=140, facecolor=SURFACE)
print(f'wrote {out}.png')

# figure 2: routed on a frozen baseline's estimate from noised images, as in training and sampling
if baseline:
    from jit_common import load_denoiser
    base, _ = load_denoiser(baseline, ema=1, device=device)
    assert base.img_size == data.img_size, f'baseline is {base.img_size}px, the data {data.img_size}px'
    route_name = 'dino' if 'dino' in router_list else next(r for r in router_list if r not in ('random', 'uniform'))
    t_list = [float(t) for t in ts.split(',')]
    clean = layout(scores_for(route_name, x)[0], noisy_tokens)
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=g, device=device) * base.noise_scale
    y_null = torch.full((n_images,), base.num_classes, device=device, dtype=torch.long)
    rows = []
    print(f'{route_name} router on the baseline estimate vs on the clean image, K={noisy_tokens} (fraction of area at the same level):')
    for t in t_list:
        tt = torch.full((n_images,), t, device=device)
        with torch.no_grad(), ctx:
            x0 = base.net(t * x + (1 - t) * noise, tt, y_null).float()
        lay = layout(scores_for(route_name, x0)[0], noisy_tokens)
        agree = layout_agreement(clean, lay, geo)
        rows.append((t, x0, lay, agree))
        print(f'  t={t:.2f}: agree {agree.mean().item():.1%} (min {agree.min().item():.1%}, max {agree.max().item():.1%})')
    fig, axes = figure(n_images, 2 + len(t_list), f'{route_name} router, K={noisy_tokens}: on the clean image vs on the baseline estimate from x_t')
    for i in range(n_images):
        axes[i, 0].imshow(to_np(x[i]))
        axes[i, 1].imshow(to_np(draw_layout(x[i:i + 1], clean[i:i + 1], geo)[0]))
        for j, (t, x0, lay, agree) in enumerate(rows):
            axes[i, 2 + j].imshow(to_np(draw_layout(x0[i:i + 1], lay[i:i + 1], geo)[0]))
            axes[i, 2 + j].set_xlabel(f'agree {agree[i].item():.0%}', color=INK_2, fontsize=7)
            if i == 0:
                axes[i, 2 + j].set_title(f'estimate at t={t:.2f}', color=INK, fontsize=8)
        if i == 0:
            axes[i, 0].set_title('image', color=INK, fontsize=8)
            axes[i, 1].set_title('routed on the clean image', color=INK, fontsize=8)
        axes[i, 0].set_ylabel(names[i][:16], color=INK_2, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out + '_noisy.png', dpi=140, facecolor=SURFACE)
    print(f'wrote {out}_noisy.png')
