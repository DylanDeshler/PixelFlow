"""
Main results of analyze_patch_sizes.py: FID vs tokens per image for every router (measured points, fitted curves), and
the two ways of reading an improvement over the uniform layout:
  matched tokens  FID change vs uniform at the same K: a drop-in replacement (measured vs measured; the line is the fits'
                  difference)
  matched FID     tokens saved, 1 - K / K_uniform, with K_uniform the budget at which uniform reaches the router's FID: a
                  compute trade-off (FLOPs are ~proportional to K for JiT-B at <= 256 tokens: attention is < 6% of them)
Fits over the trade range K = 64..256, where every router has its K = 64 / 100 / 160 points and meets the 8px model at
K = 256: FID quadratic in log2 K. Uniform below K = 64 (towards the 32px model at K = 16): log-linear between the two.

$ python plot_patch_sizes.py --json_path=in1k_128_patch_sizes.json      # -> in1k_128_patch_sizes_main.png
"""

import os
import json

import numpy as np

# -----------------------------------------------------------------------------
json_path = '' # analyze_patch_sizes.py output
routers = 'uniform,random,dino,loss,loss_exact' # + X_2level (dashed), X_pooled (dash-dot), X_perimage (dotted)
out = '' # default: {json without .json}_main.png
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

assert json_path, 'pass --json_path'
with open(json_path) as fh:
    res = json.load(fh)
fid = {k: v['fid'] for k, v in res['fid'].items()}
p32, p16, p8 = res['patches']
n_img = res.get('config', {}).get('num_images', 10000)
out = out or json_path.removesuffix('.json') + '_main.png'

two_level = lambda r: r.endswith(('_2level', '_pooled', '_perimage'))

# per router: K -> FID over the trade range (64..256), the 8px model at K = 256
anchors = {16: fid[f'uniform_{p32}px'], 64: fid[f'uniform_{p16}px'], 256: fid[f'uniform_{p8}px']}
series = {}
for r in routers.split(','):
    pts = {256: anchors[256]}
    if r == 'uniform' or two_level(r): # 2-level layouts start from the 16px model at K = 64
        pts[64] = anchors[64]
    for k, v in fid.items():
        if k.startswith(f'{r}_K') and k[len(r) + 2:].isdigit():
            pts[int(k[len(r) + 2:])] = v
    if len(pts) >= 3:
        series[r] = dict(sorted(pts.items()))
fit = {r: np.polyfit(np.log2(list(p)), list(p.values()), 2) for r, p in series.items()}
f_at = lambda r, K: np.polyval(fit[r], np.log2(K))
Kr = np.geomspace(64, 256, 200)
measured = lambda r: [K for K in series[r] if K < 256 and not (K == 64 and two_level(r))]

def k_uniform(f):
    # the budget at which uniform reaches FID f: the fit over 64..256, log-linear to the 32px model below 64
    K = np.geomspace(16, 256, 800)
    lo = K < 64
    f64 = f_at('uniform', 64) # continuous at 64
    y = np.where(lo, anchors[16] + (f64 - anchors[16]) * np.log2(K / 16) / 2, f_at('uniform', K))
    assert np.all(np.diff(y) < 0), 'uniform curve not decreasing'
    return np.interp(f, y[::-1], K[::-1], left=np.nan, right=np.nan)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_style import SURFACE, INK, INK_2, style_axis, end_labels

COLORS = {'uniform': '#2a78d6', 'random': '#1baf7a', 'dino': '#eda100', 'loss': '#e87ba4', 'loss_exact': '#a8336b', 'pixel': '#008300'}
MARKERS = {'uniform': 'o', 'random': 's', 'dino': '^', 'loss': 'D', 'loss_exact': 'P', 'pixel': 'v'}
# a 2-level variant (16 / 8 px only): its router's colour and marker, dashed line, hollow marker
base = lambda r: r.removesuffix('_pooled').removesuffix('_perimage').removesuffix('_2level')
for r in list(series):
    COLORS.setdefault(r, COLORS.get(base(r), INK_2))
    MARKERS.setdefault(r, 'h' if r.endswith('_pooled') else 'X' if r.endswith('_perimage') else MARKERS.get(base(r), 'o'))
LS = lambda r: '-.' if r.endswith('_pooled') else ':' if r.endswith('_perimage') else '--' if r.endswith('_2level') else '-'
mk = dict(markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, linestyle='none')
mk_of = lambda r: dict(mk, markerfacecolor=SURFACE, markeredgecolor=COLORS[r]) if two_level(r) else mk

def log2_axis(ax, ks):
    ax.set_xscale('log', base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5), facecolor=SURFACE)

# 1. FID vs K
ax = axes[0]
K_lo = np.geomspace(16, 64, 50)
ax.plot(K_lo, anchors[16] + (anchors[64] - anchors[16]) * np.log2(K_lo / 16) / 2, color=COLORS['uniform'], linewidth=1.5, linestyle=':')
ax.plot([16], [anchors[16]], MARKERS['uniform'], color=COLORS['uniform'], **mk)
for r, p in series.items():
    ax.plot(Kr, f_at(r, Kr), color=COLORS[r], linewidth=2, linestyle=LS(r))
    ax.plot(list(p), list(p.values()), MARKERS[r], color=COLORS[r], **mk_of(r))
for K, name in ((16, f'all {p32}px'), (64, f'all {p16}px'), (256, f'all {p8}px')):
    ax.annotate(name, (K, anchors[K]), xytext=(7, 7), textcoords='offset points', color=INK_2, fontsize=8)
style_axis(ax, 'tokens per image K', f'FID-{n_img / 1000:g}k (lower is better)', 'FID vs tokens: measured (markers), fit (lines)')
log2_axis(ax, sorted({16} | {K for p in series.values() for K in p}))
ax.legend(handles=[Line2D([], [], color=COLORS[r], marker=MARKERS[r], linewidth=2, markersize=7, label=r, linestyle=LS(r),
                          markerfacecolor=SURFACE if two_level(r) else COLORS[r]) for r in series],
          frameon=False, fontsize=8, labelcolor=INK_2, loc='upper right')

# 2. matched tokens
ax = axes[1]
ends, ys, labs = [], [], []
for r in series:
    if r == 'uniform':
        continue
    ax.plot(Kr, f_at(r, Kr) - f_at('uniform', Kr), color=COLORS[r], linewidth=2, linestyle=LS(r))
    ks = [K for K in measured(r) if K in series['uniform'] and K > 64 or (K == 64 and not two_level(r))]
    d = [series[r][K] - series['uniform'][K] for K in ks]
    ax.plot(ks, d, MARKERS[r], color=COLORS[r], **mk_of(r))
    ys.append(d); labs.append(f'{r} {d[-1]:+.1f}')
ax.axhline(0, color=INK_2, linewidth=1, linestyle=':')
style_axis(ax, 'tokens per image K', 'FID change vs uniform at the same K', 'matched tokens: drop-in replacement (negative = better)')
log2_axis(ax, [64, 100, 160, 256])
end_labels(ax, [160], ys, labs, min_gap=0.07)

# 3. matched FID (random is off scale: -130 to -140%)
ax = axes[2]
ys, labs = [], []
for r in series:
    if base(r) in ('uniform', 'random'):
        continue
    ax.plot(Kr, 100 * (1 - Kr / k_uniform(f_at(r, Kr))), color=COLORS[r], linewidth=2, linestyle=LS(r))
    ks = measured(r)
    pts = [100 * (1 - K / k_uniform(series[r][K])) for K in ks]
    ax.plot(ks, pts, MARKERS[r], color=COLORS[r], **mk_of(r))
    ys.append(pts); labs.append(f'{r} {pts[-1]:+.0f}%')
ax.axhline(0, color=INK_2, linewidth=1, linestyle=':')
style_axis(ax, 'tokens per image K', 'tokens saved vs uniform at the same FID (%)', 'matched FID: compute trade-off (random off scale)')
log2_axis(ax, [64, 100, 160, 256])
end_labels(ax, [160], ys, labs, min_gap=0.07)

fig.suptitle(f'Stitched sampling from uniform {p32}/{p16}/{p8}px JiT models (FID-{n_img / 1000:g}k, one seed): adaptive '
             f'layouts by router vs the uniform layout', color=INK, fontsize=11, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 0.97, 0.93))
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f'wrote {out}')
for r in series:
    print(f"  {r:<11} " + '  '.join(f'K={K}: {v:.2f}' for K, v in series[r].items())
          + (f" | tokens saved at matched FID: " + '  '.join(f'K={K}: {100 * (1 - K / k_uniform(series[r][K])):+.0f}%' for K in measured(r)) if r != 'uniform' else ''))
