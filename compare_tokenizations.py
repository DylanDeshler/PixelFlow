"""
How well each tokenization family can describe real images at a matched number of tokens K, every token with the same
small capacity: a polynomial per colour channel over its region (--capacity const / linear / quad: 1 / 3 / 6
coefficients). No diffusion model: this filters tokenization families before any architecture is built.

  uniform     the quadtree's content-blind coarse-first, centre-out layout (the uniform baseline of analyze_patch_sizes.py)
  quadtree8   aligned squares 32 / 16 / 8 px, greedy splits by fit-error reduction
  quadtree4   the same down to 4 px
  merge       (B) irregular regions grown bottom-up from 2x2 px units: repeatedly merge the adjacent pair whose union
              raises the fit error least, until K regions remain (exact least squares from sufficient statistics)
  gaussaxis   gauss with rotation frozen at 0 (axis-aligned, as GaussJiT's default): what rotation adds
  gaussfixed  soft decoding alone: Gaussians frozen at the quadtree8 layout (sigma = width / 3), only their polynomials fit
  gauss       (C) K soft 2D Gaussians: each pixel blends the tokens' polynomials (in each Gaussian's own frame) with weights
              normalized over tokens (a partition of unity, so every pixel is covered); centres, shapes and coefficients
              fit by Adam from the quadtree8 solution at the same K (its centres, widths and
              region fits as the starting point, so the fit starts near the quadtree's error and can only improve on it)

Each family gets its best allocation for the image itself (fit on the clean image): an upper bound on what each could
represent, not a router. Errors are PSNR on [0, 1] over held-out images.

$ python compare_tokenizations.py --data_dir=/data/in1k_128          # -> {out}.json, {out}.png, {out}_viz_K{K}.png, {out}_viz.npz
$ python compare_tokenizations.py --plot_only=True --out=...          # redraw every figure from {out}.json / {out}_viz.npz
"""

import os
import json
import heapq
import time

import numpy as np
import torch
import torch.nn.functional as F

from data_jit import MemmapImages, to_model_input
from model_quadtree import QuadGeometry, allocate, cell_maps, draw_layout

# -----------------------------------------------------------------------------
data_dir = ''
n_images = 128
seed = 0
tokens = '64,100,160,256'
families = 'uniform,quadtree8,quadtree4,merge,gaussfixed,gaussaxis,gauss'
capacities = 'const,linear,quad'
gauss_steps = 400
gauss_lr = 0.02
batch_size = 16
gauss_batch = 8 # the Gaussian fit holds [batch, K, H*W] weights (and gradients)
viz_images = 4
viz_tokens = '64,100,160' # one patchification figure per K
viz_capacity = 'linear'
out = '' # default: {data_dir name}_tokenizations
plot_only = False # redraw the PSNR figure from an existing {out}.json (no fitting; older JSONs get the PSNR correction)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

assert data_dir or (plot_only and out), 'pass --data_dir (or --plot_only with --out)'
out = out or f"{os.path.basename(os.path.normpath(data_dir))}_tokenizations"
def psnr_figure(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_style import SURFACE, INK, INK_2, style_axis, CATEGORICAL
    caps = [c for c in results['psnr']]
    fams = list(dict.fromkeys(f for c in caps for f in results['psnr'][c]))
    Ks_ = sorted({int(k) for c in caps for f in results['psnr'][c] for k in results['psnr'][c][f]})
    fixed = {'uniform': '#2a78d6', 'quadtree8': '#eb6834', 'quadtree4': '#eda100', 'merge': '#1baf7a', 'gaussfixed': '#e87ba4', 'gaussaxis': '#008300', 'gauss': '#a8336b'}
    marks = {'uniform': 'o', 'quadtree8': 's', 'quadtree4': 'D', 'merge': '^', 'gaussfixed': 'X', 'gaussaxis': 'v', 'gauss': 'P'}
    fig, axes = plt.subplots(1, len(caps), figsize=(5.2 * len(caps), 4.2), facecolor=SURFACE, squeeze=False)
    for ax, cap in zip(axes[0], caps):
        for j, fam in enumerate(fams):
            if fam not in results['psnr'][cap]:
                continue
            row = results['psnr'][cap][fam]
            ks = [K for K in Ks_ if str(K) in row]
            ax.plot(ks, [row[str(K)] for K in ks], '-' + marks.get(fam, 'o'), color=fixed.get(fam, CATEGORICAL[j % len(CATEGORICAL)]),
                    linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=fam)
        ax.set_xscale('log', base=2); ax.set_xticks(Ks_, [str(k) for k in Ks_]); ax.minorticks_off()
        style_axis(ax, 'tokens per image K', 'PSNR (dB, higher is better)', f'{cap} polynomial per token')
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    cfg_ = results.get('config', {})
    fig.suptitle(f"how well each tokenization describes held-out images at matched tokens ({cfg_.get('n_images', '?')} images; each "
                 f"family allocated on the image itself)", color=INK, fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out + '.png', dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f'wrote {out}.png')

def viz_figures(d, psnr, cap):
    # each family's patchification at every budget in d: reconstruction (top) and regions on the image (bottom)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from plot_style import SURFACE, INK
    to_np = lambda a: ((np.clip(a, -1, 1) + 1) / 2).transpose(1, 2, 0)
    Xv = d['X']
    nv, H = Xv.shape[0], Xv.shape[-1]
    Ks_ = sorted({int(k.split('|')[0][1:]) for k in d if '|' in k})
    order = ['uniform', 'quadtree8', 'quadtree4', 'merge', 'gaussfixed', 'gaussaxis', 'gauss']
    written = []
    for K in Ks_:
        fams = [f for f in order if f'K{K}|{f}|rec' in d]
        cols = ['image'] + fams
        fig, axes = plt.subplots(2 * nv, len(cols), figsize=(1.9 * len(cols), 3.9 * nv + 0.6), facecolor=SURFACE, squeeze=False)
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        for j, c in enumerate(cols):
            for i in range(nv):
                top, bot = axes[2 * i, j], axes[2 * i + 1, j]
                if c == 'image':
                    top.imshow(to_np(Xv[i])); bot.imshow(to_np(Xv[i]))
                    if i == 0:
                        top.set_title('image', color=INK, fontsize=8)
                    continue
                key = f'K{K}|{c}|'
                top.imshow(to_np(d[key + 'rec'][i]))
                bot.imshow(to_np(d[key + 'overlay'][i]))
                if c.startswith('gauss'): # owned cells (in the overlay) and each Gaussian's centre
                    mu = d[key + 'mu'][i]
                    bot.scatter(mu[:, 0] - 0.5, mu[:, 1] - 0.5, s=2, color='#a8336b', linewidths=0)
                    bot.set_xlim(-0.5, H - 0.5); bot.set_ylim(H - 0.5, -0.5)
                if i == 0:
                    db = psnr.get(cap, {}).get(c, {}).get(str(K))
                    top.set_title(f"{c}" + (f"\n{db:.1f} dB" if db is not None else ''), color=INK, fontsize=8)
        fig.suptitle(f'K={K}, {cap} polynomial per token: reconstruction (top), regions (bottom)\n'
                     f'quadtree: boxes by size; merge: region boundaries; gauss: the cell each Gaussian dominates, and its centre',
                     color=INK, fontsize=10, x=0.01, ha='left')
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / (3.9 * nv + 0.6)))
        fig.savefig(out + f'_viz_K{K}.png', dpi=140, facecolor=SURFACE)
        plt.close(fig)
        written.append(out + f'_viz_K{K}.png')
    return written

if plot_only:
    with open(out + '.json') as fh:
        res = json.load(fh)
    if res.get('psnr_version', 1) < 2: # older results divided by pixels, not pixels x channels: +10 log10(3) dB
        res['psnr'] = {c: {f: {k: v + 10 * np.log10(3) for k, v in r.items()} for f, r in d.items()} for c, d in res['psnr'].items()}
        res['psnr_version'] = 2
        with open(out + '.json', 'w') as fh:
            json.dump(res, fh, indent=1)
        print(f'{out}.json: PSNR corrected (+{10 * np.log10(3):.2f} dB)')
    psnr_figure(res)
    if os.path.exists(out + '_viz.npz'):
        print('wrote ' + ', '.join(viz_figures(dict(np.load(out + '_viz.npz')), res['psnr'], res.get('config', {}).get('viz_capacity', 'linear'))))
    else:
        print(f'no {out}_viz.npz (results from before it was saved): rerun with a few images for the patch figures, e.g. '
              f'--n_images=8 --capacities=linear')
    raise SystemExit

data = MemmapImages(os.path.join(data_dir, 'val') if os.path.isdir(os.path.join(data_dir, 'val')) else data_dir)
img = data.img_size
idx = np.sort(np.random.default_rng(seed).choice(len(data), n_images, replace=False))
X = to_model_input(data.get(idx)[0], flip=False, device=device) # [N, 3, H, W] in [-1, 1]
Ks = [int(k) for k in tokens.split(',')]
fam_list = [f for f in families.split(',') if f]
FAMILIES = ('uniform', 'quadtree8', 'quadtree4', 'merge', 'gaussfixed', 'gaussaxis', 'gauss')
assert all(f in FAMILIES for f in fam_list), f'unknown families {[f for f in fam_list if f not in FAMILIES]}: choose from {FAMILIES}'
cap_list = [c for c in capacities.split(',') if c]
DEG = {'const': 0, 'linear': 1, 'quad': 2}

def basis(u, v, cap):
    # polynomial basis [..., d] of (normalized) coordinates u, v
    terms = [torch.ones_like(u)]
    if DEG[cap] >= 1:
        terms += [u, v]
    if DEG[cap] >= 2:
        terms += [u * u, u * v, v * v]
    return torch.stack(terms, -1)

def psnr(err_sum, n_pix):
    # squared error summed over [-1, 1] pixels and the 3 channels -> PSNR on [0, 1] (mean over channels)
    return 10 * np.log10(1.0 / (err_sum / (3 * n_pix) / 4))

# -----------------------------------------------------------------------------
# quadtree (and uniform): per-node least squares, greedy allocation by error reduction

def quadtree_fits(x, geo, cap):
    """
    Per level: the fitted image (each node's polynomial fit to its pixels) [B, 3, H, W] and the per-finest-cell squared
    error of that level's fit [B, G, G]. The basis is in node-local coordinates (the same span as global ones).
    """
    B = x.shape[0]
    G = geo.grids[-1]
    recon, cell_err = [], []
    for p in geo.patches:
        r = (torch.arange(p, device=x.device, dtype=torch.float32) + 0.5) / p * 2 - 1
        u, v = torch.meshgrid(r, r, indexing='ij')
        Phi = basis(u.flatten(), v.flatten(), cap) # [p*p, d]
        P = Phi @ torch.linalg.pinv(Phi) # projection onto the span [p*p, p*p]
        blocks = F.unfold(x, p, stride=p).view(B, 3, p * p, -1) # [B, 3, p*p, nodes]
        fit = torch.einsum('ij,bcjn->bcin', P, blocks)
        rec = F.fold(fit.reshape(B, 3 * p * p, -1), x.shape[-1], p, stride=p)
        recon.append(rec)
        e = (rec - x).pow(2).sum(1, keepdim=True) # [B, 1, H, W]
        f = x.shape[-1] // G
        cell_err.append(F.avg_pool2d(e, f)[:, 0] * f * f)
    return recon, cell_err

def node_sums(E, geo, level):
    k = geo.grids[-1] // geo.grids[level]
    return (F.avg_pool2d(E.unsqueeze(1), k)[:, 0] * k * k).flatten(1)

def quadtree_layout(cell_err, geo, K, adaptive=True):
    if adaptive:
        gains = torch.cat([node_sums(cell_err[l], geo, l) - node_sums(cell_err[l + 1], geo, l) for l in range(geo.L - 1)], 1)
    else: # coarse-first, centre-out (analyze_patch_sizes.py's uniform)
        centre = geo.grids[-1] / 2
        dist = torch.sqrt((geo.cy - centre) ** 2 + (geo.cx - centre) ** 2)[:geo.n_split]
        gains = ((geo.L - 1 - geo.level[:geo.n_split]).float() * 1e3 - dist).to(cell_err[0].device).expand(cell_err[0].shape[0], -1)
    return allocate(gains, K, geo, geo.children.to(gains.device), geo.parent.to(gains.device))

def run_quadtree(x, cap, K, leaf, adaptive=True):
    geo = QuadGeometry(img, leaf, int(np.log2(32 // leaf)) + 1) # 32px roots
    recon, cell_err = quadtree_fits(x, geo, cap)
    layout = quadtree_layout(cell_err, geo, K, adaptive)
    lev = cell_maps(layout, geo)[1]
    f = img // geo.grids[-1]
    lev_px = lev.repeat_interleave(f, 1).repeat_interleave(f, 2).unsqueeze(1)
    rec = torch.zeros_like(x)
    for l in range(geo.L):
        rec = torch.where(lev_px == l, recon[l], rec)
    return rec, (layout, geo)

# -----------------------------------------------------------------------------
# B: bottom-up merging of 2x2 units (exact least squares from per-region sufficient statistics)

def run_merge(x, cap, Ks_desc):
    """
    One image [3, H, W] -> {K: (reconstruction, region labels [H, W])} for every K in Ks_desc (descending), merging
    greedily from 2x2 units: the adjacent pair whose union raises the summed fit error least.
    """
    H = x.shape[-1]
    unit = 2
    g = H // unit
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(H, dtype=torch.float64), indexing='ij')
    Phi = basis((xx + 0.5) / H * 2 - 1, (yy + 0.5) / H * 2 - 1, cap).double() # global coordinates [H, W, d]
    Y = x.double().cpu().permute(1, 2, 0) # [H, W, 3]
    d = Phi.shape[-1]
    # unit statistics: Gram [d, d], cross [d, 3], sum of squares, pixel count
    Pu = Phi.view(g, unit, g, unit, d).permute(0, 2, 1, 3, 4).reshape(g * g, unit * unit, d)
    Yu = Y.view(g, unit, g, unit, 3).permute(0, 2, 1, 3, 4).reshape(g * g, unit * unit, 3)
    Gm = (Pu.transpose(1, 2) @ Pu).numpy()
    Cm = (Pu.transpose(1, 2) @ Yu).numpy()
    Sm = (Yu ** 2).sum((1, 2)).numpy()
    ridge = 1e-8 * np.eye(d)
    def err(Gr, Cr, Sr):
        coef = np.linalg.solve(Gr + ridge, Cr)
        return Sr - float((coef * Cr).sum())
    n = g * g
    G_, C_, S_ = list(Gm), list(Cm), list(Sm)
    E_ = [err(G_[i], C_[i], S_[i]) for i in range(n)]
    alive = [True] * n
    ver = [0] * n
    nbrs = [set() for _ in range(n)]
    for r in range(g):
        for c in range(g):
            i = r * g + c
            if c + 1 < g: nbrs[i].add(i + 1); nbrs[i + 1].add(i)
            if r + 1 < g: nbrs[i].add(i + g); nbrs[i + g].add(i)
    parent = list(range(n))
    heap = []
    def push(a, b):
        cost = err(G_[a] + G_[b], C_[a] + C_[b], S_[a] + S_[b]) - E_[a] - E_[b]
        heapq.heappush(heap, (cost, a, b, ver[a], ver[b]))
    for a in range(n):
        for b in nbrs[a]:
            if a < b:
                push(a, b)
    count, snaps = n, {}
    targets = sorted(Ks_desc, reverse=True)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def snapshot():
        labels = np.array([find(i) for i in range(n)]).reshape(g, g)
        lab = torch.from_numpy(labels).repeat_interleave(unit, 0).repeat_interleave(unit, 1)
        rec = torch.zeros(H, H, 3, dtype=torch.float64)
        for root in np.unique(labels):
            m = lab == int(root)
            coef = np.linalg.solve(G_[root] + ridge, C_[root])
            rec[m] = Phi[m] @ torch.from_numpy(coef)
        return rec.permute(2, 0, 1).float(), lab
    ti = 0
    while ti < len(targets) and targets[ti] >= count:
        snaps[targets[ti]] = snapshot(); ti += 1
    while heap and ti < len(targets):
        cost, a, b, va, vb = heapq.heappop(heap)
        if not (alive[a] and alive[b]) or ver[a] != va or ver[b] != vb:
            continue
        # merge b into a
        G_[a] = G_[a] + G_[b]; C_[a] = C_[a] + C_[b]; S_[a] = S_[a] + S_[b]
        E_[a] = err(G_[a], C_[a], S_[a])
        alive[b] = False; parent[b] = a; ver[a] += 1
        for nb in nbrs[b]:
            if nb != a:
                nbrs[nb].discard(b); nbrs[nb].add(a); nbrs[a].add(nb)
        nbrs[a].discard(b); nbrs[b] = set()
        for nb in nbrs[a]:
            push(min(a, nb), max(a, nb))
        count -= 1
        while ti < len(targets) and targets[ti] >= count:
            snaps[targets[ti]] = snapshot(); ti += 1
    return snaps

# -----------------------------------------------------------------------------
# C: K soft 2D Gaussians with per-token polynomials, fit by Adam from the quadtree8 layout

def run_gauss(x, cap, K, steps=None, lr=None, fixed=False, rotate=True):
    """
    x [B, 3, H, W] -> reconstruction, (centres [B, K, 2], covariance factors) after fitting. Pixel value
    sum_k w_k(p) poly_k(p - mu_k) / sum_k w_k(p), w_k = exp(-|L_k^-1 (p - mu_k)|^2 / 2). fixed: the geometry stays at the
    quadtree layout (sigma = width / 3) and only the polynomials are fit: soft decoding without adaptive geometry.
    """
    steps, lr = steps or gauss_steps, lr or gauss_lr
    B, _, H, W = x.shape
    layout, geo = run_quadtree(x, cap, K, 8)[1]
    f = img // geo.grids[-1] # finest-unit size in pixels
    cy = geo.cy.to(x.device)[layout] * f # [B, K] pixel centres
    cx = geo.cx.to(x.device)[layout] * f
    w0 = geo.width.to(x.device)[layout] * f # token widths in pixels
    mu = torch.stack([cx, cy], -1).clone().requires_grad_(not fixed) # [B, K, 2] (x, y)
    # narrow enough that the normalized weights start close to the quadtree's hard partition
    log_s = torch.log(w0 / (3 if fixed else 4)).unsqueeze(-1).repeat(1, 1, 2).clone().requires_grad_(not fixed) # per-axis scales
    rho = torch.zeros(B, K, device=x.device, requires_grad=not fixed and rotate) # rotation
    # each token's polynomial starts at its quadtree region's least-squares fit (node-local coordinates, u = rows, v = columns)
    lvl, pos = geo.level.to(x.device)[layout], geo.index_in_level.to(x.device)[layout] # [B, K]
    coef0 = None
    for l, p in enumerate(geo.patches):
        r = (torch.arange(p, device=x.device, dtype=torch.float32) + 0.5) / p * 2 - 1
        u, v = torch.meshgrid(r, r, indexing='ij')
        pinv = torch.linalg.pinv(basis(u.flatten(), v.flatten(), cap)) # [d, p*p]
        c_l = torch.einsum('dj,bcjn->bndc', pinv, F.unfold(x, p, stride=p).view(B, 3, p * p, -1)) # [B, nodes, d, 3]
        g = c_l.gather(1, pos.clamp(max=c_l.shape[1] - 1)[..., None, None].expand(-1, -1, *c_l.shape[2:]))
        coef0 = g if coef0 is None else torch.where((lvl == l)[..., None, None], g, coef0)
        if l == 0:
            coef0 = torch.where((lvl == l)[..., None, None], g, torch.zeros_like(g))
    coef = coef0.clone().requires_grad_(True)
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device) + 0.5, torch.arange(W, device=x.device) + 0.5, indexing='ij')
    P = torch.stack([xx.flatten(), yy.flatten()], -1) # [HW, 2]
    target = x.flatten(2).transpose(1, 2) # [B, HW, 3]
    groups = [{'params': [coef], 'lr': lr * 5}] + ([] if fixed else [{'params': [mu], 'lr': lr * 8}, {'params': [log_s] + ([rho] if rotate else []), 'lr': lr}])
    opt = torch.optim.Adam(groups)
    def render():
        diff = P[None, None] - mu[:, :, None] # [B, K, HW, 2]
        c, s = torch.cos(rho)[..., None], torch.sin(rho)[..., None]
        rx = c * diff[..., 0] + s * diff[..., 1] # rotate into each Gaussian's frame
        ry = -s * diff[..., 0] + c * diff[..., 1]
        sx, sy = log_s[..., 0:1].exp(), log_s[..., 1:2].exp()
        q = (rx / sx) ** 2 + (ry / sy) ** 2
        logw = -0.5 * q
        wn = torch.softmax(logw, dim=1) # normalized over tokens: a partition of unity [B, K, HW]
        scale = (w0[..., None] / 2) # local coordinates in units of the initial half-width (rows, columns: as the quadtree fits)
        phi = basis(ry / scale, rx / scale, cap) # [B, K, HW, d]
        val = torch.einsum('bkpd,bkdc->bkpc', phi, coef)
        return (wn[..., None] * val).sum(1) # [B, HW, 3]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = (render() - target).pow(2).sum(-1).mean()
        loss.backward()
        opt.step(); sched.step()
    with torch.no_grad():
        rec = render().transpose(1, 2).reshape(B, 3, H, W)
    return rec, (mu.detach(), log_s.detach().exp(), rho.detach())

# -----------------------------------------------------------------------------
# run

results = {'config': config, 'psnr': {}, 'psnr_version': 2}
recs = {} # K -> family -> (reconstructions, regions) for the patchification figures
VIZ_K = [int(k) for k in str(viz_tokens).split(',') if k]
t0 = time.time()
for cap in cap_list:
    for fam in fam_list:
        for K in Ks:
            err_sum = 0.0
            bs = gauss_batch if fam.startswith('gauss') else batch_size
            for i in range(0, n_images, bs):
                xb = X[i:i + bs]
                if fam in ('uniform', 'quadtree8', 'quadtree4'):
                    rec, extra = run_quadtree(xb, cap, K, 4 if fam == 'quadtree4' else 8, adaptive=(fam != 'uniform'))
                elif fam == 'merge':
                    continue # done below for all K at once
                else:
                    with torch.enable_grad():
                        rec, extra = run_gauss(xb, cap, K, fixed=(fam == 'gaussfixed'), rotate=(fam != 'gaussaxis'))
                err_sum += (rec - xb).pow(2).sum().item()
                if i == 0 and cap == viz_capacity and K in VIZ_K:
                    recs.setdefault(K, {})[fam] = (rec[:viz_images].cpu(), extra)
            if fam != 'merge':
                results['psnr'].setdefault(cap, {}).setdefault(fam, {})[str(K)] = psnr(err_sum, n_images * img * img)
        if fam == 'merge':
            errs = {K: 0.0 for K in Ks}
            for j in range(n_images):
                snaps = run_merge(X[j], cap, Ks)
                for K in Ks:
                    errs[K] += (snaps[K][0].to(device) - X[j]).pow(2).sum().item()
                if j < viz_images and cap == viz_capacity:
                    for K in VIZ_K:
                        if K in snaps:
                            m = recs.setdefault(K, {}).setdefault('merge', ([], []))
                            m[0].append(snaps[K][0]); m[1].append(snaps[K][1])
            for K in Ks:
                results['psnr'].setdefault(cap, {}).setdefault('merge', {})[str(K)] = psnr(errs[K], n_images * img * img)
        row = results['psnr'][cap][fam]
        print(f"{cap:<6} {fam:<10} " + '  '.join(f'K={K}: {row[str(K)]:.2f} dB' for K in Ks) + f'   ({time.time() - t0:.0f}s)')

with open(out + '.json', 'w') as fh:
    json.dump(results, fh, indent=1)
print(f'wrote {out}.json')

# -----------------------------------------------------------------------------
# figures

psnr_figure(results)

def portable_viz():
    # the figure data as numpy arrays (saved to {out}_viz.npz, so --plot_only can redraw the patch figures): images, each
    # family's reconstruction, region overlays drawn on the image (quadtree boxes, merge boundaries) or Gaussian parameters
    nv = min(viz_images, n_images)
    d = {'X': X[:nv].float().cpu().numpy()}
    for K, rk in recs.items():
        for fam, val in rk.items():
            key = f'K{K}|{fam}|'
            if fam == 'merge':
                rec = torch.stack(val[0][:nv]); labs = torch.stack(val[1][:nv])
                edge = torch.zeros(labs.shape, dtype=torch.bool)
                edge[:, :, :-1] |= labs[:, :, :-1] != labs[:, :, 1:]; edge[:, :-1] |= labs[:, :-1] != labs[:, 1:]
                over = X[:nv].float().cpu().clone()
                over = torch.where(edge[:, None], torch.tensor([-0.78, 0.38, -0.04]).view(1, 3, 1, 1), over)
                d[key + 'rec'], d[key + 'overlay'] = rec.numpy(), over.numpy()
            elif fam.startswith('gauss'):
                rec, (mu, sc, rho) = val
                mu, sc, rho = mu[:nv].float().cpu(), sc[:nv].float().cpu(), rho[:nv].float().cpu()
                d[key + 'rec'], d[key + 'mu'], d[key + 's'], d[key + 'rho'] = rec[:nv].cpu().numpy(), mu.numpy(), sc.numpy(), rho.numpy()
                # each token's owned cell: the pixels where its normalized weight is largest (the weights are normalized
                # over all tokens, so these cells tile the image whatever the Gaussians' widths); boundaries drawn
                Hh = X.shape[-1]
                r = torch.arange(Hh, dtype=torch.float32) + 0.5
                py, px = torch.meshgrid(r, r, indexing='ij')
                dx, dy = px - mu[..., 0, None, None], py - mu[..., 1, None, None] # [nv, K, H, W]
                c, sn = torch.cos(rho)[..., None, None], torch.sin(rho)[..., None, None]
                q = ((c * dx + sn * dy) / sc[..., 0, None, None]) ** 2 + ((-sn * dx + c * dy) / sc[..., 1, None, None]) ** 2
                owner = (-q).argmax(1) # [nv, H, W]
                edge = torch.zeros(owner.shape, dtype=torch.bool)
                edge[:, :, :-1] |= owner[:, :, :-1] != owner[:, :, 1:]; edge[:, :-1] |= owner[:, :-1] != owner[:, 1:]
                over = torch.where(edge[:, None], torch.tensor([0.32, -0.6, -0.16]).view(1, 3, 1, 1), X[:nv].float().cpu().clone())
                d[key + 'overlay'] = over.numpy()
            else:
                rec, (layout, geo) = val
                d[key + 'rec'] = rec[:nv].cpu().numpy()
                d[key + 'overlay'] = draw_layout(X[:nv].float().cpu(), layout[:nv].cpu(), geo).numpy()
    return d

viz_data = portable_viz()
np.savez_compressed(out + '_viz.npz', **viz_data)
written = viz_figures(viz_data, results['psnr'], viz_capacity)
print(f'wrote {out}_viz.npz, ' + ', '.join(written))
