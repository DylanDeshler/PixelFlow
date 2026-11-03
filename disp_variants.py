"""
Dispersive Loss variants for pixel-space JiT (prototypes; nothing in the repo imports this module). See DISPERSIVE_ANALYSIS.md
for the reasoning and the experiment plan. Run `python disp_variants.py` for the CPU self-tests.

Why the repo's forms fail here (repa.dispersive_loss):
    infonce_l2   translation-invariant but not scale-invariant: the diffusion loss is ~invariant to the scale of block
                 outputs (every consumer is normalized), so normalized optimizers (Muon, AdamW) inflate the features along
                 the one direction the diffusion loss has no gradient in, until every pair is "far" and the term pins at
                 log(1/B)
    infonce_l2n  scale-invariant but no longer translation-invariant: per-sample RMS includes the batch-common component
                 (pooled block outputs of the best run have pairwise cosine ~0.985), so almost all of its range (-0.06 ->
                 -3.94 of the -3.95 optimum) is bought by removing the batch mean; once centered, isotropic noise, a
                 128-d random code and a class-hierarchical code all score within ~0.06 of the optimum, so the term has no
                 further signal about structure
Any function of one noisy view's pairwise distances that is minimized by a spread-out batch is also minimized by
high-dimensional per-sample noise, because noise enters only through the norms (the diagonal of the Gram matrix): for i != j,
E_noise <f_i, f_j> = <mu_i, mu_j> with mu = E_e f(x, e, t). The variants here:

    infonce_l2s     the paper's InfoNCE-L2 on batch-standardized features (batch mean removed, divided by the batch's RMS):
                    translation- and scale-invariant, exactly the paper's loss when the features already have zero mean and unit
                    variance. Single view, so the noise shortcut remains (as it does in latent SiT). Optional per-sample weights
                    (e.g. snr_weight(t)) and groups (the paper's per-device batch of 16 / 32).
    xview_infonce   noise-debiased InfoNCE-L2 from two views of each image (independent noise, and independent t when the
                    views come from paired_views + flow_loss): distances from cross-view products only, so in expectation the
                    loss sees the dispersion of the noise-averaged (and t-averaged) features; per-sample noise and t-specific
                    directions earn nothing, and their variance raises the loss (exp is convex; the mean over n^2 pairs
                    concentrates, so the log doesn't undo it).
    xview_rank      effective rank (participation ratio) of the cross-view covariance: VICReg-style decorrelation of the
                    noise-free feature covariance, a dimension-wise rather than sample-wise dispersion.
    xview_distill   asymmetric self-distillation across noise levels: the noisier view predicts the cleaner view's features
                    (stop-gradient) through a small head; the alignment half the Dispersive Loss leaves out (SRA-like).
    contrastive_x_loss  Contrastive Flow Matching's negative term in x-prediction units (a dispersion of the predictions).
    SublayerCapture hook on a block's MLP output instead of the block output: closes the AttnRes routing loophole (a loss on
                    a block output can be met by mixing in the raw patch embedding).
    feature_report  diagnostics (no training): batch-common share, noise share, label share, effective ranks and loss values
                    per block, from two noise draws of the same images.

Everything is computed in fp32 with autocast off (distances from norms minus dot products cancel badly in bf16).
"""

import math

import torch
import torch.nn.functional as F


def _pool(hidden, pool):
    # [B, T, C] -> [B, d]: 'mean' over tokens (what the probes read) or 'none' (tokens flattened, the paper)
    return hidden.float().mean(1) if pool == 'mean' else hidden.float().flatten(1)


def snr_weight(t, t0=0.5, width=0.5):
    """
    Per-sample weight in [0, 1] that switches a dispersive term on where the input still carries the image: a sigmoid in
    logit(t) centered at t0 (t = 1 is clean). With JiT's logit-normal(-0.8, 0.8), t0 = 0.5 keeps ~16% of a batch at weight
    > 0.5 (t0 = 0.4: ~31%). Detached: the weight is a property of the sample, not something to optimize.
    """
    lt = torch.log(t.float().clamp(1e-6, 1 - 1e-6) / (1 - t.float().clamp(1e-6, 1 - 1e-6)))
    return torch.sigmoid((lt - math.log(t0 / (1 - t0))) / width).detach()


def _weighted_lse(logits, w):
    # log( sum_ij w_i w_j exp(logits_ij) / sum_ij w_i w_j ) over the last two dims; w [..., B] or None
    if w is None:
        n = logits.shape[-1]
        return torch.logsumexp(logits.flatten(-2), -1) - math.log(n * n)
    ww = (w[..., :, None] * w[..., None, :]).clamp_min(1e-30)
    return torch.logsumexp((logits + ww.log()).flatten(-2), -1) - torch.log(ww.flatten(-2).sum(-1))


def infonce_l2s(hidden, tau=0.5, pool='none', weights=None, groups=1):
    """
    InfoNCE-L2 (Wang & He 2025) on batch-standardized features: z <- (z - mean_batch z) / rms_batch(z - mean_batch z), one
    scalar RMS over the batch and all dimensions, gradient kept through both (detaching either would bring back a radial or
    a common-mode gradient). Then D_ij = ||z_i - z_j||^2 / d and log mean exp(-D / tau) over the full B x B matrix, as the
    paper. Invariant to scale and to adding any vector shared by the batch; for features already at zero mean and unit RMS it
    is exactly repa.dispersive_loss(kind='infonce_l2'). Isotropic features give D ~ 2 for every pair, so tau keeps the meaning
    it has for unit-variance features (tau = 0.5: a typical pair contributes e^-4).
    hidden [B, T, C] -> scalar. weights: optional per-sample weights [B] (e.g. snr_weight(t)), pairs weighted w_i w_j.
    groups: compute the loss within groups of B / groups consecutive samples and average (the paper ran it per device on 16
    or 32 samples; at tau 0.5 the self term then takes ~78% of the softmax mass, i.e. ~4.5x less repulsion than one global
    batch of 1024, with the same split of repulsion between pair types). Standardization uses the whole batch.
    Scale-invariant losses have gradients ~ 1 / feature scale; the 1e-8 floor on the mean square bounds them for features
    that start at exactly zero (e.g. zero-initialized sublayer outputs, SublayerCapture).
    """
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        z = _pool(hidden, pool)
        B, d = z.shape
        # batch mean and RMS (weighted, so zero-weight samples drop out entirely)
        wn = weights.float() / weights.float().sum().clamp_min(1e-12) if weights is not None else torch.full((B,), 1.0 / B, device=z.device)
        zc = z - (wn[:, None] * z).sum(0, keepdim=True)
        zc = zc * torch.rsqrt((wn * zc.pow(2).mean(1)).sum() + 1e-8) # floor: features below RMS ~1e-4 fade out instead of 0 / 0
        G = B // groups
        assert G * groups == B, f'batch {B} not divisible into {groups} groups'
        zg = zc.view(groups, G, d)
        sq = zg.pow(2).sum(-1)
        eye = torch.eye(G, device=z.device)
        D = (sq[:, :, None] + sq[:, None, :] - 2 * zg @ zg.transpose(1, 2)).clamp_min(0) / d * (1 - eye)
        w = weights.float().view(groups, G) if weights is not None else None
        return _weighted_lse(-D / tau, w).mean()


def paired_views(X, Y):
    """
    Two views per image for the cross-view losses: the first half of the batch, twice ([x_0..x_{n-1}, x_0..x_{n-1}]).
    masking.flow_loss then draws independent noise and an independent t for every row, so rows i and i + n are the same image
    under independent noise at independent noise levels. Halves the distinct images per step (same compute).
    """
    n = X.shape[0] // 2
    return torch.cat([X[:n], X[:n]]), torch.cat([Y[:n], Y[:n]])


def split_views(hidden):
    # the two views from a paired batch (rows i and i + n), e.g. a DispCapture's .out
    n = hidden.shape[0] // 2
    return hidden[:n], hidden[n:2 * n]


def _xview_gram(h_a, h_b, pool):
    a, b = _pool(h_a, pool), _pool(h_b, pool)
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    return a, b


def xview_infonce(h_a, h_b, tau=0.5, pool='mean', norm='cross', weights=None, min_share=0.1):
    """
    Noise-debiased InfoNCE-L2 from two views per image (h_a[i], h_b[i]: image i under independent noise, and with paired_views
    independent t). With K = A B^T over centered features (cross-view products only),
        D_ij = (K_ii + K_jj - K_ij - K_ji) / (d v),
    an unbiased estimate of ||mu_i - mu_j||^2 / (d v), mu = the features averaged over the views' noise (and t). Each view's
    own noise appears only in its norm, which never enters D, so noise can't lower the loss in expectation; it only adds
    variance to D, which raises mean exp(-D / tau) (exp is convex) and so the loss (the mean over n^2 pairs concentrates). With independent t per view, a direction that depends on t but not on the
    image cancels in expectation as well; directions that depend on the label (shared by both views) do count.
    norm: v, the scale reference (scale-invariant either way)
      'cross'  (default) the cross-view covariance's mean diagonal (the noise-free variance): neutral to noise and to
               t-only directions in expectation. The estimator's noise grows as noise variance / content variance, and with
               almost no shared content D / v blows up (a 100:1 noise-to-content batch gives losses ~ +75), so v is floored
               at min_share x the 'total' value: below that content share the loss behaves like 'total' (rewards content)
      'total'  the views' mean per-dimension variance, noise and t-driven variance included: they shrink every D, so the
               loss also rewards pooled features whose variance is mostly image content, i.e. an added invariance pressure
               toward noise- and t-independent features (the direction the paper's positive-pair variants took, which hurt)
    D_ii = 0 exactly; logsumexp over the full matrix as the paper. weights: optional per-pair-row weights [n] (e.g. the product
    of the two views' snr_weight). h_a, h_b [n, T, C] -> scalar. Floor log(1/n); optimum ~ log((1 + (n - 1) e^(-2/tau)) / n)
    with noise-free, mutually orthogonal mu.
    """
    with torch.autocast(device_type=h_a.device.type, enabled=False):
        a, b = _xview_gram(h_a, h_b, pool)
        n, d = a.shape
        K = a @ b.T
        k = K.diagonal()
        tot = 0.5 * (a.pow(2).mean() + b.pow(2).mean()) + 1e-8 # floor as in infonce_l2s
        if norm == 'total':
            v = tot
        elif norm == 'cross':
            v = torch.maximum(k.mean() / d, min_share * tot)
        else:
            raise ValueError(f"norm must be 'total' or 'cross', got {norm}")
        eye = torch.eye(n, device=a.device)
        D = (k[:, None] + k[None, :] - K - K.T) / (d * v) * (1 - eye)
        return _weighted_lse(-D / tau, weights)


def xview_rank(h_a, h_b, pool='mean'):
    """
    -log of the normalized effective rank of the noise-free feature covariance: C = A^T B / n over centered views (each
    view's noise averages out of the cross product), C_s its symmetric part,
        L = log(d ||C_s||_F^2 / (tr C)^2),
    0 for an isotropic covariance (participation ratio d), log d for rank one. Scale- and translation-invariant, computed
    through n x n Gram matrices (no d x d matrix). tr C is clamped at 1e-3 x the views' total variance, so features with no
    shared content get a large loss rather than a division by ~0. A decorrelation (VICReg covariance-term) analogue: it
    spreads variance across dimensions instead of pushing samples apart, so it has no per-sample shortcut at all, but by
    itself it doesn't ask for more content in the features (only for content spread over more directions).
    """
    with torch.autocast(device_type=h_a.device.type, enabled=False):
        a, b = _xview_gram(h_a, h_b, pool)
        n, d = a.shape
        Ga, Gb, M = a @ a.T, b @ b.T, a @ b.T
        fro_C = (Ga * Gb).sum() / n ** 2 # ||A^T B||_F^2 / n^2
        tr_CC = (M * M.T).sum() / n ** 2 # tr(C C)
        fro_Cs = 0.5 * (fro_C + tr_CC) # ||(C + C^T) / 2||_F^2
        tr = M.diagonal().sum() / n
        tot = 0.5 * (a.pow(2).sum() + b.pow(2).sum()) / n
        tr = torch.maximum(tr, 1e-3 * tot)
        return torch.log(d * fro_Cs) - 2 * torch.log(tr)


class DistillHead(torch.nn.Module):
    """
    Predictor for xview_distill: identity plus a zero-initialized 2-layer MLP (so at init the loss compares the features
    directly, and the head learns only what the noisier view can't carry as is).
    """
    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or 2 * dim
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, dim))
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return x + self.mlp(x)


def xview_distill(h_a, h_b, t_a, t_b, head=None, pool='mean', min_gap=0.0):
    """
    Asymmetric self-distillation across noise levels (the 'alignment' half that the Dispersive Loss leaves out, as in SRA,
    Jiang et al. 2025, but with the online network's own cleaner view as the target, no EMA copy): for each pair from
    paired_views, the noisier view (smaller t) predicts the cleaner view's features, stop-gradient, 1 - cosine after centering
    both over the batch (so the batch-common component can't satisfy it). Only the noisy side is pulled, toward what the
    network computes when it sees more of the image: a denoising objective in feature space, not invariance (the paper's
    symmetric 'contrastive, independent noise' variant, which forces the clean view to forget its details too, was 20% worse
    than no regularizer). min_gap: skip pairs whose logit-t gap is below it. Can collapse on its own (constant features
    satisfy it), so pair it with a dispersive term (xview_infonce) or rely on the diffusion loss to keep content.
    h_a, h_b [n, T, C]; t_a, t_b [n]; head: e.g. DistillHead(C) for pool='mean'. -> scalar.
    """
    with torch.autocast(device_type=h_a.device.type, enabled=False):
        a, b = _pool(h_a, pool), _pool(h_b, pool)
        a_clean = (t_a >= t_b)[:, None]
        noisy = torch.where(a_clean, b, a)
        clean = torch.where(a_clean, a, b).detach()
        pred = head(noisy) if head is not None else noisy
        pred = pred - pred.mean(0, keepdim=True)
        clean = clean - clean.mean(0, keepdim=True)
        cos = F.cosine_similarity(pred, clean, dim=1)
        lt = lambda t: torch.log(t.clamp(1e-6, 1 - 1e-6) / (1 - t.clamp(1e-6, 1 - 1e-6)))
        keep = ((lt(t_a) - lt(t_b)).abs() >= min_gap).float()
        return ((1 - cos) * keep).sum() / keep.sum().clamp_min(1)


def contrastive_x_loss(x_pred, x, t, t_eps=0.05, perm=None):
    """
    Contrastive Flow Matching's negative term (Stoica et al. 2025, 'Delta FM') in JiT's x-prediction / v-loss units: the
    v-loss residual against another image of the batch, mean over pixels, to be SUBTRACTED with a small weight lam:
    loss = v_loss - lam * contrastive_x_loss. A dispersion of the predictions rather than of the features. Per noisy input the
    combined objective E||x_pred - x||^2 - lam E||x_pred - x'||^2 is minimized at x_pred = m + (E[x | z] - m) / (1 - lam),
    m = E[x'] the data mean: the posterior mean pushed away from the mean image by 1 / (1 - lam), a built-in guidance against
    an unconditional 'mean' model, not representation learning. perm: the batch permutation (default: roll by one).
    """
    perm = perm if perm is not None else torch.roll(torch.arange(x.shape[0], device=x.device), 1)
    tt = t.view(-1, *([1] * (x.ndim - 1)))
    return ((x[perm] - x_pred) / (1 - tt).clamp_min(t_eps)).float().pow(2).mean()


class SublayerCapture:
    """
    Forward hook on block `depth`'s MLP (1-indexed; the ungated SwiGLU output, [B, T, C]), an alternative to repa.DispCapture
    for AttnRes models. With full AttnRes a block's output is a softmax mix over every earlier source, the patch embedding
    included, and is read only by the next attention: a loss on it can be satisfied by re-weighting the mix toward the
    embedding (the raw noisy pixels, whose scale is free: x_embedder is on AdamW, no weight decay), which is what the
    infonce_l2 runs did (mixing weight on the embedding ~1 at the regularized block). The MLP output is what the block itself
    computes and every later mix can read. In-context tokens dropped as in DispCapture. Under torch.compile, check once that
    the captured tensor matches an eager forward (the hook sits inside the compiled mlp_branch).
    """
    def __init__(self, net, depth):
        self.n_strip = net.in_context_len if (net.in_context_len > 0 and depth - 1 >= net.in_context_start) else 0
        self.out = None
        net.blocks[depth - 1].mlp.register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        self.out = output[:, self.n_strip:]


def effective_rank(z):
    # participation ratio of the centered covariance of z [n, d]: (sum lambda)^2 / sum lambda^2, in [1, min(n, d)]
    zc = z.float() - z.float().mean(0, keepdim=True)
    G = zc @ zc.T
    return (G.diagonal().sum() ** 2 / G.pow(2).sum()).item()


@torch.no_grad()
def feature_report(f1, f2, f_null=None, tau=0.5):
    """
    What a batch of pooled features [n, d] is made of, from two noise draws of the same images at the same t (f1, f2) and
    optionally f1's forward with the null label (f_null). All shares are fractions of f1's total variance around its batch
    mean (computed from second moments, so finite-batch estimates):
      common   ||batch mean||^2 / E||f||^2: the part every sample shares (what infonce_l2n spends its range removing)
      noise    1 - tr Cov(f1, f2) / tr Cov(f1): the part that changes with the noise draw alone
      label    E||f1 - f_null||^2 / (2 tr Cov(f1)): the part the class label moves (0.5 would be as large as the spread)
      erank    effective rank of the total covariance; erank_x of the cross-view (noise-free) covariance
      l2n      repa infonce_l2n value (uncentered, per-sample RMS); l2s infonce_l2s; xview xview_infonce (norm 'cross')
    """
    from repa import dispersive_loss
    f1, f2 = f1.float(), f2.float()
    n, d = f1.shape
    c1, c2 = f1 - f1.mean(0), f2 - f2.mean(0)
    tot = c1.pow(2).sum(1).mean()
    out = {
        'common': (f1.mean(0).pow(2).sum() / f1.pow(2).sum(1).mean()).item(),
        'noise': (1 - (c1 * c2).sum(1).mean() / tot).item(),
        'erank': effective_rank(f1),
        'l2n': dispersive_loss(f1[:, None], tau, 'infonce_l2n', pool='mean').item(),
        'l2s': infonce_l2s(f1[:, None], tau, pool='mean').item(),
        'xview': xview_infonce(f1[:, None], f2[:, None], tau, pool='mean', norm='cross').item(),
        'xrank_loss': xview_rank(f1[:, None], f2[:, None]).item(),
        'optimum': math.log((1 + (n - 1) * math.exp(-2 / tau)) / n),
    }
    Cx = (c1.T @ c2) / n
    Cs = 0.5 * (Cx + Cx.T)
    out['erank_x'] = (Cs.trace() ** 2 / Cs.pow(2).sum()).item()
    if f_null is not None:
        out['label'] = ((f1 - f_null.float()).pow(2).sum(1).mean() / (2 * tot)).item()
    return out


@torch.no_grad()
def block_features(net, x, y, t, blocks, noise_scale, null_label, seed=0, ctx=None):
    """
    Mean-pooled outputs of `blocks` (1-indexed) for images x at noise levels t [B], for two independent noise draws and for
    the first draw with the null label, via forward hooks (as analyze_dispersion.py). Returns {block: (f1, f2, f_null)},
    ready for feature_report. Run it per t value (a constant t) to see the t-dependence.
    """
    from contextlib import nullcontext
    ctx = ctx or nullcontext()
    n_strip = lambda i: net.in_context_len if (net.in_context_len > 0 and i >= net.in_context_start) else 0
    outs = {}
    hooks = [net.blocks[b - 1].register_forward_hook(
        lambda m, i, o, b=b: outs.__setitem__(b, o[:, n_strip(b - 1):].float().mean(1))) for b in blocks]
    g = torch.Generator(device=x.device).manual_seed(seed)
    tt = t.view(-1, 1, 1, 1)
    res = {}
    try:
        for tag, yy in (('a', y), ('b', y), ('null', torch.full_like(y, null_label))):
            if tag != 'null':
                e = torch.randn(x.shape, generator=g, device=x.device) * noise_scale
                if tag == 'a':
                    e_a = e
            else:
                e = e_a
            with ctx:
                net(tt * x + (1 - tt) * e, t, yy)
            res[tag] = dict(outs)
    finally:
        for h in hooks:
            h.remove()
    return {b: (res['a'][b], res['b'][b], res['null'][b]) for b in blocks}


# -----------------------------------------------------------------------------
# CPU self-tests: python disp_variants.py

def _selftest():
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from repa import dispersive_loss
    torch.manual_seed(0)
    ok = lambda c, msg: print(('PASS ' if c else 'FAIL ') + msg) or c
    results = []
    n, T, C = 256, 4, 64
    d = T * C
    mu = torch.randn(n, T, C)

    # infonce_l2s: invariances, Euler (gradient orthogonal to the features), equals infonce_l2 on standardized features
    h = torch.randn(n, T, C) * 0.1 + 0.3
    l0 = infonce_l2s(h)
    results.append(ok(abs(infonce_l2s(h * 37.0) - l0) < 1e-4, f'l2s scale-invariant ({l0:.4f})'))
    results.append(ok(abs(infonce_l2s(h + torch.randn(1, T, C)) - l0) < 1e-4, 'l2s invariant to a batch-common shift'))
    hg = h.clone().requires_grad_(True)
    g, = torch.autograd.grad(infonce_l2s(hg), hg)
    results.append(ok(abs((g * hg).sum()) / (g.norm() * hg.norm()) < 1e-4, 'l2s gradient has no radial component (no inflation)'))
    hc = h - h.mean(0, keepdim=True)
    hc = hc / hc.pow(2).mean().sqrt()
    results.append(ok(abs(infonce_l2s(hc) - dispersive_loss(hc, 0.5, 'infonce_l2')) < 1e-4,
                      'l2s = repa infonce_l2 on zero-mean unit-RMS features'))
    results.append(ok(abs(infonce_l2s(h, weights=torch.ones(n)) - l0) < 1e-4, 'l2s with unit weights = unweighted'))
    w = (torch.rand(n) < 0.5).float()
    sub = h[w.bool()]
    results.append(ok(abs(infonce_l2s(h, weights=w) - infonce_l2s(sub)) < 1e-3, 'l2s with 0/1 weights = the loss on the kept samples'))
    results.append(ok(abs(infonce_l2s(h, groups=1) - l0) < 1e-6 and infonce_l2s(h, groups=16).item() > l0.item(),
                      f'l2s groups: 16 groups {infonce_l2s(h, groups=16):.3f} (floor log(1/16) = {math.log(1 / 16):.3f})'))
    # infonce_l2n, by contrast, is dominated by the batch-common component
    shared = torch.randn(1, T, C)
    hs = mu * 0.13 + shared # pairwise cosine ~0.98
    results.append(ok(dispersive_loss(hs, 0.5, 'infonce_l2n') > -0.2 and infonce_l2s(hs) < -3.5,
                      f'common mode: l2n {dispersive_loss(hs, 0.5, "infonce_l2n"):.3f} (looks collapsed), l2s {infonce_l2s(hs):.3f}'))

    # the noise shortcut: single-view losses fall when per-sample noise is added; the cross-view loss doesn't
    low = torch.randn(n, 8) @ torch.randn(8, d) # content in 8 directions: poorly dispersed
    low = low.view(n, T, C) / low.std()
    for s in (0.0, 1.0, 3.0):
        e1, e2 = torch.randn(n, T, C) * s, torch.randn(n, T, C) * s
        print(f'     noise {s}: l2n {dispersive_loss(low + e1, 0.5, "infonce_l2n"):.3f}  l2s {infonce_l2s(low + e1):.3f}  '
              f'xview total {xview_infonce(low + e1, low + e2, pool="none", norm="total"):.3f}  cross {xview_infonce(low + e1, low + e2, pool="none", norm="cross"):.3f}'
              f'  xrank {xview_rank(low + e1, low + e2, pool="none"):.3f}')
    e1, e2 = torch.randn(n, T, C) * 3, torch.randn(n, T, C) * 3
    results.append(ok(infonce_l2s(low + e1) < infonce_l2s(low) - 0.3, 'l2s rewards per-sample noise (the shortcut exists)'))
    results.append(ok(xview_infonce(low + e1, low + e2, pool='none', norm='total') > xview_infonce(low, low, pool='none', norm='total'),
                      'xview (total) never rewards noise'))
    results.append(ok(xview_infonce(low + e1, low + e2, pool='none', norm='cross') > xview_infonce(low, low, pool='none', norm='cross') - 0.02,
                      'xview (cross) never rewards noise'))
    results.append(ok(abs(xview_infonce(low, low, pool='none') - infonce_l2s(low)) < 1e-4, 'xview on identical views = l2s'))
    results.append(ok(abs(xview_infonce(low * 5 + 1, low * 5 + 1, pool='none') - xview_infonce(low, low, pool='none')) < 1e-4,
                      'xview scale- and shift-invariant'))

    # gradient at the shortcut: features that are almost pure noise. l2s is near its optimum and pushes nowhere useful; xview's
    # gradient asks for more content (d loss / d content scale < 0) with a non-vanishing magnitude
    z1, z2 = torch.randn(n, T, C) * 3, torch.randn(n, T, C) * 3
    rich = mu / mu.std() # content spread over all directions
    grads = {}
    for name, f in (('l2s', lambda s: infonce_l2s(z1 + s * rich)),
                    ('xview', lambda s: xview_infonce(z1 + s * rich, z2 + s * rich, pool='none'))):
        s = torch.tensor(0.3, requires_grad=True)
        gs, = torch.autograd.grad(f(s), s)
        print(f'     {name}: loss at noise-dominated features {f(torch.tensor(0.3)):.3f}, d loss / d content scale {gs:.3f}')
        grads[name] = gs.item()
    results.append(ok(grads['xview'] < 0 and abs(grads['xview']) > 50 * abs(grads['l2s']),
                      'at a noise-dominated representation xview still pulls toward content (>50x l2s, which has ~no gradient)'))

    # t shortcut: a high-dimensional function of t alone (independent t per view) disperses a single view, not the cross view
    W = torch.randn(64, d)
    tf = lambda t: (torch.cos(t[:, None] * 40 * torch.randn(1, 64, generator=torch.Generator().manual_seed(1)))
                    @ W).view(-1, T, C) / 3
    ta, tb = torch.rand(n), torch.rand(n)
    results.append(ok(infonce_l2s(low + tf(ta)) < infonce_l2s(low) - 0.3, 'l2s rewards t-only directions'))
    results.append(ok(xview_infonce(low + tf(ta), low + tf(tb), pool='none') > xview_infonce(low, low, pool='none') - 0.05,
                      'xview with independent t per view does not reward t-only directions'))

    # xview_rank: isotropic < low-rank; invariances; noise-unbiased up to estimation error
    iso = torch.randn(n, 1, d)
    results.append(ok(xview_rank(iso, iso) < 0.5 * math.log(d) and xview_rank(low, low) > xview_rank(iso, iso) + 1,
                      f'xview_rank isotropic {xview_rank(iso, iso):.3f} (ideal 0; n = d here, so ~log 2), 8-dim {xview_rank(low, low):.3f} (rank 8 of C = {C}: at least log(C / 8) = {math.log(C / 8):.3f})'))
    results.append(ok(abs(xview_rank(low * 9 + 2, low * 9 + 2) - xview_rank(low, low)) < 1e-4, 'xview_rank scale- and shift-invariant'))
    big = torch.randn(4096, 8) @ torch.randn(8, d)
    big = (big / big.std()).view(-1, T, C)
    nz1, nz2 = 2 * torch.randn_like(big), 2 * torch.randn_like(big)
    results.append(ok(abs(xview_rank(big + nz1, big + nz2) - xview_rank(big, big)) < 0.15 and
                      xview_rank(big + nz1, big + nz1) < xview_rank(big, big) - 1,
                      f'xview_rank ignores noise ({xview_rank(big + nz1, big + nz2):.3f} vs clean {xview_rank(big, big):.3f}); '
                      f'single-view rank would be fooled ({xview_rank(big + nz1, big + nz1):.3f})'))

    # xview_distill: only the noisier view gets gradient; centered (the common component can't satisfy it); head starts as identity
    ha = torch.randn(n, T, C, requires_grad=True)
    hb = torch.randn(n, T, C, requires_grad=True)
    ta, tb = torch.rand(n), torch.rand(n)
    ga, gb = torch.autograd.grad(xview_distill(ha, hb, ta, tb, DistillHead(C)), (ha, hb))
    a_noisy = (ta < tb)
    results.append(ok(ga[~a_noisy].abs().max() == 0 and gb[a_noisy].abs().max() == 0 and ga[a_noisy].abs().max() > 0,
                      'xview_distill: gradient only into the noisier view'))
    same = torch.randn(n, T, C)
    results.append(ok(xview_distill(same, same, ta, tb, DistillHead(C)) < 1e-5 and
                      abs(xview_distill(same + shared * 100, same * 0.5 + shared * 100, ta, tb) - xview_distill(same, same * 0.5, ta, tb)) < 1e-4,
                      'xview_distill: 0 for identical views, blind to a batch-common offset'))

    # contrastive_x_loss: the minimizer of v_loss - lam * contrastive over a free prediction is m + (x - m) / (1 - lam)
    lam = 0.2
    xs = torch.randn(512, 3, 4, 4) + 0.7
    tt = torch.full((512,), 0.5)
    m = xs[torch.roll(torch.arange(512), 1)] # per sample, its negative (the expectation over negatives is the data mean)
    target = ((xs - lam * m) / (1 - lam)).requires_grad_(True)
    g, = torch.autograd.grad(((xs - target) / 0.5).pow(2).mean() - lam * contrastive_x_loss(target, xs, tt), target)
    g0, = torch.autograd.grad(((xs - target) / 0.5).pow(2).mean(), target)
    results.append(ok(g.abs().max() < 1e-6 * g0.abs().max() + 1e-12, 'contrastive_x_loss: stationary at (x - lam x_neg) / (1 - lam)'))

    # snr_weight
    sw = snr_weight(torch.tensor([0.1, 0.5, 0.9]))
    results.append(ok(abs(sw[1] - 0.5) < 1e-6 and sw[0] < 0.05 and sw[2] > 0.95, f'snr_weight {sw.tolist()}'))

    # feature_report and block_features on a tiny JiT (CPU)
    from model_jit import JiT
    net = JiT(input_size=32, patch_size=16, hidden_size=64, depth=4, num_heads=4, bottleneck_dim=16, in_context_len=0,
              num_classes=10, attn_res='full', adaln_single=True).eval()
    xb = torch.rand(64, 3, 32, 32) * 2 - 1
    yb = torch.randint(10, (64,))
    feats = block_features(net, xb, yb, torch.full((64,), 0.5), (2, 4), noise_scale=0.5, null_label=10)
    rep = feature_report(*feats[2])
    print('     tiny JiT block 2 at init, t=0.5:', {k: round(v, 3) for k, v in rep.items()})
    results.append(ok(all(math.isfinite(v) for v in rep.values()) and 0 <= rep['common'] <= 1, 'feature_report on a tiny JiT'))
    cap = SublayerCapture(net, 3)
    net.train()
    with torch.no_grad(): # the MLP output layers start at zero: give them trained-like weights
        for blk in net.blocks:
            blk.mlp.w3.weight.normal_(0, 0.02)
    hx = net(xb, torch.full((64,), 0.5), yb)
    g, = torch.autograd.grad(infonce_l2s(cap.out, pool='mean'), net.x_embedder.proj1.weight if hasattr(net.x_embedder, "proj1") else next(net.x_embedder.parameters()))
    results.append(ok(cap.out.shape == (64, 4, 64) and g.abs().sum() > 0, 'SublayerCapture: block MLP output, gradient reaches the patch embedding'))
    p1, p2 = split_views(torch.arange(8.0).view(8, 1, 1))
    X2, Y2 = paired_views(torch.arange(8.0), torch.arange(8))
    results.append(ok(torch.equal(X2[:4], X2[4:]) and p1.shape == p2.shape, 'paired_views / split_views'))
    print(f'{sum(results)}/{len(results)} passed')
    return all(results)


if __name__ == '__main__':
    import torch._dynamo
    torch._dynamo.config.disable = True # eager on CPU: the blocks' @torch.compile only costs time here
    raise SystemExit(0 if _selftest() else 1)
