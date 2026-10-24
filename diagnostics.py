"""
Training diagnostics for JiT, aimed at finding where data efficiency is lost.

    1. per-noise-level loss, train vs held-out          fixed images / noise / t, so curves are smooth
    2. precision / recall of samples                    fidelity vs coverage (in train_jit.py, via fid.py)
    3. EMA vs raw weights                               FID / precision / recall of both (in train_jit.py)
    4. optimization health                              grad norms, update/weight ratios per block, adaLN gates, attention entropy
       + locality                                       mean attention distance per block, and a windowed-attention ablation:
                                                        attention confined to non-overlapping tiles, loss increase per noise level
    5. representation probes                            k-NN + linear probe on intermediate tokens, at several noise levels
    6. error by frequency band                          prediction error spectrum per noise level, sample vs real spectra
    7. memorization                                     nearest-training-image distances of samples vs held-out images
    8. input sensitivity                                Jacobian of the prediction w.r.t. its noisy input per noise level
                                                        (how much it copies / amplifies the input), sampled vs trunk-only head

Items 1, 4 (gates / attention) and 6 share one instrumented forward pass over a fixed batch.
JiT's convention: t=1 is the clean image, t=0 pure noise.
"""

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from model_jit import modulate, AttnResState, BottleneckPatchEmbed
from evaluation import knn_classifier


# -----------------------------------------------------------------------------
# instrumented forward (mirrors JiT.forward / JiTBlock.forward / Attention.forward in model_jit.py, eagerly)

_geometry = {}

def grid_geometry(g, device):
    """
    Row-major (row, col) of the g*g patch tokens, their pairwise Euclidean distances (in patches), and the mean distance
    under uniform attention, which normalizes the attention distance so 1.0 = as spread out as uniform attention.
    """
    key = (g, str(device))
    if key not in _geometry:
        r, c = torch.meshgrid(torch.arange(g), torch.arange(g), indexing='ij')
        pos = torch.stack([r.flatten(), c.flatten()], 1).float().to(device)
        dist = torch.cdist(pos, pos)
        _geometry[key] = (pos, dist, dist.mean())
    return _geometry[key]

def tile_mask(g, tile, n_prefix, device):
    """
    Attention mask for the windowed-attention ablation: patch tokens only see patches in the same non-overlapping
    tile x tile block (in every layer, so information can never leave a tile), plus the in-context class tokens.
    The in-context tokens only see each other, so they can't relay image content between tiles.
    True = allowed. Shape [n_prefix + g*g] squared.
    """
    pos, _, _ = grid_geometry(g, device)
    tile_id = (pos[:, 0] // tile) * g + (pos[:, 1] // tile)
    same_tile = tile_id.view(-1, 1) == tile_id.view(1, -1)
    if n_prefix == 0:
        return same_tile
    n = n_prefix + g * g
    mask = torch.zeros(n, n, dtype=torch.bool, device=device)
    mask[:n_prefix, :n_prefix] = True
    mask[n_prefix:, :n_prefix] = True
    mask[n_prefix:, n_prefix:] = same_tile
    return mask

def _attention(attn, x, rope, n_prefix, g, mask=None):
    B, N, C = x.shape
    H = attn.num_heads
    qkv = attn.qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = rope(attn.q_norm(q)), rope(attn.k_norm(k))
    with torch.autocast(device_type=x.device.type, enabled=False):
        logits = q.float() @ k.float().transpose(-2, -1) / math.sqrt(q.shape[-1])
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        w = torch.softmax(logits, dim=-1)
    stats = {}
    # attention entropy per query, normalized by log(N): 1 = uniform, 0 = attends to one token
    stats['attn_entropy'] = (-(w * torch.log(w.clamp_min(1e-12))).sum(-1) / math.log(N)).mean()
    # locality of patch queries: mean spatial distance to the patches they attend to (renormalized over patch keys),
    # relative to uniform attention; and the attention mass they put on the in-context class tokens
    wp = w[:, :, n_prefix:, :]
    wpp = wp[..., n_prefix:]
    wpp = wpp / wpp.sum(-1, keepdim=True).clamp_min(1e-12)
    _, dist, uniform = grid_geometry(g, x.device)
    stats['attn_dist'] = (wpp * dist).sum(-1).mean() / uniform
    stats['attn_prefix_mass'] = wp[..., :n_prefix].sum(-1).mean() if n_prefix else torch.zeros((), device=x.device)
    out = attn.proj((w @ v).transpose(1, 2).reshape(B, N, C)) # same casting as JiT: fp32 weights @ v under the caller's autocast
    return out, stats

def _block(block, x, c, rope, n_prefix, g, mask=None, res=None, index=None):
    # res: AttnResState for attention-residual models (x is then the depth-mixed attention input, see JiTBlock.forward_attn_res)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.modulation(c)
    attn_out, stats = _attention(block.attn, modulate(block.norm1(x), shift_msa, scale_msa), rope, n_prefix, g, mask)
    if res is None:
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))
    else:
        res.add(gate_msa.unsqueeze(1) * attn_out)
        res.add(gate_mlp.unsqueeze(1) * block.mlp(modulate(block.norm2(res.mix(2 * index + 1)), shift_mlp, scale_mlp)))
        x = res.mix(2 * index + 2)
    stats['gate_msa'], stats['gate_mlp'] = gate_msa.abs().mean(), gate_mlp.abs().mean()
    return x, stats

def instrumented_forward(net, x, t, y, capture_blocks=(), tile=None, skip_on=True):
    """
    Same output as net(x, t, y) (in eval: none of training's stop-gradients, so gradients flow to x through every path), plus per-block attention stats (entropy, spatial distance, mass on in-context tokens),
    |gate| sizes, and the patch tokens (in-context class tokens removed) after the blocks in capture_blocks (1-indexed).
    tile: if set, runs the windowed-attention ablation (see tile_mask) instead of full attention.
    skip_on: False runs a skip-pathway model without its skip (the trunk-only prediction, used for skip_residual).
    """
    if getattr(net, 'family', 'jit') == 'pixeldit':
        return _pixeldit_forward(net, x, t, y, capture_blocks, tile)
    g = int(round(net.x_embedder.num_patches ** 0.5))
    x_input = x
    masks = {}
    if tile is not None:
        masks = {0: tile_mask(g, tile, 0, x.device), net.in_context_len: tile_mask(g, tile, net.in_context_len, x.device)}
    t_emb = net.t_embedder(t)
    y_emb = net.y_embedder(y)
    c = t_emb + y_emb
    c_blocks = net.t_block(c) if getattr(net, 'adaln_single', False) else c
    decoder = getattr(net, 'skip_decoder', 'linear') != 'linear' and skip_on # skip pathway as a pixel / subpatch decoder
    skip = net.skip_embedder(x) if getattr(net, 'skip_pathway', False) and skip_on and not decoder else None # from the raw noisy input, as in JiT.forward
    skip_ratio = None
    if getattr(net, 'dcae_stem', False):
        x = net.input_stem(x, c)
    if getattr(net, 'gated_bottleneck', False):
        # gated bottleneck: how much the full-rank bypass adds relative to the bottleneck path (RMS ratio; 0 = shut)
        base = BottleneckPatchEmbed.forward(net.x_embedder, x) # the bottleneck path alone
        bypass = net.bottleneck_gate(c).unsqueeze(1) * net.x_embedder.bypass_tokens(x)
        h = base + bypass + net.pos_embed
        bypass_ratio = bypass.float().pow(2).mean().sqrt() / base.float().pow(2).mean().sqrt().clamp_min(1e-12)
    else:
        h = net.x_embedder(x) + net.pos_embed
        bypass_ratio = None
    stats = {k: [] for k in ('attn_entropy', 'attn_dist', 'attn_prefix_mass', 'gate_msa', 'gate_mlp')}
    feats = {}
    res = AttnResState(net, h, c) if getattr(net, 'attn_res', 'none') != 'none' else None
    if res is not None:
        res.weights = {0: torch.ones(1, device=x.device)} # record the depth-mixing weights; the first sublayer reads the embedding only
    for i, block in enumerate(net.blocks):
        if net.in_context_len > 0 and i == net.in_context_start:
            ic = y_emb.unsqueeze(1).repeat(1, net.in_context_len, 1) + net.in_context_posemb
            if res is None:
                h = torch.cat([ic, h], dim=1)
            else:
                res.prepend(ic)
                h = res.mix(2 * i)
        n_prefix = net.in_context_len if (net.in_context_len > 0 and i >= net.in_context_start) else 0
        rope = net.feat_rope if i < net.in_context_start else net.feat_rope_incontext
        h, block_stats = _block(block, h, c_blocks, rope, n_prefix, g, masks.get(n_prefix), res, i)
        if skip is not None and i + 1 == net.skip_inject_after < len(net.blocks):
            gated = net.skip_gate(c).unsqueeze(1) * skip
            skip_ratio = gated.float().pow(2).mean().sqrt() / h[:, n_prefix:].float().pow(2).mean().sqrt().clamp_min(1e-12)
            h = net.inject_skip(h, gated, res, i)
        for k, v in block_stats.items():
            stats[k].append(v)
        if (i + 1) in capture_blocks:
            feats[i + 1] = h[:, n_prefix:]
    h = h[:, net.in_context_len:]
    if skip is not None:
        if net.skip_inject_after == len(net.blocks):
            # skip pathway: RMS of the gated input patches relative to the trunk output they're added to (0 = gate shut)
            gated = net.skip_gate(c).unsqueeze(1) * skip
            skip_ratio = gated.float().pow(2).mean().sqrt() / h.float().pow(2).mean().sqrt().clamp_min(1e-12)
            h = h + gated
        for block in net.skip_head:
            h = block(h, c_blocks, net.feat_rope)
    if decoder: # the raw noisy pixels decoded under the trunk's tokens (and its AttnRes state), as JiT.forward does
        out = net.decoder_output(x_input, h, t_emb, res, c)
        # how far the decoder's input gate is open: RMS of the gated noisy pixels relative to the trunk readout
        readout, gated = net.skip_decoder_module.initial_tokens(x_input, h, c)
        skip_ratio = gated.float().pow(2).mean().sqrt() / readout.float().pow(2).mean().sqrt().clamp_min(1e-12)
    elif skip is not None and net.skip_residual:
        # trunk-only prediction (same images, noise and attention masks, skip off) + the skip branch's correction
        trunk_out, _, _ = instrumented_forward(net, x_input, t, y, tile=tile, skip_on=False)
        out = trunk_out + net.output(h, c, net.prediction_layer())
    else:
        # final layer (the skip branch's own when the skip is on), unpatchify / overlap-add, DC-AE head
        out = net.output(h, c, net.prediction_layer() if skip is not None else None,
                         head=net.sampled_head() if skip is not None or not getattr(net, 'skip_pathway', False) else net.trunk_head())
    stats = {k: torch.stack(v) for k, v in stats.items()}
    if bypass_ratio is not None:
        stats['bypass_ratio'] = bypass_ratio.reshape(1)
    if skip_ratio is not None:
        stats['skip_ratio'] = skip_ratio.reshape(1)
    if res is not None:
        stats['attn_res_alpha'] = attn_res_matrix(res.weights, x.device)
    return out, stats, feats

def _pixeldit_forward(net, x, t, y, capture_blocks=(), tile=None):
    # PixelDiT: the model's own forward, with the patch-level blocks' tokens captured by hooks for the probes; the attention
    # statistics and the windowed-attention ablation are JiT-specific and not computed
    assert tile is None, 'the windowed-attention ablation is only implemented for JiT'
    feats, hooks = {}, []
    for b in capture_blocks:
        hooks.append(net.blocks[b - 1].register_forward_hook(lambda m, i, o, b=b: feats.__setitem__(b, o)))
    try:
        out = net(x, t, y)
    finally:
        for h in hooks:
            h.remove()
    return out, {}, feats

def attn_res_matrix(weights, device):
    """
    AttnRes mixing weights as [sublayers + final, sources] (mean over batch and tokens). Row l: sublayer l's input
    (attention / MLP of each block in order, then the final layer). Column 0: the patch embedding, column j: the j-th
    source (a sublayer output for full, a block sum for block; the current block's partial sum sits at its own index).
    NaN where a source doesn't exist yet.
    """
    rows = max(weights) + 1
    cols = max(len(w) for w in weights.values())
    m = torch.full((rows, cols), float('nan'), device=device)
    for l, w in weights.items():
        m[l, :len(w)] = w
    return m

def attn_res_labels(depth, n_sources):
    # a mid-trunk skip pathway is part of the source of the block it was injected after (no column of its own)
    rows = [f'b{i // 2 + 1}.{"attn" if i % 2 == 0 else "mlp"}' for i in range(2 * depth)] + ['final']
    if n_sources == 2 * depth + 1: # full: one source per sublayer output
        cols = ['emb'] + rows[:-1]
    else:
        cols = ['emb'] + [f'group{j}' for j in range(1, n_sources)]
    return rows, cols

# -----------------------------------------------------------------------------
# 6. frequency bands

def radial_band_masks(size, n_bands, device):
    """
    n_bands equal-width rings of radial frequency from 0 to Nyquist (corners beyond Nyquist go in the last band).
    """
    f = torch.fft.fftfreq(size, device=device)
    r = torch.sqrt(f.view(-1, 1) ** 2 + f.view(1, -1) ** 2)
    idx = (r / 0.5 * n_bands).long().clamp(max=n_bands - 1)
    return F.one_hot(idx, n_bands).permute(2, 0, 1).float() # [n_bands, size, size]

def band_power(x, masks):
    """
    x: [B, C, H, W] -> summed power per band [n_bands] (summed over batch and channels).
    """
    p = torch.fft.fft2(x.float()).abs().pow(2).sum(dim=(0, 1))
    return (masks * p).sum(dim=(1, 2))

# -----------------------------------------------------------------------------
# 1 + 4 + 6: per-noise-level pass

def t_grid(n_bins, lo=-3.0, hi=3.0):
    # bin centers equally spaced in logit(t), covering the bulk of JiT's logit-normal(-0.8, 0.8) training distribution
    return torch.sigmoid(torch.linspace(lo, hi, n_bins))

@torch.no_grad()
def noise_level_pass(net, x, y, ts, noise_scale, t_eps, ctx, masks, seed=0, chunk=128, tile=None, skip_on=True):
    """
    For each t in ts, noises the fixed batch x with fixed noise and records:
      v_loss (the training objective), x_mse, relative prediction error per frequency band,
      the same band error for two estimates that need no learning, on the same images and noise:
        copy    x_hat = z / t, i.e. pass the (rescaled) noisy input through: a model that can keep the input's detail
                and edit it should never be worse than this at low noise
        wiener  the optimal linear shift-invariant denoiser of z / t, using the batch's mean power spectrum (estimated on
                the same images, so slightly optimistic)
      attention entropy / spatial distance / in-context mass and gate sizes per block (averaged over the batch).
    tile: run the windowed-attention ablation (tile_mask) instead of full attention; same images, noise and t.
    skip_on: False scores a skip-pathway model's trunk-only prediction (the one its trunk is trained on) instead of the sampled one.
    Returns dict of tensors with a leading len(ts) dimension.
    """
    g = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=g, device=x.device) * noise_scale
    out = defaultdict(list)
    spectrum = torch.fft.fft2(x.float()).abs().pow(2).mean(dim=(0, 1)) # mean signal power per frequency
    for t_val in ts.tolist():
        acc = defaultdict(float)
        err_power, x_power, copy_power, wiener_power = 0, 0, 0, 0
        # noise power per frequency in z / t (unnormalized FFT of white noise: H * W * variance), and the Wiener gain
        noise_power = ((1 - t_val) / t_val) ** 2 * noise_scale ** 2 * x.shape[-2] * x.shape[-1]
        gain = spectrum / (spectrum + noise_power)
        for i in range(0, len(x), chunk):
            xb, yb, eb = x[i:i + chunk], y[i:i + chunk], noise[i:i + chunk]
            t = torch.full((len(xb),), t_val, device=x.device)
            tt = t.view(-1, 1, 1, 1)
            z = tt * xb + (1 - tt) * eb
            with ctx:
                x_pred, stats, _ = instrumented_forward(net, z, t, yb, tile=tile, skip_on=skip_on)
            x_pred = x_pred.float()
            v = (xb - z) / (1 - tt).clamp_min(t_eps)
            v_pred = (x_pred - z) / (1 - tt).clamp_min(t_eps)
            w = len(xb) / len(x)
            acc['v_loss'] += ((v - v_pred) ** 2).mean().item() * w
            acc['x_mse'] += ((xb - x_pred) ** 2).mean().item() * w
            for k, s in stats.items():
                acc[k] = acc[k] + s.float() * w
            err_power = err_power + band_power(x_pred - xb, masks)
            x_power = x_power + band_power(xb, masks)
            copy = z.float() / t_val
            copy_power = copy_power + band_power(copy - xb, masks)
            wiener = torch.fft.ifft2(torch.fft.fft2(copy) * gain).real
            wiener_power = wiener_power + band_power(wiener - xb, masks)
        for k, val in acc.items():
            out[k].append(torch.as_tensor(val, device='cpu', dtype=torch.float32))
        out['freq_rel_err'].append((err_power / x_power).cpu())
        out['freq_rel_err_copy'].append((copy_power / x_power).cpu())
        out['freq_rel_err_wiener'].append((wiener_power / x_power).cpu())
    return {k: torch.stack(v) for k, v in out.items()}

# -----------------------------------------------------------------------------
# 8. input sensitivity

def input_sensitivity(net, x, y, ts, noise_scale, ctx, n_images=256, seed=0, chunk=64, skip_on=True, masks=None):
    """
    How strongly the x prediction reacts to its noisy input z, per noise level: the Jacobian J = d x_hat / d z of each
    image's n = 3 H W outputs w.r.t. its n inputs, from one Rademacher probe r per image through a vector-Jacobian
    product (Hutchinson): tr J ~ r . J^T r, ||J||_F^2 ~ |J^T r|^2. In units where passing the input through (x_hat = z / t)
    scores 1 and ignoring it scores 0:
      copy     t tr(J) / n          how much of the input the prediction keeps, pixel for pixel
      gain     t^2 ||J||_F^2 / n    mean squared amplification of a small random change to z, including what spreads to
                                    other pixels (texture made out of the noise counts here, not in copy)
      tweedie  copy / (SNR x_mse), SNR = t^2 / ((1 - t) noise_scale)^2. The MMSE denoiser has J = t Cov[x | z] / ((1 - t) noise_scale)^2
               (Tweedie's second-order identity), so its ratio is 1 in expectation; > 1: the prediction follows its input
               more than an optimal denoiser with the same error would (equivalently, by SURE: its output stays closer to
               the input than that denoiser's would)
    masks ([bands, H, W], radial_band_masks): the same three per frequency band, from the same probe: with P_b the
    projection onto band b, copy_b = t tr(P_b J) / tr(P_b) estimated as t (P_b r) . J^T r, gain_b = t^2 |P_b J^T r|^2 / tr(P_b)
    (the amplification of perturbations within band b), tweedie_b against the band's own x_mse. Under an ODE sampler a
    band whose copy runs above its tweedie-consistent level stops shedding the noise still in it.
    A sampler feeds the model its own earlier outputs, so these matter for sampling in a way the held-out loss can't show
    (a deterministic sampler's error depends on the Jacobian's error as well as the prediction's).
    Same images and noise as noise_level_pass (its first n_images). Runs with autograd on; parameters get no gradients.
    skip_on: False measures a skip-pathway model's trunk-only prediction instead, on the same z and probes.
    Returns dict of tensors: copy, gain, tweedie, x_mse [len(ts)] and, with masks, copy_band, gain_band, tweedie_band,
    x_mse_band [len(ts), bands].
    """
    g = torch.Generator(device=x.device).manual_seed(seed)
    noise = (torch.randn(x.shape, generator=g, device=x.device) * noise_scale)[:n_images] # noise_level_pass's noise
    x, y = x[:n_images], y[:n_images]
    g = torch.Generator(device=x.device).manual_seed(seed + 1)
    probes = torch.randint(0, 2, x.shape, generator=g, device=x.device).float() * 2 - 1 # the same for every t
    n, count, hw = x[0].numel(), len(x), x.shape[-2] * x.shape[-1]
    dof = masks.sum(dim=(1, 2)) * x.shape[1] * count if masks is not None else None # tr(P_b) summed over channels and images
    out = defaultdict(list)
    for t_val in ts.tolist():
        acc = defaultdict(float)
        for i in range(0, count, chunk):
            xb, yb, eb, rb = x[i:i + chunk], y[i:i + chunk], noise[i:i + chunk], probes[i:i + chunk]
            t = torch.full((len(xb),), t_val, device=x.device)
            tt = t.view(-1, 1, 1, 1)
            z = (tt * xb + (1 - tt) * eb).requires_grad_(True)
            with torch.enable_grad(), ctx:
                x_pred = instrumented_forward(net, z, t, yb, skip_on=skip_on)[0]
            (jt_r,) = torch.autograd.grad(x_pred, z, grad_outputs=rb.to(x_pred.dtype)) # J^T r, no parameter gradients
            jt_r, err = jt_r.float(), x_pred.detach().float() - xb
            acc['copy'] += t_val * (jt_r * rb).sum().item() / (n * count)
            acc['gain'] += t_val ** 2 * jt_r.pow(2).sum().item() / (n * count)
            acc['x_mse'] += err.pow(2).sum().item() / (n * count)
            if masks is not None: # per band, via the spectra (Parseval: a . b = sum_f Re(conj(A_f) B_f) / HW)
                R, G = torch.fft.fft2(rb), torch.fft.fft2(jt_r)
                band = lambda p: (masks * p.sum(dim=(0, 1))).sum(dim=(1, 2)) / hw
                acc['copy_band'] = acc['copy_band'] + t_val * band((R.conj() * G).real) / dof
                acc['gain_band'] = acc['gain_band'] + t_val ** 2 * band(G.abs().pow(2)) / dof
                acc['x_mse_band'] = acc['x_mse_band'] + band(torch.fft.fft2(err).abs().pow(2)) / dof
        snr = t_val ** 2 / ((1 - t_val) * noise_scale) ** 2
        acc['tweedie'] = acc['copy'] / max(snr * acc['x_mse'], 1e-12)
        if masks is not None:
            acc['tweedie_band'] = acc['copy_band'] / (snr * acc['x_mse_band']).clamp_min(1e-12)
        for k, val in acc.items():
            out[k].append(torch.as_tensor(val, dtype=torch.float32).cpu())
    return {k: torch.stack(v) for k, v in out.items()}

@torch.no_grad()
def trajectory_pass(denoiser, x, y, t0s, noise_scale, ctx, masks, n_images=256, seed=0, chunk=256):
    """
    Error along the sampler's own trajectory, on held-out images: each image noised to t0 (noise_level_pass's noise) and
    the rest of the sampling run from there (denoiser.sample_from: its method and time grid, no CFG), compared with the
    clean image per frequency band, next to the one-step prediction's error at the same z. The one-step error is what
    the loss measures; the gap to the trajectory error is what the sampler adds by feeding the model its own outputs, which
    is what samples and FID see. Per t0, [bands] each:
      onestep_err    relative error per band of x_hat(z_t0)
      traj_err       relative error per band of the sampler's final image
      traj_spectrum  log10 power per band of the final image / the clean image's (> 0: excess power, e.g. grain)
    Returns dict of [len(t0s), bands] tensors.
    """
    g = torch.Generator(device=x.device).manual_seed(seed)
    noise = (torch.randn(x.shape, generator=g, device=x.device) * noise_scale)[:n_images]
    x, y = x[:n_images], y[:n_images]
    saved = denoiser.cfg_scale, denoiser.fixed_shape
    denoiser.cfg_scale, denoiser.fixed_shape = 1.0, False
    out = defaultdict(list)
    try:
        for t0 in t0s:
            x_power, onestep_power, traj_power, final_power = 0, 0, 0, 0
            for i in range(0, len(x), chunk):
                xb, yb, eb = x[i:i + chunk], y[i:i + chunk], noise[i:i + chunk]
                z = t0 * xb + (1 - t0) * eb
                with ctx:
                    x_hat = denoiser.net(z, torch.full((len(xb),), t0, device=x.device), yb).float()
                    final = denoiser.sample_from(z, t0, yb).float()
                x_power = x_power + band_power(xb, masks)
                onestep_power = onestep_power + band_power(x_hat - xb, masks)
                traj_power = traj_power + band_power(final - xb, masks)
                final_power = final_power + band_power(final, masks)
            out['onestep_err'].append((onestep_power / x_power).cpu())
            out['traj_err'].append((traj_power / x_power).cpu())
            out['traj_spectrum'].append(torch.log10(final_power / x_power).cpu())
    finally:
        denoiser.cfg_scale, denoiser.fixed_shape = saved
    return {k: torch.stack(v) for k, v in out.items()}

# -----------------------------------------------------------------------------
# 5. representation probes

@torch.no_grad()
def probe_features(net, images, ts, blocks, null_label, ctx, device, noise_scale=1.0, seed=0, chunk=256):
    """
    Mean-pooled patch tokens after each block in `blocks`, for each noise level in ts.
    images: uint8 numpy [N, R, R, 3] (kept on CPU, moved to the GPU chunk by chunk).
    Uses the null (unconditional) label: with the true label the class would leak in through adaLN / in-context tokens.
    noise_scale: the model's training noise scale, so z = t x + (1 - t) noise_scale e matches what it was trained on.
    Returns {(block, t): [N, D] float32}.
    """
    from data_jit import to_model_input
    feats = defaultdict(list)
    for t_val in ts:
        g = torch.Generator(device=device).manual_seed(seed) # same noise for every t
        for i in range(0, len(images), chunk):
            xb = to_model_input(images[i:i + chunk], flip=False, device=device)
            eb = torch.randn(xb.shape, generator=g, device=device) * noise_scale
            t = torch.full((len(xb),), t_val, device=device)
            tt = t.view(-1, 1, 1, 1)
            z = tt * xb + (1 - tt) * eb
            y = torch.full((len(xb),), null_label, device=device, dtype=torch.long)
            with ctx:
                _, _, captured = instrumented_forward(net, z, t, y, capture_blocks=blocks)
            for b in blocks:
                feats[(b, t_val)].append(captured[b].float().mean(1))
    return {k: torch.cat(v) for k, v in feats.items()}

def linear_probe(train_x, train_y, val_x, val_y, num_classes, steps=1000, batch_size=1024, lr=1e-3, wd=1e-4):
    """
    Quick linear probe on frozen, standardized features (AdamW, cosine lr). Returns val top-1 in percent.
    """
    mu, sd = train_x.mean(0, keepdim=True), train_x.std(0, keepdim=True) + 1e-6
    train_x, val_x = (train_x - mu) / sd, (val_x - mu) / sd
    head = torch.nn.Linear(train_x.shape[1], num_classes).to(train_x.device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device=train_x.device).manual_seed(0)
    with torch.enable_grad():
        for _ in range(steps):
            idx = torch.randint(len(train_x), (min(batch_size, len(train_x)),), generator=g, device=train_x.device)
            loss = F.cross_entropy(head(train_x[idx]), train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        return 100.0 * (head(val_x).argmax(-1) == val_y).float().mean().item()

def run_probes(train_feats, train_y, val_feats, val_y, num_classes, knn_k=20):
    results = {}
    for key in train_feats:
        tr, va = train_feats[key], val_feats[key]
        knn_top1, _ = knn_classifier(F.normalize(tr, dim=-1), train_y, F.normalize(va, dim=-1), val_y,
                                     k=min(knn_k, len(tr)), num_classes=num_classes)
        results[key] = {'knn': knn_top1, 'linear': linear_probe(tr, train_y, va, val_y, num_classes)}
    return results

# -----------------------------------------------------------------------------
# 7. memorization

@torch.no_grad()
def nn_distances(queries, bank, bank_chunk=65536):
    """
    Distance from each query (on device) to its nearest neighbour in bank (numpy, e.g. a memmap of
    float16 features), streaming the bank to the GPU in chunks.
    """
    best = torch.full((len(queries),), float('inf'), device=queries.device)
    q = queries.float()
    for i in range(0, len(bank), bank_chunk):
        b = torch.from_numpy(np.asarray(bank[i:i + bank_chunk], dtype=np.float32)).to(queries.device)
        best = torch.minimum(best, torch.cdist(q, b).min(dim=1).values)
    return best

def memorization(gen_feats, train_bank, val_feats):
    # gen_feats / val_feats on device, train_bank numpy [N, 2048]
    """
    Distance from each generated sample to its nearest training image (Inception space), relative to the same
    distance for held-out real images, which are novel by construction.
      nn_ratio: median gen->train / median val->train distance. < 1 means samples sit closer to training images than novel images do.
      near_dup_rate: fraction of samples closer than the 1st percentile of val->train distances (baseline for novel images: 1%).
    """
    d_gen = nn_distances(gen_feats, train_bank)
    d_val = nn_distances(val_feats, train_bank)
    threshold = torch.quantile(d_val, 0.01)
    return {
        'nn_ratio': (d_gen.median() / d_val.median()).item(),
        'near_dup_rate': (d_gen < threshold).float().mean().item(),
    }

# -----------------------------------------------------------------------------
# 4. optimization stats

def param_group(name):
    # net.blocks.3.attn.qkv.weight -> blocks.03.attn ; net.x_embedder.proj1.weight -> x_embedder
    parts = name.split('.')
    if parts[0] == 'net':
        parts = parts[1:]
    if parts[0] == 'blocks':
        sub = 'adaLN' if parts[2] in ('adaLN_modulation', 'scale_shift_table') else parts[2] if parts[2] in ('attn', 'mlp') else 'norm'
        return f"blocks.{int(parts[1]):02d}.{sub}"
    return parts[0]

def grouped_norms(named_tensors):
    sq = defaultdict(float)
    for name, t in named_tensors:
        if t is not None:
            sq[param_group(name)] += t.detach().float().pow(2).sum().item()
    return {k: math.sqrt(v) for k, v in sq.items()}

# -----------------------------------------------------------------------------
# plots

def heatmap(values, row_labels, col_labels, title, path, log=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    v = np.log10(np.clip(values, 1e-12, None)) if log else values
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(col_labels), 1 + 0.4 * len(row_labels)))
    im = ax.imshow(v, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, label='log10' if log else None)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

def attn_res_figure(alpha, rows, cols, t_labels, path, step):
    """
    AttnRes mixing weights (rows: each sublayer's input and the final layer, cols: sources, blank = not available yet)
    at the noisiest, middle and cleanest diagnostic noise level, on a shared color scale.
    alpha: [t, rows, cols] numpy.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    picks = sorted({0, len(t_labels) // 2, len(t_labels) - 1})
    vmax = np.nanmax(alpha)
    fig, axes = plt.subplots(1, len(picks), figsize=(1.5 + 0.28 * len(cols) * len(picks), 1.5 + 0.22 * len(rows)), squeeze=False)
    for ax, i in zip(axes[0], picks):
        im = ax.imshow(np.ma.masked_invalid(alpha[i]), aspect='auto', cmap='viridis', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=6)
        ax.set_yticks(range(len(rows)), rows, fontsize=6)
        ax.set_title(f't={t_labels[i]}', fontsize=9)
    fig.colorbar(im, ax=axes[0].tolist(), label='mixing weight')
    fig.suptitle(f'AttnRes depth-mixing weights, rows: input of each sublayer, cols: source (iter {step})', fontsize=9)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path

def probe_heatmap(probes, blocks, ts, title, path):
    """
    Side-by-side k-NN and linear probe accuracy matrices (rows: block depth, cols: noise level t), values annotated.
    probes: {(block, t): {'knn': acc, 'linear': acc}}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(2.5 + 1.6 * len(ts), 1.2 + 0.55 * len(blocks)), squeeze=False)
    for ax, kind in zip(axes[0], ('knn', 'linear')):
        m = np.array([[probes[(b, t)][kind] for t in ts] for b in blocks])
        im = ax.imshow(m, aspect='auto', cmap='viridis')
        for i in range(len(blocks)):
            for j in range(len(ts)):
                ax.text(j, i, f'{m[i, j]:.1f}', ha='center', va='center', fontsize=8,
                        color='white' if m[i, j] < (m.min() + m.max()) / 2 else 'black')
        ax.set_xticks(range(len(ts)), [f't={t:g}' for t in ts], fontsize=8)
        ax.set_yticks(range(len(blocks)), [f'block {b}' for b in blocks], fontsize=8)
        ax.set_title(f"{'k-NN' if kind == 'knn' else 'linear'} probe top-1 (%)", fontsize=9)
        fig.colorbar(im, ax=ax)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
