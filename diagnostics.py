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

Items 1, 4 (gates / attention) and 6 share one instrumented forward pass over a fixed batch.
JiT's convention: t=1 is the clean image, t=0 pure noise.
"""

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from model_jit import modulate, AttnResState
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
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN_modulation(c).chunk(6, dim=-1)
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

@torch.no_grad()
def instrumented_forward(net, x, t, y, capture_blocks=(), tile=None):
    """
    Same output as net(x, t, y), plus per-block attention stats (entropy, spatial distance, mass on in-context tokens),
    |gate| sizes, and the patch tokens (in-context class tokens removed) after the blocks in capture_blocks (1-indexed).
    tile: if set, runs the windowed-attention ablation (see tile_mask) instead of full attention.
    """
    g = int(round(net.x_embedder.num_patches ** 0.5))
    masks = {}
    if tile is not None:
        masks = {0: tile_mask(g, tile, 0, x.device), net.in_context_len: tile_mask(g, tile, net.in_context_len, x.device)}
    t_emb = net.t_embedder(t)
    y_emb = net.y_embedder(y)
    c = t_emb + y_emb
    h = net.x_embedder(x) + net.pos_embed
    stats = {k: [] for k in ('attn_entropy', 'attn_dist', 'attn_prefix_mass', 'gate_msa', 'gate_mlp')}
    feats = {}
    res = AttnResState(net, h) if getattr(net, 'attn_res', 'none') != 'none' else None
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
        h, block_stats = _block(block, h, c, rope, n_prefix, g, masks.get(n_prefix), res, i)
        for k, v in block_stats.items():
            stats[k].append(v)
        if (i + 1) in capture_blocks:
            feats[i + 1] = h[:, n_prefix:]
    h = h[:, net.in_context_len:]
    fl = net.final_layer
    shift, scale = fl.adaLN_modulation(c).chunk(2, dim=1)
    out = net.unpatchify(fl.linear(modulate(fl.norm_final(h), shift, scale)), net.patch_size)
    stats = {k: torch.stack(v) for k, v in stats.items()}
    return out, stats, feats

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
def noise_level_pass(net, x, y, ts, noise_scale, t_eps, ctx, masks, seed=0, chunk=128, tile=None):
    """
    For each t in ts, noises the fixed batch x with fixed noise and records:
      v_loss (the training objective), x_mse, relative prediction error per frequency band,
      attention entropy / spatial distance / in-context mass and gate sizes per block (averaged over the batch).
    tile: run the windowed-attention ablation (tile_mask) instead of full attention; same images, noise and t.
    Returns dict of tensors with a leading len(ts) dimension.
    """
    g = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=g, device=x.device) * noise_scale
    out = defaultdict(list)
    for t_val in ts.tolist():
        acc = defaultdict(float)
        err_power, x_power = 0, 0
        for i in range(0, len(x), chunk):
            xb, yb, eb = x[i:i + chunk], y[i:i + chunk], noise[i:i + chunk]
            t = torch.full((len(xb),), t_val, device=x.device)
            tt = t.view(-1, 1, 1, 1)
            z = tt * xb + (1 - tt) * eb
            with ctx:
                x_pred, stats, _ = instrumented_forward(net, z, t, yb, tile=tile)
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
        for k, val in acc.items():
            out[k].append(torch.as_tensor(val, device='cpu', dtype=torch.float32))
        out['freq_rel_err'].append((err_power / x_power).cpu())
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
        sub = parts[2] if parts[2] in ('attn', 'mlp', 'adaLN_modulation') else 'norm'
        return f"blocks.{int(parts[1]):02d}.{sub.replace('_modulation', '')}"
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
