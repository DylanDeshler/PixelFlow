"""
Fixed-budget quadtree tokenization for JiT (research_proposal_fixed_budget_quadtree_patch_allocation.md): every image is
tokenized into exactly K multi-scale patches (32 / 16 / 8 px at 128px with 3 levels), placed by a router, and the same
JiT backbone (full Attention Residuals, adaLN-single) runs on those K tokens. A separate model so that nothing here can
change the baseline: the pieces that differ from JiT (attention with an area bias and per-token RoPE, the block, the
multi-scale embedding / heads, the forward) are copied and adapted; unmodified leaf modules are imported from model_jit.

Geometry. Levels l = 0..L-1 have patch size p_l = patch_size * 2^(L-1-l) (patch_size is the finest) on a G_l x G_l grid.
Candidates (every patch at every level) are numbered level-major, row-major: [level 0 | level 1 | ... ]. A layout is the
K candidate indices an image is tokenized with; a valid layout is a quadtree with exactly m = (K - N0) / 3 splits.

Positions (Section 3.1 of the proposal), in finest-grid units: RoPE on the physical patch centres, the footprint-integrated
2D sin-cos as the absolute embedding (sin(w c) sinc(w width / 2), likewise cos), a learned level embedding, and
proportional attention (+log area on every key's logits). Ablations: pos_embed='center' (sin-cos at the centre), rope=
'integrated' (each token's rotary pairs damped by sinc(w width / 2) on q and k), area_bias=False, level_embed=False.

Routing (Section 4): the router turns the noisy input and t into a layout, under no_grad, outside the autograd graph and
outside torch.compile. 'dino': frozen DINOv2 on a frozen baseline JiT's clean-image estimate (angular divergence of the
child quadrants' features x a PCA-1 foreground gate, standardized per level, hierarchical greedy). 'pixel': mean gradient
magnitude of the baseline's estimate. 'random': a pseudo-random function of the input (identical inputs, e.g. the two
halves of a CFG batch, get identical layouts). 'uniform': coarse-first, centre-out (K=64 at 128px is uniform 16px
patches, K=256 uniform 8px): the uniform baselines on exactly this code path. The router is not a registered submodule:
it is never trained, EMA'd or saved, and is rebuilt from the architecture fields when a checkpoint is loaded.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_jit import TimestepEmbedder, LabelEmbedder, SwiGLUFFN, FinalLayer, BottleneckPatchEmbed, AttnResState, modulate, make_norm
from util.model_util import rotate_half


# -----------------------------------------------------------------------------
# geometry

class QuadGeometry:
    """
    Candidate tables for a quadtree over an img x img canvas with `levels` levels and finest patch `patch` (all on CPU;
    the model turns the ones it needs into buffers).
    """
    def __init__(self, img, patch, levels):
        self.L = levels
        self.patches = [patch * 2 ** (levels - 1 - l) for l in range(levels)]
        assert img % self.patches[0] == 0, f'image size {img} not divisible by the coarsest patch {self.patches[0]}'
        self.grids = [img // p for p in self.patches]
        self.counts = [g * g for g in self.grids]
        self.offsets = [sum(self.counts[:l]) for l in range(levels)]
        self.n = sum(self.counts)
        self.n0 = self.counts[0]
        self.n_split = sum(self.counts[:-1]) # splittable candidates: every level but the finest
        level, index, row, col = [], [], [], []
        for l, g in enumerate(self.grids):
            r, c = torch.meshgrid(torch.arange(g), torch.arange(g), indexing='ij')
            level.append(torch.full((g * g,), l))
            index.append(torch.arange(g * g))
            row.append(r.flatten())
            col.append(c.flatten())
        self.level, self.index_in_level = torch.cat(level), torch.cat(index)
        row, col = torch.cat(row), torch.cat(col)
        self.width = torch.tensor([2 ** (levels - 1 - l) for l in range(levels)], dtype=torch.float32)[self.level] # finest units
        self.cy, self.cx = (row.float() + 0.5) * self.width, (col.float() + 0.5) * self.width
        # children of every splittable candidate [n_split, 4] and parent of every candidate (-1 at level 0)
        children = []
        parent = torch.full((self.n,), -1, dtype=torch.long)
        for l in range(levels - 1):
            g, g2 = self.grids[l], self.grids[l + 1]
            for i in range(self.counts[l]):
                r, c = divmod(i, g)
                kids = [self.offsets[l + 1] + (2 * r + dr) * g2 + (2 * c + dc) for dr in (0, 1) for dc in (0, 1)]
                children.append(kids)
                parent[kids] = self.offsets[l] + i
        self.children = torch.tensor(children, dtype=torch.long).view(-1, 4)
        self.parent = parent

    def max_tokens(self):
        return self.counts[-1]

    def check_budget(self, K):
        assert (K - self.n0) % 3 == 0 and self.n0 <= K <= self.max_tokens(), \
            f'tokens must be {self.n0} + 3m with at most {self.max_tokens()} (got {K}): each split replaces 1 patch by 4'
        return (K - self.n0) // 3

def allocate(scores, K, geo, children, parent):
    """
    Hierarchical greedy (Section 4.4): starting from the level-0 patches, repeatedly split the highest-scoring splittable
    leaf (its children become splittable), m = (K - N0) / 3 times. scores: [B, n_split]. Returns the layout [B, K]: the
    candidate indices of the leaves, ascending. Vectorized over the batch; m iterations.
    """
    B, device = scores.shape[0], scores.device
    m = geo.check_budget(K)
    available = torch.zeros(B, geo.n, dtype=torch.bool, device=device)
    available[:, :geo.n0] = True
    split = torch.zeros(B, geo.n, dtype=torch.bool, device=device)
    scores = scores.float()
    rows = torch.arange(B, device=device)
    for _ in range(m):
        s = scores.masked_fill(~available[:, :geo.n_split], float('-inf'))
        j = s.argmax(1)
        split[rows, j] = True
        available[rows, j] = False
        available[rows[:, None], children[j]] = True # children of a split node become candidates (finest ones never split)
    in_tree = torch.ones(B, geo.n, dtype=torch.bool, device=device)
    in_tree[:, geo.n0:] = split[:, parent[geo.n0:]]
    leaves = in_tree & ~split
    return torch.arange(geo.n, device=device).expand(B, -1)[leaves].view(B, K)


def cell_maps(layout, geo):
    """
    Which token covers each finest-grid cell, and at which level: (token position in the layout, level), each [B, G, G].
    """
    B, K = layout.shape
    G, dev = geo.grids[-1], layout.device
    lvl, pos = geo.level.to(dev)[layout], geo.index_in_level.to(dev)[layout]
    tok = torch.full((B, G, G), -1, dtype=torch.long, device=dev)
    lev = torch.full((B, G, G), -1, dtype=torch.long, device=dev)
    k = torch.arange(K, device=dev).expand(B, K)
    for l, g in enumerate(geo.grids):
        slot = torch.where(lvl == l, pos, g * g)
        grid = torch.full((B, g * g + 1), -1, dtype=torch.long, device=dev).scatter(1, slot, k)[:, :g * g].view(B, g, g)
        f = G // g
        grid = grid.repeat_interleave(f, 1).repeat_interleave(f, 2)
        tok = torch.where(grid >= 0, grid, tok)
        lev = torch.where(grid >= 0, torch.full_like(lev, l), lev)
    return tok, lev

def layout_agreement(a, b, geo):
    # fraction of the image area tokenized at the same level by layouts a and b, per image [B]
    return (cell_maps(a, geo)[1] == cell_maps(b, geo)[1]).float().flatten(1).mean(1)

LEVEL_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'] # coarsest first (plot_style.CATEGORICAL order)

def draw_layout(images, layout, geo, colors=LEVEL_COLORS):
    """
    images [N, 3, H, W] in [-1, 1] with every token's boundary drawn (1px), coloured by the finer of the two tokens it
    separates (LEVEL_COLORS: coarsest first).
    """
    tok, lev = cell_maps(layout.to(images.device), geo)
    f = images.shape[-1] // geo.grids[-1]
    tok = tok.repeat_interleave(f, 1).repeat_interleave(f, 2)
    lev = lev.repeat_interleave(f, 1).repeat_interleave(f, 2)
    edge = torch.full_like(lev, -1)
    right = tok[:, :, :-1] != tok[:, :, 1:]
    edge[:, :, :-1] = torch.where(right, torch.maximum(lev[:, :, :-1], lev[:, :, 1:]), edge[:, :, :-1])
    down = tok[:, :-1] != tok[:, 1:]
    edge[:, :-1] = torch.where(down, torch.maximum(torch.maximum(lev[:, :-1], lev[:, 1:]), edge[:, :-1]), edge[:, :-1])
    out = images.clone()
    for l in range(geo.L):
        rgb = torch.tensor([int(colors[l][i:i + 2], 16) / 127.5 - 1 for i in (1, 3, 5)], device=images.device, dtype=images.dtype)
        out = torch.where((edge == l).unsqueeze(1), rgb.view(1, 3, 1, 1), out)
    return out


# -----------------------------------------------------------------------------
# positions

def integrated_sincos(dim, cy, cx, width, integrated=True):
    """
    2D sin-cos absolute embedding at continuous centres (half the channels per axis, MAE's layout), averaged over each
    token's box footprint when integrated: per axis and frequency sin(w c) sinc(w width / 2), likewise cos (Mip-NeRF's
    integrated positional encoding for boxes). A parent's embedding is then exactly the mean of its four children's.
    """
    assert dim % 4 == 0
    omega = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float64) / (dim // 4))
    out = []
    for c in (cy, cx):
        ang = c.double()[:, None] * omega[None]
        damp = torch.sinc(width.double()[:, None] * omega[None] / 2 / math.pi) if integrated else 1.0 # torch.sinc(x) = sin(pi x) / (pi x)
        out += [torch.sin(ang) * damp, torch.cos(ang) * damp]
    return torch.cat(out, dim=1).float()

def rope_tables(head_dim, cy, cx, width, integrated=False, theta=10000):
    """
    cos / sin tables [N, head_dim] for 2D RoPE at continuous centres, in JiT's layout (VisionRotaryEmbeddingFast: first half
    of the head dims rotates with the row coordinate, second half with the column, frequencies repeated per pair).
    integrated: each rotary pair scaled by sinc(w width / 2) (on q and k alike, so q . k stays a function of the
    displacement, damped for coarse tokens).
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, 2)[:half // 2].double() / half))
    cos, sin = [], []
    for c in (cy, cx):
        ang = c.double()[:, None] * freqs[None]
        damp = torch.sinc(width.double()[:, None] * freqs[None] / 2 / math.pi) if integrated else torch.ones_like(ang)
        cos.append((torch.cos(ang) * damp).repeat_interleave(2, dim=1))
        sin.append((torch.sin(ang) * damp).repeat_interleave(2, dim=1))
    return torch.cat(cos, dim=1).float(), torch.cat(sin, dim=1).float()

class TokenRope:
    """
    Per-token rotary embedding (cos / sin [B, 1, K, head_dim]) plus the attention bias added to every query's logits for
    each key ([B, 1, 1, K] or None), passed through the blocks where JiT passes its grid RoPE.
    """
    def __init__(self, cos, sin, bias):
        self.cos, self.sin, self.bias = cos, sin, bias

    def __call__(self, t):
        return t * self.cos + rotate_half(t) * self.sin


# -----------------------------------------------------------------------------
# blocks (copied from model_jit: Attention / JiTBlock, with the key bias)

class QuadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0., layernorm=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = make_norm(head_dim, layernorm) if qk_norm else nn.Identity()
        self.k_norm = make_norm(head_dim, layernorm) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = rope(self.q_norm(q)), rope(self.k_norm(k))
        # JiT's attention: fp32 logits and softmax; plus the per-key bias (proportional attention)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = q.float() @ k.float().transpose(-2, -1) / math.sqrt(q.shape[-1])
            if rope.bias is not None:
                logits = logits + rope.bias
            w = torch.softmax(logits, dim=-1)
        w = torch.dropout(w, self.attn_drop.p if self.training else 0.0, train=True)
        x = (w.to(v.dtype) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class QuadBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, adaln_single=False, layernorm=False):
        super().__init__()
        self.norm1 = make_norm(hidden_size, layernorm)
        self.attn = QuadAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                                  attn_drop=attn_drop, proj_drop=proj_drop, layernorm=layernorm)
        self.norm2 = make_norm(hidden_size, layernorm)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaln_single = adaln_single
        if adaln_single:
            self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def modulation(self, c):
        if self.adaln_single:
            return (self.scale_shift_table[None] + c.view(c.shape[0], 6, -1)).to(c.dtype).unbind(1)
        return self.adaLN_modulation(c).chunk(6, dim=-1)

    def forward(self, x, c, rope, res=None, index=None):
        if res is None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(c)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope)
            return x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # Attention Residuals: returns the next sublayer's (depth-mixed) input, as JiTBlock.forward_attn_res
        res.add(self.attn_branch(x, c, rope))
        res.add(self.mlp_branch(res.mix(2 * index + 1), c))
        return res.mix(2 * index + 2)

    def attn_branch(self, x, c, rope):
        shift_msa, scale_msa, gate_msa = self.modulation(c)[:3]
        return gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope)

    def mlp_branch(self, x, c):
        shift_mlp, scale_mlp, gate_mlp = self.modulation(c)[3:]
        return gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


# -----------------------------------------------------------------------------
# router

class QuadRouter:
    """
    Layouts from the noisy input and t (Section 4), on the device of the model that owns it. Not an nn.Module of the model.
    """
    def __init__(self, geo, K, mode, baseline='', dino='dinov2_vitb14_reg', t_min=0.0, gate_tau=1.0, num_classes=1000, img=128):
        assert mode in ('dino', 'pixel', 'random', 'uniform'), f"router must be dino, pixel, random or uniform, got {mode}"
        self.geo, self.K, self.mode, self.t_min, self.gate_tau, self.num_classes = geo, K, mode, t_min, gate_tau, num_classes
        geo.check_budget(K)
        self.children, self.parent, self.level = geo.children, geo.parent, geo.level
        # neutral scores: coarse levels first, then centre-out within a level (uniform layouts; also t < t_min)
        centre = geo.grids[-1] / 2
        dist = torch.sqrt((geo.cy - centre) ** 2 + (geo.cx - centre) ** 2)[:geo.n_split]
        self.neutral = (geo.L - 1 - geo.level[:geo.n_split]).float() * 1e3 - dist
        g = torch.Generator().manual_seed(0)
        self.rand_w = torch.randn(geo.n_split, 3, generator=g) # pseudo-random scores: fixed projection of each node's input mean
        self.rand_b = torch.rand(geo.n_split, generator=g) * 100
        self.baseline = self.dino = None
        if mode in ('dino', 'pixel'):
            assert baseline, f"router '{mode}' needs route_baseline: a checkpoint of a uniform JiT that predicts the clean image"
            from jit_common import load_denoiser # deferred: jit_common imports this module
            base, _ = load_denoiser(baseline, ema=1, device='cpu')
            assert base.img_size == img and base.num_classes == num_classes, \
                f'route_baseline is {base.img_size}px / {base.num_classes} classes, the model {img}px / {num_classes}'
            self.baseline = base.net.eval().requires_grad_(False)
        if mode == 'dino':
            self.load_dino(dino)
        self.route_every, self.sampling, self.calls, self.cache = 1, False, 0, None
        self.record, self.history = False, [] # record: every layout produced, with its mean t (sample-grid visualizations)

    def load_dino(self, name):
        # frozen DINOv2 from torch.hub, run at one DINO patch per finest-level patch (224px at 128px / 8px patches)
        self.dino = torch.hub.load('facebookresearch/dinov2', name).eval().requires_grad_(False)
        ps = self.dino.patch_embed.patch_size
        self.dino_patch = ps[0] if isinstance(ps, tuple) else ps
        self.dino_res = self.dino_patch * self.geo.grids[-1]
        return self

    def to(self, device):
        for k in ('children', 'parent', 'level', 'neutral', 'rand_w', 'rand_b'):
            setattr(self, k, getattr(self, k).to(device))
        for m in (self.baseline, self.dino):
            if m is not None:
                m.to(device)
        return self

    # --- scores (higher = split first), [B, n_split]

    def node_means(self, img):
        # per-node mean of img [B, C, H, W] for every splittable level: [B, n_split, C]
        return torch.cat([F.avg_pool2d(img, p).flatten(2).transpose(1, 2) for p in self.geo.patches[:-1]], dim=1)

    def standardize(self, s):
        # z-score within each level, per image, so scores compare across levels
        out = []
        for l in range(self.geo.L - 1):
            x = s[:, self.geo.offsets[l]:self.geo.offsets[l] + self.geo.counts[l]]
            out.append((x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6))
        return torch.cat(out, dim=1)

    def random_scores(self, z):
        v = (self.node_means(z.float()) * self.rand_w).sum(-1) * 12.9898 + self.rand_b
        return torch.frac(torch.sin(v) * 43758.5453).abs()

    def pixel_scores(self, x0):
        # mean gradient magnitude of the estimate over each node's footprint
        x0 = x0.float()
        gy = F.pad((x0[:, :, 1:] - x0[:, :, :-1]).abs().mean(1, keepdim=True), (0, 0, 0, 1))
        gx = F.pad((x0[:, :, :, 1:] - x0[:, :, :, :-1]).abs().mean(1, keepdim=True), (0, 1, 0, 0))
        return self.standardize(self.node_means(torch.sqrt(gx ** 2 + gy ** 2))[..., 0])

    def dino_features(self, x0):
        # DINOv2 patch features [B, C, G, G] of the estimate, one per finest patch
        x = F.interpolate((x0.float().clamp(-1, 1) + 1) / 2, size=(self.dino_res, self.dino_res), mode='bicubic', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        f = self.dino.forward_features((x - mean) / std)['x_norm_patchtokens'].float() # [B, G*G, C]
        G = self.geo.grids[-1]
        return f.transpose(1, 2).reshape(f.shape[0], -1, G, G)

    def dino_scores(self, x0, parts=False):
        """
        Split scores from DINOv2 on the estimate x0. parts: also return the intermediate maps (for viz_quadtree_router.py):
        proj [B, G, G] (standardized PCA-1 projection, foreground positive), gate_cells [B, G, G], div / gate / raw [B, n_split].
        """
        feat = self.dino_features(x0) # [B, C, G, G]
        B, C, G, _ = feat.shape
        geo = self.geo
        # Micro: angular divergence of each splittable node's four child-quadrant features (mean of their DINO patches)
        div = []
        for l in range(geo.L - 1):
            child = F.avg_pool2d(feat, geo.patches[l + 1] // geo.patches[-1]) # level l+1 features [B, C, 2 g, 2 g]
            g = geo.grids[l]
            kids = F.normalize(child.view(B, C, g, 2, g, 2).permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, 4, C), dim=-1)
            cos = kids @ kids.transpose(-1, -2) # [B, g*g, 4, 4]
            div.append(1 - (cos.sum((-1, -2)) - 4) / 12) # mean over the 6 pairs (off-diagonal entries / 2)
        div = torch.cat(div, dim=1)
        # Macro: foreground gate from the first principal direction of the image's patch features (power iteration)
        X = feat.flatten(2).transpose(1, 2) # [B, G*G, C]
        Xc = X - X.mean(1, keepdim=True)
        v = F.normalize(torch.ones(B, C, device=X.device), dim=-1)
        for _ in range(20):
            v = F.normalize(torch.einsum('bnc,bn->bc', Xc, torch.einsum('bnc,bc->bn', Xc, v)), dim=-1)
        proj = torch.einsum('bnc,bc->bn', Xc, v).view(B, 1, G, G)
        proj = proj / (proj.flatten(1).std(1).view(B, 1, 1, 1) + 1e-6)
        # sign: the foreground side has the larger mean over the centre than over the border ring
        inner = torch.zeros(G, G, dtype=torch.bool, device=X.device)
        inner[G // 4:G - G // 4, G // 4:G - G // 4] = True
        ring = torch.ones(G, G, dtype=torch.bool, device=X.device)
        ring[1:-1, 1:-1] = False
        flip = (proj[:, 0][:, inner].mean(1) < proj[:, 0][:, ring].mean(1)).float() * -2 + 1
        proj = proj * flip.view(B, 1, 1, 1)
        gate = torch.sigmoid(torch.cat([F.avg_pool2d(proj, p // geo.patches[-1]).flatten(1) for p in geo.patches[:-1]], dim=1) / self.gate_tau)
        scores = self.standardize(gate * div)
        if parts:
            return scores, dict(proj=proj[:, 0], gate_cells=torch.sigmoid(proj[:, 0] / self.gate_tau), div=div, gate=gate, raw=gate * div)
        return scores

    # --- layouts

    @torch.no_grad()
    def clean_layout(self, x):
        """
        The layout this router's scorer gives the clean image itself (a diagnostic reference: it can't exist at sampling
        time); None for routers without an image scorer (random, uniform).
        """
        if self.mode not in ('dino', 'pixel'):
            return None
        scores = self.dino_scores(x) if self.mode == 'dino' else self.pixel_scores(x)
        return allocate(scores, self.K, self.geo, self.children, self.parent)

    @torch.no_grad()
    def __call__(self, z, t):
        B = z.shape[0]
        if self.sampling and self.route_every > 1 and self.cache is not None and self.cache.shape[0] == B and self.calls % self.route_every:
            self.calls += 1
            if self.record:
                self.history.append((self.cache.cpu(), float(t.float().mean())))
            return self.cache
        if self.mode == 'uniform':
            scores = self.neutral.expand(B, -1)
        elif self.mode == 'random':
            scores = self.random_scores(z)
        else:
            y_null = torch.full((B,), self.num_classes, device=z.device, dtype=torch.long) # one layout for a CFG pair
            x0 = self.baseline(z, t, y_null)
            scores = self.pixel_scores(x0) if self.mode == 'pixel' else self.dino_scores(x0)
        if self.t_min > 0 and self.mode != 'uniform':
            scores = torch.where((t < self.t_min).view(B, 1), self.neutral.expand(B, -1), scores)
        idx = allocate(scores, self.K, self.geo, self.children, self.parent)
        if self.sampling:
            self.cache, self.calls = idx, self.calls + 1
        if self.record:
            self.history.append((idx.cpu(), float(t.float().mean())))
        return idx


# -----------------------------------------------------------------------------
# model

class QuadtreeJiT(nn.Module):
    family = 'quadtree'

    def __init__(self, input_size=128, patch_size=8, levels=3, tokens=64, in_channels=3, hidden_size=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, num_classes=1000, bottleneck_dim=128, attn_res='none',
                 attn_res_block_size=0, adaln_single=False, layernorm=False, router='uniform', route_baseline='',
                 route_dino='dinov2_vitb14_reg', route_t_min=0.0, route_gate_tau=1.0, pos_embed='integrated', rope='plain',
                 area_bias=True, level_embed=True, in_context_len=0, in_context_start=0, family='quadtree'):
        super().__init__()
        assert in_context_len == 0, 'QuadtreeJiT has no in-context class tokens'
        assert pos_embed in ('integrated', 'center') and rope in ('plain', 'integrated')
        assert attn_res in ('none', 'full', 'block')
        self.input_size, self.patch_size, self.in_channels, self.out_channels = input_size, patch_size, in_channels, in_channels
        self.hidden_size, self.num_heads, self.num_classes, self.tokens = hidden_size, num_heads, num_classes, tokens
        self.in_context_len, self.in_context_start = 0, 0 # interface shared with JiT (probes, eval_linear)
        self.adaln_single, self.attn_res, self.attn_res_adaln = adaln_single, attn_res, False
        self.attn_res_block_size = attn_res_block_size or max(1, round(depth / 8))
        self.skip_pathway, self.skip_stop_grad = False, False
        self.geo = geo = QuadGeometry(input_size, patch_size, levels)
        geo.check_budget(tokens)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        if adaln_single:
            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        # one bottleneck patch embedding and one final layer per level
        self.x_embedders = nn.ModuleList([BottleneckPatchEmbed(input_size, p, in_channels, bottleneck_dim, hidden_size, bias=True) for p in geo.patches])
        self.final_layers = nn.ModuleList([FinalLayer(hidden_size, p, in_channels, adaln_single=adaln_single, layernorm=layernorm) for p in geo.patches])
        self.level_embed = nn.Parameter(torch.zeros(levels, hidden_size)) if level_embed else None
        self.blocks = nn.ModuleList([
            QuadBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                      attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                      proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0, adaln_single=adaln_single, layernorm=layernorm)
            for i in range(depth)])
        if attn_res != 'none':
            self.attn_res_queries = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(2 * depth + 1)])
            self.attn_res_norms = nn.ModuleList([make_norm(hidden_size, layernorm) for _ in range(2 * depth + 1)])

        # per-candidate tables (fixed): absolute embedding, RoPE, log area, level, position within the level
        self.register_buffer('pos_table', integrated_sincos(hidden_size, geo.cy, geo.cx, geo.width, pos_embed == 'integrated'), persistent=False)
        cos, sin = rope_tables(hidden_size // num_heads, geo.cy, geo.cx, geo.width, integrated=rope == 'integrated')
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)
        self.register_buffer('log_area', torch.log(geo.width ** 2) if area_bias else torch.zeros(geo.n), persistent=False)
        self.area_bias = area_bias
        self.register_buffer('cand_level', geo.level, persistent=False)
        self.register_buffer('cand_index', geo.index_in_level, persistent=False)
        self.initialize_weights()
        # the router: rebuilt from these fields on every construction (checkpoint loading included), never saved
        self.router = QuadRouter(geo, tokens, router, route_baseline, route_dino, route_t_min, route_gate_tau, num_classes, input_size)

    # the router is a plain attribute (not a submodule): keep it out of parameters / state_dict / EMA, move it with the model
    def __setattr__(self, name, value):
        if name == 'router':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        if 'router' in self.__dict__:
            self.router.to(self.pos_table.device)
        return out

    def initialize_weights(self):
        # JiT's initialization (adaLN-single: zero shared projection, zero attention / MLP output projections; final
        # layers zero, so the model starts at a zero prediction)
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        for emb in self.x_embedders:
            nn.init.xavier_uniform_(emb.proj1.weight.data.view([emb.proj1.weight.shape[0], -1]))
            nn.init.xavier_uniform_(emb.proj2.weight.data.view([emb.proj2.weight.shape[0], -1]))
            nn.init.constant_(emb.proj2.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.level_embed is not None:
            nn.init.normal_(self.level_embed, std=0.02)
        if self.adaln_single:
            nn.init.constant_(self.t_block[-1].weight, 0)
            nn.init.constant_(self.t_block[-1].bias, 0)
            for block in self.blocks:
                nn.init.constant_(block.attn.proj.weight, 0)
                nn.init.constant_(block.mlp.w3.weight, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for fl in self.final_layers:
            if not fl.adaln_single:
                nn.init.constant_(fl.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(fl.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(fl.linear.weight, 0)
            nn.init.constant_(fl.linear.bias, 0)

    # --- routing interface

    @torch.compiler.disable
    def route(self, x, t):
        """
        Layout [B, K] (candidate indices) for the noisy input x and t, from the router.
        """
        return self.router(x, t)

    @torch.compiler.disable
    def route_clean(self, x):
        # the layout the router's scorer gives the clean image (diagnostics only); None for random / uniform routers
        return self.router.clean_layout(x)

    @property
    def route_every(self):
        return self.router.route_every

    @route_every.setter
    def route_every(self, n):
        self.router.route_every = max(1, int(n))

    def begin_sampling(self):
        # sampling only: layouts may be reused for route_every network calls (reset at the start of every trajectory)
        self.router.sampling, self.router.calls, self.router.cache = True, 0, None

    def end_sampling(self):
        self.router.sampling, self.router.cache = False, None

    def record_layouts(self, on=True):
        # start (clearing) or stop recording every layout the router produces: self.router.history = [(layout [B, K], mean t), ...]
        self.router.record = on
        if on:
            self.router.history = []

    def level_area(self, layout):
        # fraction of the image area tokenized at each level [B, L] (rows sum to 1)
        lvl = self.cand_level[layout]
        w2 = 4.0 ** (self.geo.L - 1 - lvl).float() # token area in finest patches
        return torch.stack([(w2 * (lvl == l)).sum(1) for l in range(self.geo.L)], dim=1) / self.geo.counts[-1]

    # --- forward

    def embed(self, x, layout):
        # every candidate at every level embedded (cheap), the layout's K gathered, plus absolute position and level
        cand = torch.cat([emb(x) for emb in self.x_embedders], dim=1) # [B, n, D]
        tokens = cand.gather(1, layout.unsqueeze(-1).expand(-1, -1, cand.shape[-1]))
        add = self.pos_table[layout]
        if self.level_embed is not None:
            add = add + self.level_embed[self.cand_level[layout]]
        return tokens + add.to(tokens.dtype)

    def positions(self, layout):
        bias = self.log_area[layout][:, None, None, :] if self.area_bias else None # [B, 1, 1, K]
        return TokenRope(self.rope_cos[layout].unsqueeze(1), self.rope_sin[layout].unsqueeze(1), bias)

    def unpatchify(self, x, p, g):
        x = x.reshape(x.shape[0], g, g, p, p, self.out_channels)
        return torch.einsum('nhwpqc->nchpwq', x).reshape(x.shape[0], self.out_channels, g * p, g * p)

    def output(self, h, c, layout):
        # each level's final layer on every token, each token's patch scattered to its place; levels tile the canvas
        lvl, pos = self.cand_level[layout], self.cand_index[layout]
        img = 0
        for l, (layer, p, g) in enumerate(zip(self.final_layers, self.geo.patches, self.geo.grids)):
            o = layer(h, c) # [B, K, p * p * C]
            slot = torch.where(lvl == l, pos, g * g) # other levels' tokens go to a discarded extra slot
            canvas = o.new_zeros(o.shape[0], g * g + 1, o.shape[-1]).scatter(1, slot.unsqueeze(-1).expand_as(o), o)
            img = img + self.unpatchify(canvas[:, :g * g], p, g)
        return img

    def forward(self, x, t, y, layout=None):
        if layout is None:
            layout = self.route(x, t)
        t_emb = self.t_embedder(t)
        c = t_emb + self.y_embedder(y)
        c_blocks = self.t_block(c) if self.adaln_single else c
        h = self.embed(x, layout)
        rope = self.positions(layout)
        res = AttnResState(self, h, c) if self.attn_res != 'none' else None
        for i, block in enumerate(self.blocks):
            h = block(h, c_blocks, rope, res=res, index=i)
        return self.output(h, c, layout)


# presets: JiT-B's backbone, named by the patch range (32px coarsest, 8px finest; patch_size is the finest level)
QUADTREE_PRESETS = {
    'QuadJiT-B/32-8': dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, patch_size=8, levels=3, tokens=64),
}
