"""
Gaussian-placed tokens for JiT (deformable patch embedding): each of the K tokens is an oriented, scaled patch whose
centre, size and orientation a small placer network predicts from the noisy input in the same forward pass, trained only
by the denoising loss. A separate model (as model_quadtree.py), so nothing here can change the baseline: JiT's blocks,
Attention Residuals, adaLN-single, the bottleneck patch embedding and the final layer are reused unchanged in form.

Geometry. Anchors on a grid with spacing S = patch_size (K = (input_size / S)^2: 64 at 128px / 16px). Token k's frame:
  centre      mu_k = anchor_k + max_offset * S * tanh(o)                         (per axis)
  half-size   h_k  = extent * S * scale_range^tanh(s)                            (per axis; extent 0.75: 1.5 S at init)
  rotation    theta_k = max_rotation * tanh(r)
with (o, s, r) the placer's 5 outputs per anchor, zero at init (the last layer is zero-initialized), so training starts from
a regular grid of overlapping square frames. Bounded parameters keep tokens from collapsing or wandering off without any
auxiliary loss.

Encoder: each token samples a canvas x canvas grid over its frame from the noisy image (bilinear, differentiable in the
frame), then the bottleneck patch embedding (linear canvas -> bottleneck_dim -> hidden), as JiT's patch embedding.
Decoder: each token's final layer predicts a canvas x canvas x 3 patch in its frame; every pixel blends the tokens covering
it, weighted by a Hann window of its frame coordinates (cos^2(pi u / 2) cos^2(pi v / 2), smooth and zero at the frame's
edge) and normalized over tokens (a partition of unity). A tiny weight for each pixel's own anchor's token keeps every
pixel covered whatever the geometry. Rendering is local (a fixed window around each token) and scattered back.
Positions: the footprint-integrated sin-cos and RoPE at the continuous centres (in anchor units, detached: geometry is
learned through the sampling and the blending), a learned shape embedding of (log scales, rotation), and the area
attention bias (log of the frame's area relative to the anchor cell).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_jit import TimestepEmbedder, LabelEmbedder, FinalLayer, AttnResState, make_norm
from model_quadtree import QuadBlock, TokenRope, integrated_sincos, rope_tables


class CanvasEmbed(nn.Module):
    # JiT's bottleneck patch embedding on a per-token canvas: linear (C * canvas^2 -> bottleneck), linear (-> hidden)
    def __init__(self, canvas, in_chans, pca_dim, embed_dim):
        super().__init__()
        self.proj1 = nn.Linear(in_chans * canvas * canvas, pca_dim, bias=False)
        self.proj2 = nn.Linear(pca_dim, embed_dim, bias=True)

    def forward(self, patches): # [B, K, C * canvas^2]
        return self.proj2(self.proj1(patches))


class Placer(nn.Module):
    """
    Geometry from the noisy input: a 4x-downsampling patch conv, residual 3x3 conv blocks modulated by t (FiLM), average
    pooled to the anchor grid, and a zero-initialized 1x1 conv to 5 numbers per anchor (offset x / y, log-scale x / y,
    rotation, before the bounding tanh).
    """
    def __init__(self, in_chans, patch_size, width, depth, cond_dim):
        super().__init__()
        self.stride = max(1, patch_size // 4)
        self.stem = nn.Conv2d(in_chans, width, self.stride, stride=self.stride)
        self.norms = nn.ModuleList([nn.GroupNorm(8 if width % 8 == 0 else 1, width) for _ in range(depth)])
        self.convs = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1) for _ in range(depth)])
        self.films = nn.ModuleList([nn.Linear(cond_dim, 2 * width) for _ in range(depth)])
        self.pool = patch_size // self.stride
        self.head = nn.Conv2d(width, 5, 1)

    def forward(self, x, c):
        h = self.stem(x)
        for norm, conv, film in zip(self.norms, self.convs, self.films):
            g, b = film(F.silu(c))[:, :, None, None].chunk(2, dim=1)
            h = h + conv(F.silu(norm(h) * (1 + g) + b))
        h = F.avg_pool2d(h, self.pool)
        return self.head(h).flatten(2).transpose(1, 2) # [B, K, 5]


class GaussJiT(nn.Module):
    family = 'gauss'

    def __init__(self, input_size=128, patch_size=16, in_channels=3, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0, num_classes=1000, bottleneck_dim=128, attn_res='none', attn_res_block_size=0,
                 adaln_single=False, layernorm=False, extent=0.75, canvas=0, max_offset=0.25, scale_range=1.33,
                 max_rotation=math.pi / 4, placer_width=64, placer_depth=2, area_bias=True, shape_embed=True,
                 in_context_len=0, in_context_start=0, family='gauss'):
        super().__init__()
        assert in_context_len == 0, 'GaussJiT has no in-context class tokens'
        assert attn_res in ('none', 'full', 'block')
        assert input_size % patch_size == 0 and patch_size % 4 == 0
        self.input_size, self.patch_size, self.in_channels, self.out_channels = input_size, patch_size, in_channels, in_channels
        self.hidden_size, self.num_heads, self.num_classes = hidden_size, num_heads, num_classes
        self.in_context_len, self.in_context_start = 0, 0 # interface shared with JiT (probes, eval_linear)
        self.adaln_single, self.attn_res, self.attn_res_adaln = adaln_single, attn_res, False
        self.attn_res_block_size = attn_res_block_size or max(1, round(depth / 8))
        self.skip_pathway, self.skip_stop_grad = False, False
        self.extent, self.max_offset, self.scale_range, self.max_rotation = extent, max_offset, scale_range, max_rotation
        self.canvas = canvas or int(round(2 * extent * patch_size)) # ~1 sample per pixel at the initial frame size
        self.grid = input_size // patch_size
        self.tokens = self.grid ** 2
        self.area_bias = area_bias
        # rendering window around each token: covers the largest (rotated) frame, plus a pixel for rounding
        h_max = extent * patch_size * scale_range
        reach = h_max * (math.cos(min(max_rotation, math.pi / 4)) + math.sin(min(max_rotation, math.pi / 4)))
        self.window = 2 * math.ceil(reach) + 3

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        if adaln_single:
            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.placer = Placer(in_channels, patch_size, placer_width, placer_depth, hidden_size)
        self.x_embedder = CanvasEmbed(self.canvas, in_channels, bottleneck_dim, hidden_size)
        self.final_layer = FinalLayer(hidden_size, self.canvas, in_channels, adaln_single=adaln_single, layernorm=layernorm)
        self.shape_mlp = nn.Sequential(nn.Linear(4, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)) if shape_embed else None
        self.blocks = nn.ModuleList([
            QuadBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                      attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                      proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0, adaln_single=adaln_single, layernorm=layernorm)
            for i in range(depth)])
        if attn_res != 'none':
            self.attn_res_queries = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(2 * depth + 1)])
            self.attn_res_norms = nn.ModuleList([make_norm(hidden_size, layernorm) for _ in range(2 * depth + 1)])

        g = self.grid
        iy, ix = torch.meshgrid(torch.arange(g), torch.arange(g), indexing='ij')
        anchors = torch.stack([ix.flatten(), iy.flatten()], -1).float() * patch_size + patch_size / 2 # [K, 2] (x, y) pixels
        self.register_buffer('anchors', anchors, persistent=False)
        r = (torch.arange(self.canvas, dtype=torch.float32) + 0.5) / self.canvas * 2 - 1 # canvas sample positions in [-1, 1]
        v, u = torch.meshgrid(r, r, indexing='ij')
        self.register_buffer('canvas_uv', torch.stack([u, v], -1), persistent=False) # [canvas, canvas, 2] (u along x, v along y)
        w = torch.arange(self.window, dtype=torch.float32) - (self.window - 1) / 2
        wy, wx = torch.meshgrid(w, w, indexing='ij')
        self.register_buffer('window_offsets', torch.stack([wx, wy], -1), persistent=False) # [W, W, 2]
        self.last_geometry = None # the last forward's (mu, h, theta), detached: for visualizations and logging
        self.initialize_weights()

    def __setattr__(self, name, value):
        if name == 'last_geometry': # a plain attribute, never part of the module state
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def initialize_weights(self):
        # JiT's initialization (adaLN-single: zero shared projection, zero attention / MLP output projections; the final
        # layer zero, so the model starts at a zero prediction); the placer's head, its FiLMs and the shape embedding zero,
        # so the geometry starts at the anchor grid
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.placer.head.weight, 0)
        nn.init.constant_(self.placer.head.bias, 0)
        for film in self.placer.films:
            nn.init.constant_(film.weight, 0)
            nn.init.constant_(film.bias, 0)
        if self.shape_mlp is not None:
            nn.init.constant_(self.shape_mlp[-1].weight, 0)
            nn.init.constant_(self.shape_mlp[-1].bias, 0)
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
        if not self.final_layer.adaln_single:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # --- geometry

    def geometry(self, x, c):
        """
        Token frames from the noisy input: centres mu [B, K, 2] (x, y pixels), half-sizes h [B, K, 2], rotations theta [B, K],
        and the bounded log-scales [B, K, 2].
        """
        raw = self.placer(x, c).float()
        S = self.patch_size
        mu = self.anchors + self.max_offset * S * torch.tanh(raw[..., 0:2])
        log_f = math.log(self.scale_range) * torch.tanh(raw[..., 2:4])
        h = self.extent * S * log_f.exp()
        theta = self.max_rotation * torch.tanh(raw[..., 4])
        return mu, h, theta, log_f

    def sample_canvases(self, x, mu, h, theta):
        # each token's canvas from the image: bilinear samples at its frame's grid [B, K, C * canvas^2]
        B, K = mu.shape[:2]
        n = self.canvas
        uv = self.canvas_uv # [n, n, 2]
        c, s = torch.cos(theta)[..., None, None], torch.sin(theta)[..., None, None] # [B, K, 1, 1]
        lu, lv = uv[..., 0] * h[..., 0, None, None], uv[..., 1] * h[..., 1, None, None] # frame -> scaled [B, K, n, n]
        px = mu[..., 0, None, None] + c * lu - s * lv
        py = mu[..., 1, None, None] + s * lu + c * lv
        grid = torch.stack([px / self.input_size * 2 - 1, py / self.input_size * 2 - 1], -1).reshape(B, K * n, n, 2)
        out = F.grid_sample(x.float(), grid, mode='bilinear', padding_mode='reflection', align_corners=False) # [B, C, K*n, n]
        return out.view(B, -1, K, n, n).permute(0, 2, 1, 3, 4).reshape(B, K, -1)

    def render(self, patches, mu, h, theta, eps=1e-3):
        """
        Pixels from the tokens' canvases [B, K, C, n, n]: every pixel in each token's window gets the token's canvas sampled
        at its frame coordinates, weighted by the Hann window of those coordinates (plus eps inside the token's own anchor
        cell, so every pixel is covered), summed over tokens and normalized.
        """
        B, K, C, n, _ = patches.shape
        H = self.input_size
        W = self.window
        # the window follows the token, by whole pixels (the values use the exact frame): pixel centres [B, K, W, W, 2] (x, y)
        p = torch.floor(mu)[:, :, None, None, :] + self.window_offsets + 0.5
        d = p - mu[:, :, None, None, :]
        c, s = torch.cos(theta)[..., None, None], torch.sin(theta)[..., None, None]
        u = (c * d[..., 0] + s * d[..., 1]) / h[..., 0, None, None]
        v = (-s * d[..., 0] + c * d[..., 1]) / h[..., 1, None, None]
        hann = lambda z: torch.where(z.abs() < 1, torch.cos(math.pi * z / 2) ** 2, torch.zeros_like(z))
        w = hann(u) * hann(v)
        S = self.patch_size
        own = ((p - (self.anchors[None, :, None, None, :] - S / 2)).div(S).floor() == 0).all(-1) # pixel in the token's anchor cell
        w = w + eps * own.float()
        inside = (p >= 0).all(-1) & (p < H).all(-1)
        w = w * inside.float()
        vals = F.grid_sample(patches.reshape(B * K, C, n, n), torch.stack([u, v], -1).reshape(B * K, W, W, 2),
                             mode='bilinear', padding_mode='border', align_corners=False).view(B, K, C, W, W)
        idx = (p[..., 1].clamp(0, H - 1).long() * H + p[..., 0].clamp(0, H - 1).long()).view(B, 1, K * W * W)
        num = torch.zeros(B, C, H * H, device=patches.device).scatter_add_(2, idx.expand(B, C, -1),
                                                                            (vals * w[:, :, None]).permute(0, 2, 1, 3, 4).reshape(B, C, -1))
        den = torch.zeros(B, 1, H * H, device=patches.device).scatter_add_(2, idx, w.reshape(B, 1, -1))
        return (num / den.clamp_min(1e-12)).view(B, C, H, H)

    # --- forward

    def forward(self, x, t, y):
        t_emb = self.t_embedder(t)
        c = t_emb + self.y_embedder(y)
        c_blocks = self.t_block(c) if self.adaln_single else c
        with torch.autocast(device_type=x.device.type, enabled=False): # geometry, sampling and rendering in fp32
            mu, h, theta, log_f = self.geometry(x, t_emb.float())
            patches = self.sample_canvases(x, mu, h, theta)
        self.last_geometry = (mu.detach(), h.detach(), theta.detach())
        tok = self.x_embedder(patches)
        # positions at the continuous centres (anchor units), detached: geometry learns through sampling and blending
        S = self.patch_size
        cy, cx = (mu[..., 1] / S).detach().flatten(), (mu[..., 0] / S).detach().flatten()
        width = (2 * h.detach().mean(-1) / S).flatten()
        B, K = mu.shape[:2]
        add = integrated_sincos(self.hidden_size, cy, cx, width).view(B, K, -1)
        if self.shape_mlp is not None:
            shape = torch.cat([log_f, torch.sin(2 * theta)[..., None], torch.cos(2 * theta)[..., None] - 1], -1).detach()
            add = add + self.shape_mlp(shape)
        tok = tok + add.to(tok.dtype)
        cos, sin = rope_tables(self.hidden_size // self.num_heads, cy, cx, width)
        bias = (log_f.sum(-1).detach()[:, None, None, :]) if self.area_bias else None # log(frame area / initial area)
        rope = TokenRope(cos.view(B, 1, K, -1), sin.view(B, 1, K, -1), bias)
        res = AttnResState(self, tok, c) if self.attn_res != 'none' else None
        hid = tok
        for i, block in enumerate(self.blocks):
            hid = block(hid, c_blocks, rope, res=res, index=i)
        out = self.final_layer(hid, c) # [B, K, canvas^2 * C] (canvas rows, columns, channels: JiT's patch layout)
        n = self.canvas
        with torch.autocast(device_type=x.device.type, enabled=False):
            patches_out = out.float().view(B, K, n, n, self.out_channels).permute(0, 1, 4, 2, 3)
            return self.render(patches_out, mu, h, theta)

    def dense_weights(self, mu, h, theta):
        # every token's blending weight (Hann window of its frame coordinates) at every pixel [B, K, H, W] (visualization)
        H = self.input_size
        r = torch.arange(H, device=mu.device, dtype=torch.float32) + 0.5
        py, px = torch.meshgrid(r, r, indexing='ij')
        dx, dy = px - mu[..., 0, None, None], py - mu[..., 1, None, None]
        c, s = torch.cos(theta)[..., None, None], torch.sin(theta)[..., None, None]
        u = (c * dx + s * dy) / h[..., 0, None, None]
        v = (-s * dx + c * dy) / h[..., 1, None, None]
        hann = lambda z: torch.where(z.abs() < 1, torch.cos(math.pi * z / 2) ** 2, torch.zeros_like(z))
        return hann(u) * hann(v)

    def geometry_stats(self):
        # summary of the last forward's geometry: mean |offset| and scale spread in anchor units, mean |rotation| (degrees)
        if self.last_geometry is None:
            return {}
        mu, h, theta = self.last_geometry
        S = self.patch_size
        off = ((mu - self.anchors) / S).norm(dim=-1)
        f = h / (self.extent * S)
        return {'offset': off.mean().item(), 'scale_min': f.min().item(), 'scale_max': f.max().item(),
                'log_scale_std': f.log().std().item(), 'rotation_deg': theta.abs().mean().item() * 180 / math.pi}


def draw_geometry(images, mu, h, theta, extent=0.75, color='#a8336b', points=96, core=True):
    """
    images [N, 3, H, W] in [-1, 1] with every token's outline drawn (1px rotated rectangles): its core (core=True: the frame
    scaled by 0.5 / extent, the part a token owns when its neighbours tile; at init these tile the image exactly like JiT's
    patches) or its full frame (core=False: the whole region it samples and blends over, overlapping its neighbours).
    """
    out = images.clone()
    N, _, H, W = out.shape
    rgb = torch.tensor([int(color[i:i + 2], 16) / 127.5 - 1 for i in (1, 3, 5)], device=out.device, dtype=out.dtype)
    t = torch.linspace(0, 4, points + 1, device=out.device)[:-1]
    side, frac = t.floor(), t - t.floor()
    u = torch.where(side == 0, -1 + 2 * frac, torch.where(side == 1, torch.ones_like(t), torch.where(side == 2, 1 - 2 * frac, -torch.ones_like(t))))
    v = torch.where(side == 0, -torch.ones_like(t), torch.where(side == 1, -1 + 2 * frac, torch.where(side == 2, torch.ones_like(t), 1 - 2 * frac)))
    core = h[:N].to(out.device) * ((0.5 / extent) if core else 1.0)
    c, s = torch.cos(theta[:N].to(out.device))[..., None], torch.sin(theta[:N].to(out.device))[..., None]
    lu, lv = u * core[..., 0, None], v * core[..., 1, None]
    px = (mu[:N, :, 0, None].to(out.device) + c * lu - s * lv).round().long().clamp(0, W - 1)
    py = (mu[:N, :, 1, None].to(out.device) + s * lu + c * lv).round().long().clamp(0, H - 1)
    for i in range(N):
        out[i, :, py[i].flatten(), px[i].flatten()] = rgb[:, None]
    return out


# presets: JiT-B's backbone with Gaussian-placed tokens; the anchor spacing names the token count (/16: 64 tokens at 128px)
# bottleneck_dim keeps JiT's compression ratio (768 -> 128 for a 16px patch): the default canvas (2 x 0.75 x patch_size per
# side) holds (1.5)^2 = 2.25x a patch's pixels, so 128 x 2.25 = 288
GAUSS_PRESETS = {
    'GaussJiT-B/16': dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=288, patch_size=16),
    'GaussJiT-B/8': dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=288, patch_size=8),
}


def coverage_panels(net, x, mu, h, theta, n=8, threshold=0.05):
    """
    Per sample: the sample, core outlines, full frame outlines, the owner map (each pixel coloured by the token with the
    largest blending weight, token boundaries darkened: the effective tokenization) and the overlap map (how many tokens
    carry at least `threshold` of the pixel's normalized weight; dark = 1, bright = 4+). Returns images [5 n, 3, H, W] in
    [-1, 1] and labels, for save_labeled_grid with nrow=5.
    """
    import matplotlib
    n = min(n, x.shape[0])
    x, mu, h, theta = x[:n], mu[:n].to(x.device), h[:n].to(x.device), theta[:n].to(x.device)
    w = net.dense_weights(mu, h, theta) # [n, K, H, W]
    wn = w / w.sum(1, keepdim=True).clamp_min(1e-12)
    owner = wn.argmax(1) # [n, H, W]
    K = w.shape[1]
    g = torch.Generator().manual_seed(0)
    palette = (torch.rand(K, 3, generator=g) * 1.4 - 0.7).to(x.device) # distinct mid-range colours per token
    own = palette[owner].permute(0, 3, 1, 2) # [n, 3, H, W]
    edge = torch.zeros_like(owner, dtype=torch.bool)
    edge[:, :, :-1] |= owner[:, :, :-1] != owner[:, :, 1:]
    edge[:, :-1] |= owner[:, :-1] != owner[:, 1:]
    own = torch.where(edge[:, None], torch.full_like(own, -1.0), own)
    count = (wn >= threshold).sum(1).float() # [n, H, W]
    cmap = matplotlib.colormaps['Blues']
    lut = torch.tensor([cmap(0.25 + 0.75 * min(k, 4) / 4)[:3] for k in range(K + 1)], device=x.device) * 2 - 1
    cov = lut[count.long().clamp(0, K)].permute(0, 3, 1, 2)
    cores = draw_geometry(x, mu, h, theta, extent=net.extent)
    frames = draw_geometry(x, mu, h, theta, extent=net.extent, core=False)
    cells, labels = [], []
    for i in range(n):
        cells += [x[i:i + 1], cores[i:i + 1], frames[i:i + 1], own[i:i + 1], cov[i:i + 1]]
        labels += ['sample', 'cores', 'full frames', 'owner', f'overlap (mean {count[i].mean():.1f})']
    return torch.cat(cells), labels
