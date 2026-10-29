"""
Gaussian-placed tokens for JiT (deformable patch embedding): each of the K tokens is an oriented, scaled patch whose
centre, size and orientation a small placer network predicts from the noisy input in the same forward pass, trained only
by the denoising loss. A separate model (as model_quadtree.py), so nothing here can change the baseline: JiT's blocks,
Attention Residuals, adaLN-single, the bottleneck patch embedding and the final layer are reused unchanged in form.

Geometry. Anchors on a grid with spacing S = patch_size (K = (input_size / S)^2: 64 at 128px / 16px). Token k's frame:
  centre      mu_k = anchor_k + max_offset * S * tanh(o)                         (per axis)
  half-size   h_k  = extent * S * f, log f = tanh(s) * (log scale_max if s > 0 else log scale_range)
                                                                                 (per axis; extent 0.75: 1.5 S at init; f in
                                                                                 [1 / scale_range, scale_max], 1 at init)
  rotation    theta_k = max_rotation * tanh(r)                                    (default 0: axis-aligned frames, see below)
with (o, s, r) the placer's 5 outputs per anchor, zero at init (the last layer is zero-initialized), so training starts from
a regular grid of overlapping square frames. Bounded parameters keep tokens from collapsing or wandering off without any
auxiliary loss.

Tokens are continuous functions over their frames, so no pixel is ever resampled (no bilinear interpolation: the noise
statistics a token sees don't depend on its sub-pixel position). With frame coordinates (u, v) in [-1, 1] and the cosine
basis psi_a(z) = s_a cos(pi a (z + 1) / 2) (s_0 = 1, s_a = sqrt 2; at a frame aligned with n x n pixels this is the
orthonormal DCT-II):
Encoder: a token's n x n x 3 coefficients are the projection of the actual pixels inside its frame onto the basis,
  C = sum_p psi(v_p) psi(u_p) x_p / sqrt(N), N = the number of those pixels (exact sums over real pixels; white noise
  gives unit-variance coefficients at any frame size), then JiT's bottleneck embedding (linear -> bottleneck_dim -> hidden).
  At the initial aligned frame with n = its width in pixels, this is the orthonormal DCT of the pixels: lossless.
Decoder: each token's final layer predicts n x n x 3 coefficients; its value at a pixel is the basis evaluated exactly at the
  pixel's frame coordinates, sum C psi(v) psi(u) / n (the inverse DCT at the aligned frame). Pixels blend the tokens near
  them with normalized Gaussian weights of the frame coordinates, exp(-(u^2 + v^2) / (2 sigma^2)) / sum over tokens (a
  local softmax, as Slot Attention's partition of the image), truncated only where the weights are negligible: each token
  renders within its Gaussian's bounding box out to u^2 + v^2 = u_cut^2 (weights 1e-3), at least out to the pixels it may
  own. Every pixel lies within its own anchor's token's box and Gaussians are positive, so every pixel is covered whatever
  the geometry.
Both are computed over a fixed window of pixels around each token, in chunks of tokens, and recomputed in the backward pass
(activation checkpointing), so the pixel x basis intermediates are never stored. Axis-aligned frames (max_rotation = 0, the
default: tokens still shift and scale each axis independently) make the basis separable, psi(v_y) psi(u_x), so encoding and
decoding are two small matmuls per token (as a 2D DCT) instead of a basis evaluation at every pixel of the window. On CUDA,
the fused Triton kernels in gauss_kernels.py do both layers per token in registers, over each token's own bounding box
(separable kernels for axis-aligned frames), storing nothing per pixel.
Positions: the footprint-integrated sin-cos and RoPE at the continuous centres (in anchor units, detached: geometry is
learned through the sampling and the blending), a learned shape embedding of (log scales, rotation), and the area
attention bias (log of the frame's area relative to the anchor cell).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import gauss_kernels
from model_jit import TimestepEmbedder, LabelEmbedder, FinalLayer, AttnResState, make_norm
from model_quadtree import QuadBlock, TokenRope


# positions at continuous centres, recomputed every forward (the tokens move), on the inputs' device: model_quadtree's
# integrated_sincos / rope_tables, which build their tables once at construction on the CPU
def integrated_sincos(dim, cy, cx, hx, hy, theta):
    # each axis's sin-cos averaged over the token's frame (half-sizes hx, hy in anchor units, rotated by theta): the frame's
    # Fourier transform at that axis's frequency, sinc(w hx |cos|) sinc(w hy |sin|) along x and sinc(w hx |sin|) sinc(w hy |cos|)
    # along y (sinc(z) = sin z / z), so an elongated or rotated token is smoothed along its actual extent on each axis
    omega = 1.0 / 10000 ** (torch.arange(dim // 4, device=cy.device, dtype=torch.float32) / (dim // 4))
    c, s = torch.cos(theta.float())[:, None], torch.sin(theta.float())[:, None]
    hx, hy = hx.float()[:, None], hy.float()[:, None]
    sinc = lambda z: torch.sinc(z / math.pi) # torch.sinc(z) = sin(pi z) / (pi z)
    damp_x = sinc(omega * hx * c) * sinc(omega * hy * s)
    damp_y = sinc(omega * hx * s) * sinc(omega * hy * c)
    out = []
    for ctr, damp in ((cy, damp_y), (cx, damp_x)):
        ang = ctr.float()[:, None] * omega[None]
        out += [torch.sin(ang) * damp, torch.cos(ang) * damp]
    return torch.cat(out, dim=1)

def rope_tables(head_dim, cy, cx, theta=10000):
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, 2, device=cy.device)[:half // 2].float() / half))
    cos, sin = [], []
    for c in (cy, cx):
        ang = c.float()[:, None] * freqs[None]
        cos.append(torch.cos(ang).repeat_interleave(2, dim=1))
        sin.append(torch.sin(ang).repeat_interleave(2, dim=1))
    return torch.cat(cos, dim=1), torch.cat(sin, dim=1)


class CoefEmbed(nn.Module):
    # JiT's bottleneck patch embedding on a token's basis coefficients: linear (C * n^2 -> bottleneck), linear (-> hidden)
    def __init__(self, n, in_chans, pca_dim, embed_dim):
        super().__init__()
        self.proj1 = nn.Linear(in_chans * n * n, pca_dim, bias=False)
        self.proj2 = nn.Linear(pca_dim, embed_dim, bias=True)

    def forward(self, coef): # [B, K, C * n^2]
        return self.proj2(self.proj1(coef))


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


class ConvNeXtPlacer(nn.Module):
    """
    A ConvNeXt-style placer (Liu et al. 2022), t-conditioned: an overlapping 2s x 2s stride-s stem conv (s = stride, 4 or 8),
    blocks of [7x7 depthwise conv, LayerNorm with FiLM from t (and class), pointwise MLP 4x, layer scale, residual] at stride
    s, then a linear layer over each anchor cell's (patch_size / s)^2 features (a non-overlapping strided conv: keeps where
    things are inside the cell, unlike average pooling) to twice the width, more blocks on the anchor grid (each 7x7
    depthwise conv reaches 3 anchors away, so tokens see their neighbours' content), and a zero-initialized linear head to
    5 numbers per anchor. Receptive field at 128px with the defaults (2 + 2 blocks): the whole image. The stride-s blocks are
    most of the cost (memory-bound at large batch): stride 8 runs them on 4x fewer positions, with twice the reach per block.
    """
    def __init__(self, in_chans, patch_size, width, depth, grid_depth, cond_dim, stride=4):
        super().__init__()
        s = stride
        assert patch_size % s == 0 and s % 2 == 0, f'placer stride {s} must be even and divide the patch size {patch_size}'
        self.stem = nn.Conv2d(in_chans, width, 2 * s, stride=s, padding=s // 2) # centred on each s x s cell
        self.stem_norm = nn.LayerNorm(width, eps=1e-6)
        self.blocks = nn.ModuleList([PlacerBlock(width) for _ in range(depth)])
        self.down_norm = nn.LayerNorm(width, eps=1e-6)
        self.down = nn.Conv2d(width, 2 * width, patch_size // s, stride=patch_size // s) # linear over each anchor cell
        self.grid_blocks = nn.ModuleList([PlacerBlock(2 * width) for _ in range(grid_depth)])
        self.films = nn.ModuleList([nn.Linear(cond_dim, 2 * b.width) for b in [*self.blocks, *self.grid_blocks]])
        self.head_norm = nn.LayerNorm(2 * width, eps=1e-6)
        self.head = nn.Linear(2 * width, 5)

    def forward(self, x, c):
        c = F.silu(c)
        films = iter(self.films)
        h = self.stem_norm(self.stem(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for block in self.blocks:
            h = block(h, next(films)(c))
        h = self.down(self.down_norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        for block in self.grid_blocks:
            h = block(h, next(films)(c))
        return self.head(self.head_norm(h.flatten(2).transpose(1, 2))) # [B, K, 5]


class PlacerBlock(nn.Module):
    # ConvNeXt block with FiLM (zero-initialized: a plain affine-free LayerNorm at init) on its LayerNorm
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.dw = nn.Conv2d(width, width, 7, padding=3, groups=width)
        self.norm = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.pw1 = nn.Linear(width, 4 * width)
        self.pw2 = nn.Linear(4 * width, width)
        self.gamma = nn.Parameter(torch.full((width,), 1e-6)) # layer scale

    def forward(self, x, film):
        g, b = film[:, None, None, :].chunk(2, dim=-1)
        h = self.norm(self.dw(x).permute(0, 2, 3, 1)) * (1 + g) + b
        h = self.pw2(F.gelu(self.pw1(h)))
        return x + (self.gamma * h).permute(0, 3, 1, 2)


class GaussJiT(nn.Module):
    family = 'gauss'

    def __init__(self, input_size=128, patch_size=16, in_channels=3, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0, num_classes=1000, bottleneck_dim=128, attn_res='none', attn_res_block_size=0,
                 adaln_single=False, layernorm=False, extent=0.75, basis=0, max_offset=0.25, scale_range=1.33, scale_max=0.0,
                 max_rotation=0.0, sigma=0.3, placer='conv', placer_width=64, placer_depth=2, placer_grid_depth=2, placer_stride=4, scale_jitter=0.0, encoder_window='box', encoder_sigma=0.0, decoder_extent=0.0, decoder_basis=0, area_bias=True, shape_embed=True,
                 chunks=8, fused='auto', in_context_len=0, in_context_start=0, family='gauss'):
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
        self.scale_max = scale_max or scale_range # upper scale bound (the lower one stays 1 / scale_range)
        self.basis_n = basis or int(round(2 * extent * patch_size)) # basis size n per axis: the initial frame's width in pixels
        # the decoder's frames: the encoder's scaled by decoder_extent / extent (same centres, shapes and rotations), with
        # their own basis size (default: the decoder frame's width in pixels). The defaults reuse the encoder's
        self.dec_extent = decoder_extent or extent
        self.dec_ratio = self.dec_extent / extent
        self.dec_n = decoder_basis or (int(round(2 * self.dec_extent * patch_size)) if decoder_extent else self.basis_n)
        self.grid = input_size // patch_size
        self.tokens = self.grid ** 2
        self.area_bias, self.sigma, self.chunks = area_bias, sigma, chunks
        # fully fused Triton kernels (gauss_kernels.encode_op / render_op): 'auto' / 'always' wherever available (CUDA; separable
        # kernels for axis-aligned frames), 'never' (the chunked PyTorch path everywhere)
        assert fused in ('auto', 'always', 'never'), f"fused must be 'auto', 'always' or 'never', got {fused}"
        self.fused = fused
        # pixel windows around each token (whole pixels around floor(mu)): the encoder's covers the largest rotated frame;
        # the decoder's reaches where the Gaussian weight falls to 1e-3 of its peak, and always contains the pixel's own
        # anchor cell (so every pixel has positive weight)
        h_max = extent * patch_size * self.scale_max
        rot = min(max_rotation, math.pi / 4)
        self.window_enc = 2 * math.ceil(h_max * (math.cos(rot) + math.sin(rot))) + 3
        u_cut = gauss_kernels.render_cut(sigma)
        own = max_offset * patch_size * math.sqrt(2) + patch_size / math.sqrt(2) + 1
        self.window = 2 * math.ceil(max(u_cut * h_max * self.dec_ratio, own)) + 3
        self.r_own = math.ceil(own) + 1
        # training only: Gaussian noise (std scale_jitter) on each token's per-axis log-scale, within the bounds. Smooths the
        # loss over nearby scales (randomized smoothing), so the placer gets a gradient across plateaus: a frame whose
        # content is finer than its basis can represent has a loss flat in its scale, and only nearby scales see the way out
        self.scale_jitter = scale_jitter
        # the encoder's window over a frame: 'box' (every pixel in the frame weighs the same) or 'gauss' (exp(-(u^2 + v^2) /
        # 2 sigma_e^2), sigma_e = encoder_sigma or sigma, cut at the frame edge): smooth in the geometry, so the encoder too
        # tells the placer which pixels to take in more or less. Coefficients are normalized by sqrt(sum of squared weights)
        assert encoder_window in ('box', 'gauss'), f"encoder_window must be 'box' or 'gauss', got {encoder_window}"
        self.encoder_window = encoder_window
        sigma_e = encoder_sigma or sigma
        self.enc_inv2s2 = 1.0 / (2 * sigma_e ** 2) if encoder_window == 'gauss' else 0.0 # each token renders within its Gaussian's box (u^2 + v^2 < u_cut^2), at least this far

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        if adaln_single:
            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        # placer: 'conv' (the original: stride-4 conv, 3x3 blocks, average pooled to the anchors; ~32px receptive field per
        # token) or 'convnext' (ConvNeXtPlacer: whole-image receptive field, position-aware readout of each anchor cell)
        # 'none': no placer, every token keeps its anchor-grid frame (for ablations: the tokenization without adaptation)
        assert placer in ('conv', 'convnext', 'none'), f"placer must be 'conv', 'convnext' or 'none', got {placer}"
        if placer == 'none':
            self.placer = None
        elif placer == 'convnext':
            self.placer = ConvNeXtPlacer(in_channels, patch_size, placer_width, placer_depth, placer_grid_depth, hidden_size, placer_stride)
        else:
            assert placer_stride == 4, "placer_stride is a 'convnext' option (the 'conv' placer works at patch_size / 4)"
            self.placer = Placer(in_channels, patch_size, placer_width, placer_depth, hidden_size)
        self.x_embedder = CoefEmbed(self.basis_n, in_channels, bottleneck_dim, hidden_size)
        self.final_layer = FinalLayer(hidden_size, self.dec_n, in_channels, adaln_single=adaln_single, layernorm=layernorm)
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
        n = self.basis_n
        self.register_buffer('freqs', torch.arange(n, dtype=torch.float32), persistent=False)
        self.register_buffer('basis_scale', torch.cat([torch.ones(1), torch.full((n - 1,), math.sqrt(2))]), persistent=False)
        nd = self.dec_n
        self.register_buffer('dec_freqs', torch.arange(nd, dtype=torch.float32), persistent=False)
        self.register_buffer('dec_basis_scale', torch.cat([torch.ones(1), torch.full((nd - 1,), math.sqrt(2))]), persistent=False)
        for name, W in (('offsets_enc', self.window_enc), ('offsets_dec', self.window)):
            w = torch.arange(W, dtype=torch.float32) - (W - 1) // 2
            wy, wx = torch.meshgrid(w, w, indexing='ij')
            self.register_buffer(name, torch.stack([wx, wy], -1).view(-1, 2), persistent=False) # [W * W, 2] (x, y)
            self.register_buffer(name + '_1d', w, persistent=False) # [W]
        self.separable = max_rotation == 0 # axis-aligned frames: the 2D basis is an outer product of 1D tables
        self.last_geometry = None # the last forward's (mu, h, theta), detached: for visualizations and logging
        self.geo_record = None # while recording (record_geometry): {t: lists of per-call samples}, e.g. over an FID generation
        self.initialize_weights()

    def __setattr__(self, name, value):
        if name in ('last_geometry', '_h_placed'): # plain attributes, never part of the module state
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
        if self.placer is not None:
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
        raw = self.placer(x, c).float() if self.placer is not None else x.new_zeros(x.shape[0], self.tokens, 5, dtype=torch.float32)
        S = self.patch_size
        mu = self.anchors + self.max_offset * S * torch.tanh(raw[..., 0:2])
        ts = torch.tanh(raw[..., 2:4]) # growing and shrinking bounded separately; scale 1 (the anchor grid's frames) at init
        log_f = ts * torch.where(ts > 0, math.log(self.scale_max), math.log(self.scale_range))
        self._h_placed = (self.extent * S * log_f.exp()).detach() # the placer's own scales, before any jitter (for logging)
        if self.training and self.scale_jitter > 0:
            log_f = (log_f + self.scale_jitter * torch.randn_like(log_f)).clamp(-math.log(self.scale_range), math.log(self.scale_max))
        h = self.extent * S * log_f.exp()
        theta = self.max_rotation * torch.tanh(raw[..., 4])
        return mu, h, theta, log_f

    def basis(self, z):
        # the cosine basis at frame coordinates z [...] -> [..., n]
        return torch.cos(math.pi * self.freqs * (z[..., None] + 1) / 2) * self.basis_scale

    def dec_basis(self, z):
        # the decoder's cosine basis (dec_n functions) at decoder-frame coordinates z [...] -> [..., dec_n]
        return torch.cos(math.pi * self.dec_freqs * (z[..., None] + 1) / 2) * self.dec_basis_scale

    def window_coords(self, mu, h, theta, offsets, reflect=False):
        # pixel centres in a window around each token [B, k, P, 2], their flat indices [B, k, P], whether they're in the
        # image, and their frame coordinates u, v [B, k, P]. reflect: pixels outside the image index their mirror image
        # inside it (the encoder reads a full frame at the borders; the cosine basis is itself an even extension)
        H = self.input_size
        p = torch.floor(mu)[:, :, None, :] + offsets + 0.5
        inside = (p >= 0).all(-1) & (p < H).all(-1)
        q = p.floor().long()
        if reflect:
            q = torch.where(q < 0, -q - 1, q)
            q = torch.where(q >= H, 2 * H - 1 - q, q).clamp(0, H - 1)
        else:
            q = q.clamp(0, H - 1)
        idx = q[..., 1] * H + q[..., 0]
        d = p - mu[:, :, None, :]
        c, s = torch.cos(theta)[..., None], torch.sin(theta)[..., None]
        u = (c * d[..., 0] + s * d[..., 1]) / h[..., 0, None]
        v = (-s * d[..., 0] + c * d[..., 1]) / h[..., 1, None]
        return idx, inside, u, v

    def enc_weight(self, z):
        # the encoder's window along one frame axis (1 inside the frame for the box); the 2D window is w(u) w(v)
        return torch.where(z.abs() < 1, torch.exp(-z * z * self.enc_inv2s2), torch.zeros_like(z))

    def _encode_chunk(self, x_flat, mu, h, theta):
        # coefficients [B, k, C, n, n] (rows: v, columns: u) of the pixels inside each token's frame, weighted by the window
        B, C = x_flat.shape[:2]
        idx, _, u, v = self.window_coords(mu, h, theta, self.offsets_enc, reflect=True)
        wu, wv = self.enc_weight(u), self.enc_weight(v) # [B, k, P]: the frame's pixels (reflected past the borders)
        k, P = idx.shape[1], idx.shape[2]
        vals = x_flat.gather(2, idx.view(B, 1, k * P).expand(B, C, -1)).view(B, C, k, P).transpose(1, 2) # [B, k, C, P]
        pu, pv = self.basis(u) * wu[..., None], self.basis(v) * wv[..., None] # [B, k, P, n]
        coef = torch.einsum('bkcp,bkpq,bkpa->bkcqa', vals, pv, pu)
        return coef / (wu * wv).square().sum(-1).clamp_min(1e-6).sqrt()[..., None, None, None]

    def render_radius(self, h, theta):
        # per token, the decoder's box radii around floor(mu) [B, k] (x, y): gauss_kernels.render_radius
        geom = torch.cat([h.new_zeros(h.shape), h, theta[..., None]], -1).reshape(-1, 5)
        rx, ry = gauss_kernels.render_radius(geom, self.sigma, self.r_own, (self.window - 1) // 2)
        return rx.view(h.shape[:-1]), ry.view(h.shape[:-1])

    def _decode_chunk(self, coef, mu, h, theta):
        # each token's values and Gaussian blending weights at the pixels of its window: (numerator [B, C, H*H] contribution,
        # weight [B, 1, H*H] contribution)
        B, k, C, n, _ = coef.shape
        H = self.input_size
        idx, inside, u, v = self.window_coords(mu, h, theta, self.offsets_dec)
        rx, ry = self.render_radius(h, theta)
        inside = inside & (self.offsets_dec.abs() <= torch.stack([rx, ry], -1)[:, :, None]).all(-1)
        w = torch.exp(-(u ** 2 + v ** 2) / (2 * self.sigma ** 2)) * inside.float() # [B, k, P]
        vals = torch.einsum('bkcqa,bkpq,bkpa->bkcp', coef, self.dec_basis(v), self.dec_basis(u)) / n # [B, k, C, P]
        P = idx.shape[2]
        num = torch.zeros(B, C, H * H, device=coef.device).scatter_add_(
            2, idx.view(B, 1, k * P).expand(B, C, -1), (vals * w[:, :, None]).transpose(1, 2).reshape(B, C, k * P))
        den = torch.zeros(B, 1, H * H, device=coef.device).scatter_add_(2, idx.view(B, 1, k * P), w.reshape(B, 1, k * P))
        return num, den

    def axis_coords(self, mu, h, offsets_1d, reflect=False):
        # axis-aligned frames: per axis, the window's pixel indices [B, k, W] and frame coordinates [B, k, W], for x and y
        H = self.input_size
        p = torch.floor(mu)[..., None] + offsets_1d + 0.5 # [B, k, 2, W] pixel centres per axis
        z = (p - mu[..., None]) / h[..., None]
        q = p.floor().long()
        inside = (q >= 0) & (q < H)
        if reflect:
            q = torch.where(q < 0, -q - 1, q)
            q = torch.where(q >= H, 2 * H - 1 - q, q)
        return q.clamp(0, H - 1), inside, z

    def _block_index(self, q):
        # flat pixel indices of each token's W x W window [B, k, W * W] (rows: y, columns: x) from per-axis indices [B, k, 2, W]
        H = self.input_size
        return (q[:, :, 1, :, None] * H + q[:, :, 0, None, :]).flatten(2)

    def _encode_chunk_sep(self, x_flat, mu, h):
        B, C = x_flat.shape[:2]
        q, _, z = self.axis_coords(mu, h, self.offsets_enc_1d, reflect=True)
        m = self.enc_weight(z) # [B, k, 2, W]: the window per axis (the frame mask for the box)
        k, W = q.shape[1], q.shape[-1]
        idx = self._block_index(q)
        X = x_flat.gather(2, idx.view(B, 1, -1).expand(B, C, -1)).view(B, C, k, W, W).transpose(1, 2) # [B, k, C, Wy, Wx]
        pu = self.basis(z[:, :, 0]) * m[:, :, 0, :, None] # [B, k, Wx, n]
        pv = self.basis(z[:, :, 1]) * m[:, :, 1, :, None] # [B, k, Wy, n]
        coef = pv.transpose(-1, -2)[:, :, None] @ X @ pu[:, :, None] # [B, k, C, n, n] (rows v, columns u)
        S = m[:, :, 0].square().sum(-1) * m[:, :, 1].square().sum(-1)
        return coef / S.clamp_min(1e-6).sqrt()[..., None, None, None]

    def _decode_chunk_sep(self, coef, mu, h):
        B, k, C, n, _ = coef.shape
        H = self.input_size
        q, inside, z = self.axis_coords(mu, h, self.offsets_dec_1d)
        rx, ry = self.render_radius(h, torch.zeros_like(h[..., 0]))
        inside = inside & (self.offsets_dec_1d.abs() <= torch.stack([rx, ry], -1)[..., None])
        W = q.shape[-1]
        g = torch.exp(-z ** 2 / (2 * self.sigma ** 2)) * inside.float() # [B, k, 2, W]: the Gaussian factors per axis
        vals = self.dec_basis(z[:, :, 1])[:, :, None] @ coef @ self.dec_basis(z[:, :, 0]).transpose(-1, -2)[:, :, None] / n # [B, k, C, Wy, Wx]
        w = g[:, :, 1, :, None] * g[:, :, 0, None, :] # [B, k, Wy, Wx]
        idx = self._block_index(q).view(B, 1, -1)
        num = torch.zeros(B, C, H * H, device=coef.device).scatter_add_(
            2, idx.expand(B, C, -1), (vals * w[:, :, None]).transpose(1, 2).reshape(B, C, -1))
        den = torch.zeros(B, 1, H * H, device=coef.device).scatter_add_(2, idx, w.reshape(B, 1, -1))
        return num, den

    def _chunked(self, fn, *args):
        # fn over chunks of tokens (args: per-token tensors [B, K, ...] after the first, shared, argument), recomputed in the
        # backward pass when training, so the pixel x basis intermediates are never stored
        shared, per_token = args[0], args[1:]
        K = per_token[0].shape[1]
        size = math.ceil(K / max(1, self.chunks))
        outs = []
        for i in range(0, K, size):
            part = [t[:, i:i + size] for t in per_token]
            if torch.is_grad_enabled():
                outs.append(torch.utils.checkpoint.checkpoint(fn, shared, *part, use_reentrant=False))
            else:
                outs.append(fn(shared, *part))
        return outs

    def _encode_fused(self, x_flat, mu, h, theta):
        # gauss_kernels.encode_op does the window, masks, pixel reads and basis projection per token in one kernel (and the
        # geometry gradients in its backward; separable kernels for axis-aligned frames); only the image and the geometry
        # are kept for backward
        B, C = x_flat.shape[:2]
        K, n, H = mu.shape[1], self.basis_n, self.input_size
        geom = torch.cat([mu, h, theta[..., None]], -1).reshape(B * K, 5)
        coef, _ = gauss_kernels.encode_op(x_flat.view(B, C, H, H), geom, K, n, self.window_enc, self.separable, self.enc_inv2s2)
        return coef.view(B, K, C, n, n)

    def _render_fused(self, coef, mu, h, theta):
        # gauss_kernels.render_op evaluates, weights and accumulates every token's box into the image in one kernel (its
        # backward returns the coefficient and geometry gradients); nothing per pixel is stored
        B, K, C, n, _ = coef.shape
        H = self.input_size
        geom = torch.cat([mu, h, theta[..., None]], -1).reshape(B * K, 5)
        num, den = gauss_kernels.render_op(coef.reshape(B * K, C, n, n).contiguous(), geom, B, H, self.window, float(self.sigma),
                                           self.r_own, self.separable)
        return num / den.clamp_min(1e-30)

    def use_fused(self, device):
        # 'auto' and 'always': the fused kernels wherever they're available (CUDA); 'never': the PyTorch path
        return self.fused != 'never' and gauss_kernels.kernels_available(device)

    def encode(self, x, mu, h, theta):
        # basis coefficients of every token [B, K, C * n^2]
        B, C = x.shape[:2]
        x_flat = x.float().reshape(B, C, -1)
        if self.use_fused(x.device):
            return self._encode_fused(x_flat, mu, h, theta).reshape(B, mu.shape[1], -1)
        if self.separable:
            coef = torch.cat(self._chunked(lambda xf, m_, h_: self._encode_chunk_sep(xf, m_, h_), x_flat, mu, h), dim=1)
        else:
            coef = torch.cat(self._chunked(lambda xf, m_, h_, t_: self._encode_chunk(xf, m_, h_, t_), x_flat, mu, h, theta), dim=1)
        return coef.reshape(B, mu.shape[1], -1)

    def render(self, coef, mu, h, theta):
        # pixels from every token's coefficients [B, K, C, n, n]: Gaussian-weighted, normalized over tokens
        B, K, C = coef.shape[:3]
        H = self.input_size
        if self.use_fused(coef.device):
            return self._render_fused(coef, mu, h, theta)
        if self.separable:
            parts = self._chunked(lambda _, c_, m_, h_: self._decode_chunk_sep(c_, m_, h_), None, coef, mu, h)
        else:
            parts = self._chunked(lambda _, c_, m_, h_, t_: self._decode_chunk(c_, m_, h_, t_), None, coef, mu, h, theta)
        num = sum(p_[0] for p_ in parts)
        den = sum(p_[1] for p_ in parts)
        return (num / den.clamp_min(1e-30)).view(B, C, H, H)

    # --- forward

    def forward(self, x, t, y):
        t_emb = self.t_embedder(t)
        c = t_emb + self.y_embedder(y)
        c_blocks = self.t_block(c) if self.adaln_single else c
        with torch.autocast(device_type=x.device.type, enabled=False): # geometry, sampling and rendering in fp32
            mu, h, theta, log_f = self.geometry(x, t_emb.float())
            patches = self.encode(x, mu, h, theta)
        self.last_geometry = (mu.detach(), self._h_placed, theta.detach()) # logged geometry: the placer's, without training jitter
        if self.geo_record is not None:
            self._record_geometry(t, mu, h, theta)
        tok = self.x_embedder(patches)
        # positions at the continuous centres (anchor units), detached: geometry learns through sampling and blending
        S = self.patch_size
        cy, cx = (mu[..., 1] / S).detach().flatten(), (mu[..., 0] / S).detach().flatten()
        hx, hy = (h[..., 0].detach() / S).flatten(), (h[..., 1].detach() / S).flatten() # frame half-sizes (anchor units)
        B, K = mu.shape[:2]
        add = integrated_sincos(self.hidden_size, cy, cx, hx, hy, theta.detach().flatten()).view(B, K, -1)
        if self.shape_mlp is not None:
            shape = torch.cat([log_f, torch.sin(2 * theta)[..., None], torch.cos(2 * theta)[..., None] - 1], -1).detach()
            add = add + self.shape_mlp(shape)
        tok = tok + add.to(tok.dtype)
        cos, sin = rope_tables(self.hidden_size // self.num_heads, cy, cx)
        bias = (log_f.sum(-1).detach()[:, None, None, :]) if self.area_bias else None # log(frame area / initial area)
        rope = TokenRope(cos.view(B, 1, K, -1), sin.view(B, 1, K, -1), bias)
        res = AttnResState(self, tok, c) if self.attn_res != 'none' else None
        hid = tok
        for i, block in enumerate(self.blocks):
            hid = block(hid, c_blocks, rope, res=res, index=i)
        out = self.final_layer(hid, c) # [B, K, n^2 * C]: coefficients (rows v, columns u, channels), JiT's patch layout
        n = self.dec_n
        with torch.autocast(device_type=x.device.type, enabled=False):
            coef = out.float().view(B, K, n, n, self.out_channels).permute(0, 1, 4, 2, 3)
            return self.render(coef, mu, h * self.dec_ratio, theta) # the decoder's frames

    def dense_weights(self, mu, h, theta):
        # every token's blending weight (the Gaussian of its frame coordinates, untruncated) at every pixel [B, K, H, W]
        H = self.input_size
        r = torch.arange(H, device=mu.device, dtype=torch.float32) + 0.5
        py, px = torch.meshgrid(r, r, indexing='ij')
        dx, dy = px - mu[..., 0, None, None], py - mu[..., 1, None, None]
        c, s = torch.cos(theta)[..., None, None], torch.sin(theta)[..., None, None]
        u = (c * dx + s * dy) / (h[..., 0, None, None] * self.dec_ratio) # h: the encoder's frames; blending uses the decoder's
        v = (-s * dx + c * dy) / (h[..., 1, None, None] * self.dec_ratio)
        return torch.exp(-(u ** 2 + v ** 2) / (2 * self.sigma ** 2))

    def spread_stats(self, h, theta):
        """
        Token-to-token variation vs whole-image shifts, for geometry of N images [N, K, ...]:
          scale_within    mean over images of the std of log-scale across the image's tokens (both axes): tokens differing
                          inside an image (adaptive allocation)
          scale_between   std over images of the image's mean log-scale: whole images tokenized coarser / finer
          aspect_p50/p95  each token's elongation, long axis / short axis (1 = circle), median and 95th percentile
          rotation_mean   signed mean rotation (degrees); rotation_same_sign: the share of rotated tokens (|theta| > 0.5 deg)
                          turning the majority's way (1 = one global direction); rotation_within: mean within-image std (degrees)
        """
        lf = (h / (self.extent * self.patch_size)).float().log().flatten(1) # [N, 2K]
        deg = theta.float() * 180 / math.pi # [N, K]
        rot = deg[deg.abs() > 0.5]
        same = max((rot > 0).float().mean().item(), (rot < 0).float().mean().item()) if rot.numel() else float('nan')
        a50, a95 = torch.quantile(self._aspect(h).flatten()[:2 ** 20], torch.tensor([0.5, 0.95], device=h.device)).tolist()
        return {'scale_within': lf.std(1).mean().item(), 'scale_between': lf.mean(1).std().item() if lf.shape[0] > 1 else 0.0,
                'aspect_p50': a50, 'aspect_p95': a95,
                'rotation_mean': deg.mean().item(), 'rotation_same_sign': same, 'rotation_within': deg.std(1).mean().item()}

    @staticmethod
    def _aspect(h):
        # long axis / short axis per token
        return (h[..., 0].float().log() - h[..., 1].float().log()).abs().exp()

    def record_geometry(self, on=True):
        # start (clearing) or stop recording every forward's geometry by noise level, e.g. around an FID generation
        self.geo_record = {} if on else None

    @torch.compiler.disable
    @torch.no_grad()
    def _record_geometry(self, t, mu, h, theta, n_tokens=32, n_coverage=8):
        # this call's geometry for the first n_tokens images (CFG batches carry the conditional half first) and its
        # outside-frame share over the first n_coverage, under the call's noise level
        key = round(float(t.flatten()[0]), 3)
        S = self.patch_size
        m_, h_, t_ = mu[:n_tokens].float(), h[:n_tokens].float(), theta[:n_tokens].float()
        r = self.geo_record.setdefault(key, {'offset': [], 'scale': [], 'rotation': [], 'outside': [], 'lf_std': [], 'lf_mean': [],
                                             'rot_signed': [], 'rot_std': [], 'aspect': []})
        lf = (h_ / (self.extent * S)).log().flatten(1)
        deg = t_ * 180 / math.pi
        r['lf_std'].append(lf.std(1).cpu()); r['lf_mean'].append(lf.mean(1).cpu())
        r['rot_signed'].append(deg.flatten().cpu()); r['rot_std'].append(deg.std(1).cpu())
        r['offset'].append(((m_ - self.anchors) / S).norm(dim=-1).flatten().cpu())
        r['scale'].append((h_ / (self.extent * S)).flatten().cpu())
        r['aspect'].append(self._aspect(h_).flatten().cpu())
        r['rotation'].append((t_.abs() * 180 / math.pi).flatten().cpu())
        r['outside'].append(self.outside_frame_share(m_[:n_coverage], h_[:n_coverage], t_[:n_coverage]))

    def geometry_record_summary(self):
        # {t: geometry statistics} from the recording, the same keys as jit_common.save_geometry_live's
        out = {}
        for key, r in sorted((self.geo_record or {}).items()):
            off, f, rot = torch.cat(r['offset']), torch.cat(r['scale']), torch.cat(r['rotation'])
            qs = lambda z, q: torch.quantile(z[:2 ** 24], torch.tensor(q)).tolist()
            p5, p50, p95 = qs(f, [0.05, 0.5, 0.95])
            r50, r95 = qs(rot, [0.5, 0.95])
            a50, a95 = qs(torch.cat(r['aspect']), [0.5, 0.95]) if r.get('aspect') else (float('nan'), float('nan'))
            rs = torch.cat(r['rot_signed'])
            rr = rs[rs.abs() > 0.5]
            out[key] = {'offset_mean': off.mean().item(), 'offset_p95': qs(off, [0.95])[0], 'scale_p5': p5, 'scale_p50': p50,
                        'scale_p95': p95, 'aspect_p50': a50, 'aspect_p95': a95, 'rotation_p50': r50, 'rotation_p95': r95, 'outside_frame': sum(r['outside']) / len(r['outside']),
                        'scale_within': torch.cat(r['lf_std']).mean().item(), 'scale_between': torch.cat(r['lf_mean']).std().item(),
                        'rotation_mean': rs.mean().item(), 'rotation_within': torch.cat(r['rot_std']).mean().item(),
                        'rotation_same_sign': max((rr > 0).float().mean().item(), (rr < 0).float().mean().item()) if rr.numel() else float('nan'),
                        'samples': len(r['outside'])}
        return out

    def geometry_stats(self):
        # distribution of the last forward's geometry over tokens: |offset| (anchor spacings), per-axis scale (relative to
        # the initial frame), |rotation| (degrees): mean, percentiles (p5, p50, p95), min, max
        if self.last_geometry is None:
            return {}
        mu, h, theta = self.last_geometry
        S = self.patch_size
        q = torch.tensor([0.05, 0.5, 0.95], device=mu.device)
        out = {}
        for name, z in (('offset', ((mu - self.anchors) / S).norm(dim=-1)), ('scale', h / (self.extent * S)),
                        ('rotation', theta.abs() * 180 / math.pi)):
            z = z.flatten().float()
            p5, p50, p95 = torch.quantile(z[:2 ** 20], q).tolist() # quantile's input limit; a batch has far fewer values
            out.update({f'{name}_mean': z.mean().item(), f'{name}_p5': p5, f'{name}_p50': p50, f'{name}_p95': p95,
                        f'{name}_min': z.min().item(), f'{name}_max': z.max().item()})
        out.update(self.spread_stats(h, theta))
        return out

    def outside_frame_share(self, mu, h, theta, chunk=16):
        # share of pixels whose dominant token (largest blending weight) doesn't contain them in its frame (|u| or |v| >= 1):
        # those pixels take that token's basis extended past its frame (a fallback), not values from pixels it read
        # (evaluated over all pixels and tokens, `chunk` images at a time: [chunk, K, H, W] intermediates)
        H = self.input_size
        r = torch.arange(H, device=mu.device, dtype=torch.float32) + 0.5
        py, px = torch.meshgrid(r, r, indexing='ij')
        n_out = 0.0
        for i in range(0, mu.shape[0], chunk):
            m_, h_, t_ = mu[i:i + chunk], h[i:i + chunk], theta[i:i + chunk]
            dx, dy = px - m_[..., 0, None, None], py - m_[..., 1, None, None]
            c, s = torch.cos(t_)[..., None, None], torch.sin(t_)[..., None, None]
            u = (c * dx + s * dy) / h_[..., 0, None, None]
            v = (-s * dx + c * dy) / h_[..., 1, None, None]
            own = (u ** 2 + v ** 2).argmin(1, keepdim=True) # the largest Gaussian weight = the smallest frame radius
            n_out += ((u.gather(1, own).abs() >= 1) | (v.gather(1, own).abs() >= 1)).float().sum().item()
        return n_out / (mu.shape[0] * H * H)


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
# bottleneck_dim keeps JiT's compression ratio (768 -> 128 for a 16px patch): the default basis (2 x 0.75 x patch_size per
# side, the initial frame's width in pixels) has (1.5)^2 = 2.25x a patch's coefficients, so 128 x 2.25 = 288
GAUSS_PRESETS = {
    'GaussJiT-B/16': dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=288, patch_size=16),
    'GaussJiT-B/8': dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=288, patch_size=8),
}


TOKEN_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a4fc9', '#00a3c4', '#c43c3c']

def token_colors(grid):
    # one colour per anchor from 9, in a 3 x 3 repeating pattern: every token's 8 neighbours have other colours
    import matplotlib.colors as mcolors
    rgb = torch.tensor([mcolors.to_rgb(c) for c in TOKEN_COLORS])
    iy, ix = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
    return rgb[((iy % 3) * 3 + ix % 3).flatten()] # [K, 3] in [0, 1]

def draw_gaussians(ax, net, image, mu, h, theta, arrow_scale=4.0, contour=2.0):
    """
    On a matplotlib axis: the image (dimmed), every token's Gaussian as a filled ellipse at its `contour`-sigma level (the
    blending weight exp(-(u^2 + v^2) / 2 sigma^2) in the token's frame: semi-axes contour * sigma * h, rotated by theta)
    in its own colour, and an arrow from its anchor to its centre (drawn arrow_scale x longer).
    """
    from matplotlib.patches import Ellipse
    import numpy as np
    H = image.shape[-1]
    ax.imshow(0.35 + 0.65 * ((image.float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy())
    cols = token_colors(net.grid).numpy()
    anchors = net.anchors.cpu().numpy()
    mu, h, theta = mu.float().cpu().numpy(), h.float().cpu().numpy(), theta.float().cpu().numpy()
    for k in range(mu.shape[0]):
        e = Ellipse(mu[k] - 0.5, 2 * contour * net.sigma * net.dec_ratio * h[k, 0], 2 * contour * net.sigma * net.dec_ratio * h[k, 1], angle=float(np.degrees(theta[k])),
                    facecolor=(*cols[k], 0.28), edgecolor=cols[k], linewidth=0.8)
        ax.add_patch(e)
        d = (mu[k] - anchors[k]) * arrow_scale
        ax.plot(anchors[k, 0] - 0.5, anchors[k, 1] - 0.5, 'o', color='k', markersize=1.2)
        if np.hypot(*d) > 0.3:
            ax.annotate('', xy=(anchors[k, 0] - 0.5 + d[0], anchors[k, 1] - 0.5 + d[1]), xytext=(anchors[k, 0] - 0.5, anchors[k, 1] - 0.5),
                        arrowprops=dict(arrowstyle='-|>', color='k', lw=0.7, mutation_scale=5))
    ax.set_xlim(-0.5, H - 0.5); ax.set_ylim(H - 0.5, -0.5)

def soft_ownership(net, image, mu, h, theta):
    # [H, W, 3]: each pixel the blend of its tokens' colours by their normalized blending weights, over the dimmed image
    w = net.dense_weights(mu[None].float(), h[None].float(), theta[None].float())[0] # [K, H, W]
    w = w / w.sum(0, keepdim=True).clamp_min(1e-30)
    col = torch.einsum('khw,kc->hwc', w, token_colors(net.grid).to(w.device))
    img = ((image.float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).to(w.device)
    return (0.6 * col + 0.4 * img).cpu().numpy()

def geometry_figure(net, path, images, columns, row_labels=None, ownership=True, title=''):
    """
    A figure of token geometry: one row per image; per column (title, mu [N, K, 2], h [N, K, 2], theta [N, K]) the
    Gaussians with displacement arrows, and (ownership) the last column's soft ownership. images [N, 3, H, W] in [-1, 1].
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    N = images.shape[0]
    ncol = 1 + len(columns) + (1 if ownership else 0)
    fig, axes = plt.subplots(N, ncol, figsize=(2.1 * ncol, 2.15 * N + 0.7), squeeze=False, facecolor='#fcfcfb')
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for i in range(N):
        axes[i, 0].imshow(((images[i].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy())
        if row_labels:
            axes[i, 0].set_ylabel(row_labels[i][:18], fontsize=7, color='#52514e')
        for j, (ct, mu, h, theta) in enumerate(columns):
            draw_gaussians(axes[i, 1 + j], net, images[i], mu[i], h[i], theta[i])
            if i == 0:
                axes[i, 1 + j].set_title(ct, fontsize=8)
        if ownership:
            ct, mu, h, theta = columns[-1]
            axes[i, -1].imshow(soft_ownership(net, images[i], mu[i], h[i], theta[i]))
            if i == 0:
                axes[i, -1].set_title(f'soft ownership ({ct})', fontsize=8)
    if i == N - 1:
        axes[0, 0].set_title('image', fontsize=8)
    fig.suptitle((title + '\n' if title else '') + f'each token: its Gaussian at 2 sigma (filled, own colour); arrow: anchor -> centre, '
                 f'drawn 4x longer; ownership: pixel colour = tokens\' colours mixed by their normalized blending weights',
                 fontsize=8, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / (2.15 * N + 0.7)))
    fig.savefig(path, dpi=130, facecolor='#fcfcfb')
    plt.close(fig)


# presets: JiT-B's backbone with Gaussian-placed tokens; the anchor spacing names the token count (/16: 64 tokens at 128px)
# bottleneck_dim keeps JiT's compression ratio (768 -> 128 for a 16px patch): the default basis (2 x 0.75 x patch_size per
# side, the initial frame's width in pixels) has (1.5)^2 = 2.25x a patch's coefficients, so 128 x 2.25 = 288
GAUSS_PRESETS = {
    'GaussJiT-B/16': dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=288, patch_size=16),
    'GaussJiT-B/8': dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=288, patch_size=8),
}


def owner_map(net, mu, h, theta):
    # the token each pixel's blending weight is largest for [B, H, W], -1 where no frame covers the pixel
    w = net.dense_weights(mu, h, theta) # [B, K, H, W]
    owner = w.argmax(1)
    return torch.where(w.amax(1) > 0, owner, torch.full_like(owner, -1))

def draw_cells(images, net, mu, h, theta, color='#a8336b'):
    """
    images [N, 3, H, W] in [-1, 1] with every token's owned cell (the pixels where its blending weight is largest) outlined
    and its centre marked: GaussJiT's effective tokenization, drawn like the quadtree's boxes and merge regions in
    compare_tokenizations.py. Pixels no frame covers (only the fallback weight) are lightened.
    """
    N = images.shape[0]
    mu, h, theta = mu[:N].to(images.device), h[:N].to(images.device), theta[:N].to(images.device)
    owner = owner_map(net, mu, h, theta)
    out = images.clone()
    hole = owner < 0
    out = torch.where(hole[:, None], 0.5 * out + 0.5, out)
    edge = torch.zeros_like(hole)
    edge[:, :, :-1] |= owner[:, :, :-1] != owner[:, :, 1:]
    edge[:, :-1] |= owner[:, :-1] != owner[:, 1:]
    rgb = torch.tensor([int(color[i:i + 2], 16) / 127.5 - 1 for i in (1, 3, 5)], device=out.device, dtype=out.dtype)
    out = torch.where(edge[:, None], rgb.view(1, 3, 1, 1), out)
    H, W = out.shape[-2:]
    cx, cy = mu[..., 0].long().clamp(0, W - 1), mu[..., 1].long().clamp(0, H - 1)
    for i in range(N):
        out[i, :, cy[i], cx[i]] = rgb[:, None]
    return out

def coverage_panels(net, x, mu, h, theta, n=8, threshold=0.05):
    """
    Per sample: the sample, its owned cells (draw_cells), full frame outlines, the owner map (each pixel coloured by the
    token with the largest blending weight, token boundaries darkened, uncovered pixels white) and the overlap map (how many
    tokens carry at least `threshold` of the pixel's normalized weight; light = 1, dark = 4+; white = uncovered). Returns
    images [5 n, 3, H, W] in [-1, 1] and labels, for save_labeled_grid with nrow=5.
    """
    import matplotlib
    n = min(n, x.shape[0])
    x, mu, h, theta = x[:n], mu[:n].to(x.device), h[:n].to(x.device), theta[:n].to(x.device)
    w = net.dense_weights(mu, h, theta) # [n, K, H, W]
    wn = w / w.sum(1, keepdim=True).clamp_min(1e-12)
    owner = torch.where(w.amax(1) > 0, wn.argmax(1), torch.full_like(wn[:, 0], -1, dtype=torch.long)) # [n, H, W], -1 uncovered
    K = w.shape[1]
    g = torch.Generator().manual_seed(0)
    palette = (torch.rand(K, 3, generator=g) * 1.4 - 0.7).to(x.device) # distinct mid-range colours per token
    own = palette[owner.clamp_min(0)].permute(0, 3, 1, 2) # [n, 3, H, W]
    own = torch.where((owner < 0)[:, None], torch.ones_like(own), own)
    edge = torch.zeros_like(owner, dtype=torch.bool)
    edge[:, :, :-1] |= owner[:, :, :-1] != owner[:, :, 1:]
    edge[:, :-1] |= owner[:, :-1] != owner[:, 1:]
    own = torch.where(edge[:, None], torch.full_like(own, -1.0), own)
    count = (wn >= threshold).sum(1).float() # [n, H, W]
    cmap = matplotlib.colormaps['Blues']
    lut = torch.tensor([cmap(0.25 + 0.75 * min(k, 4) / 4)[:3] for k in range(K + 1)], device=x.device) * 2 - 1
    cov = lut[count.long().clamp(0, K)].permute(0, 3, 1, 2)
    cov = torch.where((owner < 0)[:, None], torch.ones_like(cov), cov)
    cells = draw_cells(x, net, mu, h, theta)
    frames = draw_geometry(x, mu, h, theta, extent=net.extent, core=False)
    panels, labels = [], []
    for i in range(n):
        panels += [x[i:i + 1], cells[i:i + 1], frames[i:i + 1], own[i:i + 1], cov[i:i + 1]]
        labels += ['sample', 'cells', 'full frames', 'owner', f'overlap {count[i].mean():.1f}, gaps {(owner[i] < 0).float().mean():.1%}']
    return torch.cat(panels), labels
