"""
PixelDiT (Yu et al. 2025, "PixelDiT: Pixel Diffusion Transformers for Image Generation", arXiv 2511.20645), as a drop-in
network for this repo's denoiser: net(z, t, y) -> prediction of the clean image, like JiT, so train_jit.py / eval_jit.py /
eval_linear.py work unchanged (select it with --model=PixelDiT-B/16).

Dual-level design, following the paper's Section 3:
    patch level   N DiT blocks (RMSNorm, 2D RoPE, adaLN-Zero from the global condition c = t_emb + y_emb) on 16px patch
                  tokens from a plain linear patch embedding. JiT's blocks are this LightningDiT-style block.
    semantic      s_cond = s_N + t_emb (the patch-level output plus the timestep embedding)
    pixel level   every pixel of the noisy image embedded to pixel_dim (16) channels (1x1 patchify), then M PiT blocks:
                    pixel-wise adaLN: a projection of s_cond to p^2 x 6 x pixel_dim, i.e. separate shift / scale / gate
                    for every pixel of the patch (paper Eq. 7)
                    token compaction: each patch's p^2 x pixel_dim pixel tokens flattened through a linear map to the
                    hidden width, global RoPE attention over the patches, a linear map back to pixels
                    a per-pixel FFN, both residuals behind pixel-wise gates
                  then RMSNorm + a zero-initialized linear map to RGB per pixel.
Presets (paper Table 3): B N=12 M=2 D=768, L N=22 M=4 D=1024, XL N=26 M=4 D=1152, pixel_dim 16, patch 16.

Choices the paper leaves open: the per-pixel FFN is a SwiGLU FFN with ratio 4 (as in the patch-level blocks), the pixel
output layer is RMSNorm + linear, the patch level keeps JiT's fixed sin-cos position embedding alongside RoPE.
Deviations kept on purpose for comparison with this repo's JiT runs: the default objective is the repo's x-prediction
(prediction='v' gives the paper's velocity output, converted to x = z + (1 - t) v so the denoiser and its v-loss are
unchanged), and no REPA by default (the paper's 1.61 FID uses DINOv2 REPA at block 8 with weight 0.5; train_jit.py's
--repa_* flags reproduce that).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_jit import JiTBlock, Attention, SwiGLUFFN, TimestepEmbedder, LabelEmbedder, modulate
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


class PatchEmbed(nn.Module):
    # plain linear patch embedding (PixelDiT Eq. 1); num_patches as the rest of the repo expects from x_embedder
    def __init__(self, input_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class PiTBlock(nn.Module):
    """
    Pixel Transformer block. x: pixel tokens [B, L, p^2, pixel_dim]; s_cond: semantic tokens [B, L, hidden].
    """
    def __init__(self, hidden_size, num_heads, patch_size, pixel_dim, mlp_ratio=4.0):
        super().__init__()
        self.p2, self.pixel_dim = patch_size * patch_size, pixel_dim
        self.norm1 = RMSNorm(pixel_dim, eps=1e-6)
        self.compress = nn.Linear(self.p2 * pixel_dim, hidden_size) # token compaction C
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True)
        self.expand = nn.Linear(hidden_size, self.p2 * pixel_dim) # expansion E
        self.norm2 = RMSNorm(pixel_dim, eps=1e-6)
        self.mlp = SwiGLUFFN(pixel_dim, int(pixel_dim * mlp_ratio))
        # pixel-wise adaLN (Phi): per-pixel shift / scale / gate for both residuals, from the semantic token
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, self.p2 * 6 * pixel_dim, bias=True))

    def forward(self, x, s_cond, rope):
        B, L = x.shape[:2]
        mod = self.modulation(s_cond).view(B, L, self.p2, 6, self.pixel_dim)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.unbind(3) # each [B, L, p^2, pixel_dim]
        h = self.norm1(x) * (1 + scale1) + shift1
        h = self.expand(self.attn(self.compress(h.reshape(B, L, -1)), rope=rope)).view_as(x)
        x = x + gate1 * h
        x = x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)
        return x


class PixelDiT(nn.Module):
    family = 'pixeldit'

    def __init__(self, input_size=256, patch_size=16, in_channels=3, hidden_size=768, depth=12, num_heads=12,
                 pixel_depth=2, pixel_dim=16, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, num_classes=1000,
                 prediction='x', in_context_len=0, in_context_start=0, family='pixeldit'):
        super().__init__()
        assert prediction in ('x', 'v'), f"prediction must be 'x' or 'v', got {prediction}"
        assert in_context_len == 0, 'PixelDiT has no in-context class tokens'
        self.input_size, self.patch_size, self.in_channels, self.out_channels = input_size, patch_size, in_channels, in_channels
        self.hidden_size, self.num_heads, self.num_classes, self.pixel_dim = hidden_size, num_heads, num_classes, pixel_dim
        self.prediction = prediction
        self.in_context_len, self.in_context_start = 0, 0 # interface shared with JiT (probes, REPA capture, eval_linear)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        half_head_dim = hidden_size // num_heads // 2
        self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=input_size // patch_size, num_cls_token=0)
        self.blocks = nn.ModuleList([JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
                                     for _ in range(depth)])
        self.pixel_embedder = nn.Linear(in_channels, pixel_dim) # 1x1 patchify of the noisy image
        self.pixel_blocks = nn.ModuleList([PiTBlock(hidden_size, num_heads, patch_size, pixel_dim, mlp_ratio) for _ in range(pixel_depth)])
        self.pixel_norm = RMSNorm(pixel_dim, eps=1e-6)
        self.pixel_out = nn.Linear(pixel_dim, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks: # adaLN-Zero
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.pixel_blocks: # zero pixel-wise gates: PiT blocks start as identities
            nn.init.constant_(block.modulation[-1].weight, 0)
            nn.init.constant_(block.modulation[-1].bias, 0)
        nn.init.constant_(self.pixel_out.weight, 0) # zero output at init, as JiT's final layer
        nn.init.constant_(self.pixel_out.bias, 0)

    def patchify(self, x):
        # [B, C, H, W] -> [B, L, p^2, C], patches row-major, pixels row-major within a patch
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.view(B, C, H // p, p, W // p, p).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, (H // p) * (W // p), p * p, C)

    def unpatchify(self, x):
        # [B, L, p^2, C] -> [B, C, H, W]
        B, L, _, C = x.shape
        p, g = self.patch_size, int(round(L ** 0.5))
        return x.view(B, g, g, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, g * p, g * p)

    def semantic_tokens(self, x, t, y):
        # patch-level DiT; returns (s_N, c, t_emb)
        t_emb = self.t_embedder(t)
        c = t_emb + self.y_embedder(y)
        s = self.x_embedder(x)
        s += self.pos_embed # in place: keeps the embedding's dtype under autocast
        for block in self.blocks:
            s = block(s, c, self.feat_rope)
        return s, c, t_emb

    def forward(self, x, t, y):
        s, _, t_emb = self.semantic_tokens(x, t, y)
        s_cond = s + t_emb.unsqueeze(1)
        h = self.pixel_embedder(self.patchify(x).to(s.dtype))
        for block in self.pixel_blocks:
            h = block(h, s_cond, self.feat_rope)
        out = self.unpatchify(self.pixel_out(self.pixel_norm(h)))
        if self.prediction == 'v': # the paper's velocity output, as an x-prediction: z = t x + (1 - t) e, v = (x - z) / (1 - t)
            out = x + (1 - t).view(-1, 1, 1, 1) * out
        return out


# presets (paper Table 3); patch size 16
PIXELDIT_PRESETS = {
    'PixelDiT-B/16': dict(depth=12, pixel_depth=2, hidden_size=768, num_heads=12, pixel_dim=16, patch_size=16),
    'PixelDiT-L/16': dict(depth=22, pixel_depth=4, hidden_size=1024, num_heads=16, pixel_dim=16, patch_size=16),
    'PixelDiT-XL/16': dict(depth=26, pixel_depth=4, hidden_size=1152, num_heads=16, pixel_dim=16, patch_size=16),
}
