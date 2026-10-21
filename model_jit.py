# Vendored from https://github.com/LTH14/JiT (commit cbc743a, MIT license, see LICENSE_JiT).
# Local changes: attn_bias is created on query.device instead of via .cuda(), so the model runs on any device;
# optional Attention Residuals (attn_res, see AttnResState), adaLN-single (adaln_single), a gated bottleneck
# (gated_bottleneck, see GatedBottleneckPatchEmbed), a skip pathway (skip_pathway, see SkipEmbed), DC-AE style conv
# stem / head (dcae_stem / dcae_head, see DCAEBranch), overlapping patches (patch_overlap) and LayerNorm in place of
# every RMSNorm (layernorm), all off by default, in which case the model is unchanged.
# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import copy
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


class LayerNorm(nn.Module):
    """
    LayerNorm with a learnable gain and bias, computed in fp32 and returned in the input's dtype like JiT's RMSNorm
    (nn.LayerNorm under autocast returns fp32, which would promote whatever it feeds). Same eps attribute name as RMSNorm.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        return F.layer_norm(x.float(), self.weight.shape, self.weight, self.bias, self.variance_epsilon).to(x.dtype)


def make_norm(dim, layernorm=False):
    # every norm in the network: RMSNorm (JiT), or LayerNorm with layernorm=True
    return LayerNorm(dim, eps=1e-6) if layernorm else RMSNorm(dim, eps=1e-6)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True, overlap=0):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # overlap (patch_overlap): each token also sees `overlap` pixels of its neighbours on every side
        k = (patch_size[0] + 2 * overlap, patch_size[1] + 2 * overlap)
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=k, stride=patch_size, padding=overlap, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class GatedBottleneckPatchEmbed(BottleneckPatchEmbed):
    """
    JiT's low-rank bottleneck patch embedding plus a full-rank one behind a gate from the conditioning (time + class):
        embedding = proj2(proj1(patch)) + gate(c) * bypass(patch)
    The gate (computed by JiT.bottleneck_gate, zero-initialized) starts closed, i.e. exactly JiT's bottleneck, and can
    open where the input's fine detail is worth having (low noise) while staying shut where the patch is mostly noise.
    Both terms are linear in the patch, so this is one linear embedding whose rank depends on c, not two copies of it.
    Computed inside the embedder so token masking (a hook on its output) replaces the whole embedding.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True, overlap=0):
        super().__init__(img_size, patch_size, in_chans, pca_dim, embed_dim, bias, overlap)
        self.bypass = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def bypass_tokens(self, x):
        return self.bypass(x).flatten(2).transpose(1, 2)

    def forward(self, x, gate=None):
        out = super().forward(x)
        if gate is None:
            return out
        return out + gate.unsqueeze(1) * self.bypass_tokens(x)


class DCAEBranch(nn.Module):
    """
    Residual conv branch of a DC-AE style resampling layer (Chen et al. 2024, Deep Compression Autoencoder). At JiT's
    widths every DC-AE shortcut is an exact reshape (3 * 4^2 = 48, 48 * 4^2 = 768 = one 16px patch), so a DC-AE layer
    is shuffle(x + branch(x)) and the branch is a residual at one resolution:
        x + gate(c) * conv_k2(SiLU(conv3x3(modulate(RMSNorm_channels(x), shift(c), scale(c)))))   channels -> width -> channels
    adaLN modulation from the conditioning with a zero-initialized gate, so the branch starts as the identity.
    """
    def __init__(self, channels, width, kernel2, cond_dim, layernorm=False):
        super().__init__()
        self.norm = make_norm(channels, layernorm)
        self.conv1 = nn.Conv2d(channels, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, channels, kernel_size=kernel2, padding=kernel2 // 2)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * channels, bias=True))

    def forward(self, x, c):
        shift, scale, gate = self.modulation(c).to(x.dtype).chunk(3, dim=-1)
        h = self.norm(x.permute(0, 2, 3, 1)) * (1 + scale[:, None, None]) + shift[:, None, None]
        h = self.conv2(F.silu(self.conv1(h.permute(0, 3, 1, 2))))
        return x + gate[:, :, None, None] * h


class DCAEStem(nn.Module):
    """
    DC-AE downsampling (the conv runs at the higher resolution of each stage) in front of JiT's patch embedding:
    a branch at full resolution (3 ch), PixelUnshuffle(4), a branch at the 4px level (48 ch), PixelShuffle(4) back to an
    image for the (unchanged) bottleneck patch embedding, which does the last 4 -> 16 step.
    """
    def __init__(self, in_chans, width, kernel2, cond_dim, layernorm=False):
        super().__init__()
        self.full = DCAEBranch(in_chans, max(8, width // 4), kernel2, cond_dim, layernorm) # 3 channels at full resolution: kept narrow
        self.quarter = DCAEBranch(in_chans * 16, width, kernel2, cond_dim, layernorm)

    def forward(self, x, c):
        x = self.full(x, c)
        return F.pixel_shuffle(self.quarter(F.pixel_unshuffle(x, 4), c), 4)


class DCAEHead(nn.Module):
    """
    DC-AE upsampling (the conv runs at the lower resolution of each stage) on the predicted image: a branch at the token
    level (768 ch, one 16px patch per position), PixelShuffle(4), a branch at the 4px level (48 ch), PixelShuffle(4).
    """
    def __init__(self, in_chans, patch_size, width, kernel2, cond_dim, layernorm=False):
        super().__init__()
        assert patch_size == 16, 'dcae_head is built for 16px patches (two 4x stages)'
        self.token = DCAEBranch(in_chans * 256, width, kernel2, cond_dim, layernorm)
        self.quarter = DCAEBranch(in_chans * 16, width, kernel2, cond_dim, layernorm)

    def forward(self, img, c):
        t16 = self.token(F.pixel_unshuffle(F.pixel_unshuffle(img, 4), 4), c) # two-stage ordering, so the shuffles invert
        return F.pixel_shuffle(self.quarter(F.pixel_shuffle(t16, 4), c), 4)


class SkipEmbed(nn.Module):
    """
    Skip pathway: a full-rank patch embedding of the noisy input that bypasses the trunk entirely and is added, behind
    a gate from the conditioning (JiT.skip_gate, zero-initialized), to the trunk's output just before the final layer
    (and the optional skip_head blocks). The trunk only ever sees the bottlenecked embedding, at every noise level, so
    it can't solve low-noise steps by editing the input locally; the input's fine detail is available only at the
    output stage. A separate module so token masking can blank masked patches here too (masking.TokenMasker).
    """
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def resolve_skip_inject(skip_inject_block, depth):
    # skip_inject_block: 0 -> after the last block, k > 0 -> after block k, -k -> before the last k blocks
    after = depth if skip_inject_block == 0 else (depth + skip_inject_block if skip_inject_block < 0 else skip_inject_block)
    assert 1 <= after <= depth, f'skip_inject_block {skip_inject_block} is outside the {depth}-block trunk'
    return after


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype, device=query.device)

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
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
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, adaln_single=False, layernorm=False):
        super().__init__()
        self.norm_final = make_norm(hidden_size, layernorm)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaln_single = adaln_single
        if adaln_single:
            # PixArt-alpha's final layer: a learned table plus the conditioning itself, no projection
            self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

    def modulation(self, c):
        if self.adaln_single:
            return (self.scale_shift_table[None] + F.silu(c)[:, None]).unbind(1)
        return self.adaLN_modulation(c).chunk(2, dim=1)

    @torch.compile
    def forward(self, x, c):
        shift, scale = (m.to(x.dtype) for m in self.modulation(c)) # adaln_single's fp32 table would promote x otherwise
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, adaln_single=False, layernorm=False):
        super().__init__()
        self.norm1 = make_norm(hidden_size, layernorm)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop, layernorm=layernorm)
        self.norm2 = make_norm(hidden_size, layernorm)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaln_single = adaln_single
        if adaln_single:
            # adaLN-single (PixArt-alpha): the 6 modulation vectors come from one projection shared by all blocks
            # (JiT.t_block), plus this block's learned table
            self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

    def modulation(self, c):
        """
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp), each [N, hidden]. c is the conditioning
        vector [N, hidden], or with adaln_single the shared projection of it [N, 6 * hidden].
        """
        if self.adaln_single:
            # back to c's dtype (bf16 under autocast, like a Linear's output): the fp32 table would otherwise promote the
            # modulation, and with it every gated branch output / AttnRes source, to fp32
            return (self.scale_shift_table[None] + c.view(c.shape[0], 6, -1)).to(c.dtype).unbind(1)
        return self.adaLN_modulation(c).chunk(6, dim=-1)

    def forward(self, x, c, feat_rope=None, res=None, index=None):
        if res is None:
            return self.forward_residual(x, c, feat_rope)
        return self.forward_attn_res(x, c, feat_rope, res, index)

    @torch.compile
    def forward_residual(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(c)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    @torch.compile
    def attn_branch(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa = self.modulation(c)[:3]
        return gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)

    @torch.compile
    def mlp_branch(self, x, c):
        shift_mlp, scale_mlp, gate_mlp = self.modulation(c)[3:]
        return gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

    def forward_attn_res(self, x, c, feat_rope, res, index):
        """
        x: this block's attention input (already depth-mixed). Returns the next block's attention input (the final
        layer's input for the last block), so forward hooks see the representation the network actually passes on.
        """
        res.add(self.attn_branch(x, c, feat_rope))
        res.add(self.mlp_branch(res.mix(2 * index + 1), c))
        return res.mix(2 * index + 2)


class _DepthMix(torch.autograd.Function):
    """
    h = sum_i alpha_i v_i over S sources, accumulated in place (one read of each source, no stacked copy), saving only
    alpha and references to the sources, which autograd keeps anyway. alpha: [S, B, T]; sources: S tensors [B, T, D].
    """
    @staticmethod
    def forward(ctx, alpha, *sources):
        ctx.save_for_backward(alpha, *sources)
        a = alpha.unsqueeze(-1).to(sources[0].dtype)
        h = sources[0] * a[0]
        for i in range(1, len(sources)):
            h.addcmul_(sources[i], a[i])
        return h

    @staticmethod
    def backward(ctx, grad):
        alpha, *sources = ctx.saved_tensors
        grad_alpha = torch.stack([torch.linalg.vecdot(v, grad.to(v.dtype)) for v in sources]).to(alpha.dtype) # [S, B, T]
        a = alpha.unsqueeze(-1).to(grad.dtype)
        return (grad_alpha, *(grad * a[i] for i in range(len(sources))))


class AttnResState:
    """
    Attention Residuals (Kimi Team 2026, arXiv 2603.15031): instead of the fixed sum h_l = h_1 + sum_i f_i(h_i), the
    input of every sublayer l (each attention and each MLP counts as one) is a softmax mix over depth,
        h_l = sum_i alpha_{i->l} v_i,  alpha_{i->l} = softmax_i(w_l . RMSNorm(v_i)),
    with one learned pseudo-query w_l per sublayer, initialized to zero (uniform mixing at the start of training).
    Sources v_i: the patch embedding, then
        full    every sublayer output
        block   (Block AttnRes) the sum of the sublayer outputs of each completed block of block_size JiT blocks, plus
                the running partial sum of the current block
    The final layer reads a mix with its own query. In JiT the sublayer outputs are the adaLN-gated branch outputs.
    In-context class tokens: when they are prepended, the embedding source gets the class tokens at those positions
    and earlier sources get zeros (those positions had no earlier outputs), matching what the plain residual stream holds.

    Efficiency: w_l . RMSNorm(v) = (v . (g_l * w_l)) / rms(v). The queries are parameters, so when a source is created its
    logits for every sublayer are one matmul against all the (g_l * w_l) at once, and a mix only gathers them (as the
    paper's two-phase computation does). The weighted sum is accumulated in place in eager mode (_DepthMix) and left to
    the compiler's fusion under whole-model compile. Nothing [S, B, T, D]-sized is saved for backward.
    """
    def __init__(self, net, emb, c=None):
        """
        c: the conditioning vector [B, D], needed with attn_res_adaln: adaLN on the queries,
        w_l(c) = w_l * (1 + scale(c)) + shift(c), from one zero-initialized projection shared by all of them (as in
        adaLN-single). On the key side the shift would cancel in the softmax over sources, so it goes on the query.
        """
        self.queries, self.norms = net.attn_res_queries, net.attn_res_norms
        self.eps = self.norms[0].variance_epsilon
        self.centered = isinstance(self.norms[0], LayerNorm) # LayerNorm keys: scores need the mean subtracted
        self.group = 1 if net.attn_res == 'full' else 2 * net.attn_res_block_size # sublayers per source
        self.keys = self.make_keys(net, c)
        self.sources, self.logits = [], [] # completed sources and their logits for every sublayer [B, T, 2 * depth + 1]
        self.partial = None
        self.count = 0
        self.weights = None # {sublayer: mean mixing weight per source [S]} when recording (diagnostics only)
        self._append(emb)

    def make_keys(self, net, c):
        gains = torch.stack([n.weight for n in self.norms]) # [2 * depth + 1, D]
        queries = torch.stack(list(self.queries))
        if net.attn_res_adaln:
            shift, scale = net.attn_res_modulation(c).unsqueeze(1).chunk(2, dim=-1) # [B, 1, D] each
            queries = queries * (1 + scale) + shift # [B, 2 * depth + 1, D]
        return gains * queries # the key norm's gain folded into the queries: [2 * depth + 1, D] or per sample [B, ., D]

    def detached(self, net, c):
        """
        Copy whose sources (and running block sum) are cut from the autograd graph, for a second branch that continues
        from the same state without sending gradient back into what produced it (skip_stop_grad). Scores are recomputed
        so later layers' queries still learn from the copy; c is the (detached) conditioning for attn_res_adaln.
        """
        new = copy.copy(self)
        new.keys = new.make_keys(net, c)
        new.sources = [s.detach() for s in self.sources]
        new.logits = [new._logits(s, new.keys) for s in new.sources]
        new.partial = self.partial.detach() if self.partial is not None else None
        new.weights = None
        return new

    def _logits(self, v, keys):
        # v: [B, T, D] -> w . RMSNorm(v) for each row of keys, fp32 [B, T, K]
        ms = torch.linalg.vecdot(v, v).unsqueeze(-1).float() / v.shape[-1] # mean square, [B, T, 1]
        dots = (v @ keys.transpose(-1, -2).to(v.dtype)).float()
        if not self.centered:
            return dots * torch.rsqrt(ms + self.eps)
        # w . LayerNorm(v) = (v . u - mean(v) sum(u)) / std(v) + w . b, with u = g * w; w . b is the same for every source,
        # so it cancels in the softmax over sources and is left out
        mean = v.float().mean(-1, keepdim=True)
        ksum = keys.float().sum(-1)
        ksum = ksum[:, None, :] if ksum.dim() == 2 else ksum
        return (dots - mean * ksum) * torch.rsqrt(ms - mean ** 2 + self.eps)

    def _append(self, v):
        self.sources.append(v)
        self.logits.append(self._logits(v, self.keys))

    def add(self, out):
        self.partial = out if self.partial is None else self.partial + out
        self.count += 1
        if self.count == self.group:
            self._append(self.partial)
            self.partial, self.count = None, 0

    def mix(self, l):
        sources, logits = self.sources, [c[..., l] for c in self.logits]
        if self.partial is not None:
            sources = sources + [self.partial]
            logits.append(self._logits(self.partial, self.keys[..., l:l + 1, :])[..., 0])
        if len(sources) == 1: # the first sublayer only sees the embedding (so its query never trains)
            if self.weights is not None:
                self.weights[l] = torch.ones(1, device=sources[0].device)
            return sources[0]
        alpha = torch.stack(logits).softmax(0)
        if self.weights is not None:
            self.weights[l] = alpha.float().mean(dim=(1, 2))
        if torch.compiler.is_compiling():
            # whole-model compile unrolls the block loop, so S is fixed here: the plain weighted sum fuses into one
            # kernel and the compiler saves only alpha and the sources for backward
            return sum(alpha[i].unsqueeze(-1).to(v.dtype) * v for i, v in enumerate(sources))
        return _DepthMix.apply(alpha, *sources)

    def prepend(self, prefix):
        # in-context tokens: the embedding source gets them, every other source zeros at those positions
        zeros = torch.zeros_like(prefix)
        sources, self.sources, self.logits = self.sources, [], []
        for i, v in enumerate(sources):
            self._append(torch.cat([prefix if i == 0 else zeros, v], dim=1))
        if self.partial is not None:
            self.partial = torch.cat([zeros, self.partial], dim=1)


class JiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        attn_res='none',
        attn_res_block_size=0,
        adaln_single=False,
        attn_res_adaln=False,
        gated_bottleneck=False,
        skip_pathway=False,
        skip_head_blocks=0,
        skip_inject_block=0,
        dcae_stem=False,
        dcae_head=False,
        dcae_width=128,
        dcae_kernel2=1,
        patch_overlap=0,
        layernorm=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        assert attn_res in ('none', 'full', 'block'), f'attn_res must be none, full or block, got {attn_res}'
        self.attn_res = attn_res
        # JiT blocks per Block AttnRes block; 0 -> about 8 blocks, the paper's default
        self.attn_res_block_size = attn_res_block_size or max(1, round(depth / 8))
        self.adaln_single = adaln_single
        assert not attn_res_adaln or attn_res != 'none', 'attn_res_adaln needs attn_res'
        self.attn_res_adaln = attn_res_adaln
        self.gated_bottleneck = gated_bottleneck
        assert skip_head_blocks == 0 or skip_pathway, 'skip_head_blocks needs skip_pathway'
        self.skip_pathway = skip_pathway
        self.dcae_stem, self.dcae_head, self.patch_overlap = dcae_stem, dcae_head, patch_overlap
        self.layernorm = layernorm
        assert not dcae_stem or patch_size == 16, 'dcae_stem is built for 16px patches'
        # where the skip enters: after block skip_inject_after (1-indexed); 0 -> after the last block, -k -> before the last k
        self.skip_inject_after = resolve_skip_inject(skip_inject_block, depth)
        # training only (set by train_jit.py, not part of the architecture): fraction of samples whose skip is dropped,
        # so the trunk still has to reconstruct them through the bottleneck
        self.skip_drop_prob = 0.0
        # training only (set by train_jit.py): also train a trunk-only prediction, with the skip branch's loss stopped
        # at the injection point (masking.flow_loss asks for both predictions via forward(..., return_trunk=True))
        self.skip_stop_grad = False
        assert skip_inject_block == 0 or skip_pathway, 'skip_inject_block needs skip_pathway'

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        if adaln_single:
            # one modulation projection shared by every block (PixArt-alpha's t_block), instead of one per block
            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # linear embed
        embedder = GatedBottleneckPatchEmbed if gated_bottleneck else BottleneckPatchEmbed
        self.x_embedder = embedder(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True, overlap=patch_overlap)
        if dcae_stem:
            self.input_stem = DCAEStem(in_channels, dcae_width, dcae_kernel2, hidden_size, layernorm)
        if gated_bottleneck:
            # per-channel gate on the full-rank patch embedding, from the conditioning (adaLN-style, zero-initialized)
            self.bottleneck_gate = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )

        # transformer
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0, adaln_single=adaln_single, layernorm=layernorm)
            for i in range(depth)
        ])

        # skip pathway: full-rank input patches merged into the trunk's output (gated), then optional head blocks
        if skip_pathway:
            self.skip_embedder = SkipEmbed(patch_size, in_channels, hidden_size)
            self.skip_gate = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
            self.skip_head = nn.ModuleList([JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, adaln_single=adaln_single, layernorm=layernorm)
                                            for _ in range(skip_head_blocks)])

        # linear predict
        # with patch_overlap the final layer predicts overlapping (patch + 2 * overlap)^2 patches, blended by overlap-add
        self.final_layer = FinalLayer(hidden_size, patch_size + 2 * patch_overlap, self.out_channels, adaln_single=adaln_single, layernorm=layernorm)
        if dcae_head:
            self.output_head = DCAEHead(self.out_channels, patch_size, dcae_width, dcae_kernel2, hidden_size, layernorm)

        # attention residuals: a zero-initialized pseudo-query and a key RMSNorm per sublayer (2 per block) and the final layer
        if self.attn_res != 'none':
            self.attn_res_queries = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(2 * depth + 1)])
            self.attn_res_norms = nn.ModuleList([make_norm(hidden_size, layernorm) for _ in range(2 * depth + 1)])
            if attn_res_adaln:
                # adaLN on the depth-attention queries: one (shift, scale) projection of c shared by all of them
                self.attn_res_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)
        if self.gated_bottleneck: # the bypass like the patch embedding; its gate starts closed (= JiT's bottleneck)
            w = self.x_embedder.bypass.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.bottleneck_gate[-1].weight, 0)
            nn.init.constant_(self.bottleneck_gate[-1].bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.skip_pathway: # the skip embedding like the patch embedding; its gate starts closed (= no skip)
            w = self.skip_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.skip_gate[-1].weight, 0)
            nn.init.constant_(self.skip_gate[-1].bias, 0)
        for branch in [b for m in (getattr(self, 'input_stem', None), getattr(self, 'output_head', None)) if m is not None
                       for b in m.children()]: # DC-AE branches start as identities
            nn.init.constant_(branch.modulation[-1].weight, 0)
            nn.init.constant_(branch.modulation[-1].bias, 0)
        if self.attn_res_adaln: # starts as plain AttnRes
            nn.init.constant_(self.attn_res_modulation[-1].weight, 0)
            nn.init.constant_(self.attn_res_modulation[-1].bias, 0)
        if self.adaln_single:
            # PixArt-alpha's init: the shared projection starts at zero, so modulation starts at the (random) per-block
            # tables; blocks start as identities through zero-initialized attention / MLP output projections instead of
            # zero gates (JiT's adaLN-Zero)
            nn.init.constant_(self.t_block[-1].weight, 0)
            nn.init.constant_(self.t_block[-1].bias, 0)
            for block in self.all_blocks():
                nn.init.constant_(block.attn.proj.weight, 0)
                nn.init.constant_(block.mlp.w3.weight, 0)
        else:
            # Zero-out adaLN modulation layers:
            for block in self.all_blocks():
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def gated_skip(self, skip, c):
        # gate(c) * skip; in training, each sample's skip is dropped (gate zeroed) with probability skip_drop_prob, the way
        # label dropout works for CFG, so the trunk keeps reconstructing those samples through the bottleneck alone
        gate = self.skip_gate(c)
        if self.training and self.skip_drop_prob > 0:
            gate = gate * (torch.rand(gate.shape[0], 1, device=gate.device) >= self.skip_drop_prob).to(gate.dtype)
        return gate.unsqueeze(1) * skip

    def inject_skip(self, x, gated, res, i):
        """
        Mid-trunk skip injection after block i (0-indexed), returning the next block's input. Without AttnRes the gated
        skip is added to the residual stream. With AttnRes it's added into block i's output source (the running block
        sum for Block AttnRes), the AttnRes form of a residual add: later sublayers and the final layer reach it through
        that source. (A separate source would not be a no-op with the gate shut: a zero source still takes softmax weight
        and dilutes the others.) In-context class tokens, if present at this depth, get zeros.
        """
        n_prefix = x.shape[1] - gated.shape[1]
        if n_prefix:
            gated = torch.cat([gated.new_zeros(gated.shape[0], n_prefix, gated.shape[2]), gated], dim=1)
        if res is None:
            return x + gated
        if res.partial is not None:
            res.partial = res.partial + gated
        else:
            res.sources[-1] = res.sources[-1] + gated
            res.logits[-1] = res._logits(res.sources[-1], res.keys)
        return res.mix(2 * (i + 1))

    def all_blocks(self):
        # the trunk's blocks plus the skip pathway's head blocks (for initialization)
        return list(self.blocks) + (list(self.skip_head) if self.skip_pathway else [])

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def run_blocks(self, x, res, start, end, c_blocks, y_emb):
        # trunk blocks start..end-1 (0-indexed), with the in-context class tokens prepended at in_context_start
        for i in range(start, end):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                if res is None:
                    x = torch.cat([in_context_tokens, x], dim=1)
                else:
                    res.prepend(in_context_tokens)
                    x = res.mix(2 * i)
            x = self.blocks[i](x, c_blocks, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext, res=res, index=i)
        return x

    def forward(self, x, t, y, return_trunk=False):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        return_trunk (skip_pathway): also return the trunk-only prediction, i.e. the same network without the skip
        (skip_stop_grad in training). The skip branch then continues from a detached copy of the trunk's state at the
        injection point, so its loss trains only the skip pathway, the blocks after the injection point and the final
        layer, never the embeddings or the blocks before it. Returns (prediction with skip, trunk-only prediction).
        """
        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        c_blocks = self.t_block(c) if self.adaln_single else c

        # forward JiT
        skip = self.skip_embedder(x) if self.skip_pathway else None # from the noisy input, before x is reused
        x = self.embed(x, c)

        res = AttnResState(self, x, c) if self.attn_res != 'none' else None
        depth = len(self.blocks)
        k = self.skip_inject_after if self.skip_pathway else depth
        x = self.run_blocks(x, res, 0, k, c_blocks, y_emb)
        if not self.skip_pathway:
            return self.output(x[:, self.in_context_len:], c)

        two = return_trunk
        if two: # skip branch from a detached copy of the trunk at the injection point
            xs, cs, cbs, ys = x.detach(), c.detach(), c_blocks.detach(), y_emb.detach()
            ress = res.detached(self, cs) if res is not None else None
        else:
            xs, cs, cbs, ys, ress = x, c, c_blocks, y_emb, res
        gated = self.gated_skip(skip, cs)
        if k < depth: # mid-trunk: into block k's output, then the remaining blocks
            xs = self.inject_skip(xs, gated, ress, k - 1)
            xs = self.run_blocks(xs, ress, k, depth, cbs, ys)[:, self.in_context_len:]
        else: # after the trunk: added to its output directly
            xs = xs[:, self.in_context_len:] + gated
        for block in self.skip_head:
            xs = block(xs, cbs, self.feat_rope)
        output = self.output(xs, cs)
        if not two:
            return output
        # trunk-only branch, run last so block hooks (e.g. REPA's capture) see the trunk's own activations
        x_trunk = self.run_blocks(x, res, k, depth, c_blocks, y_emb)[:, self.in_context_len:]
        return output, self.output(x_trunk, c)

    def embed(self, x, c):
        # (DC-AE stem) -> patch embedding (gated bottleneck) -> + position embedding
        if self.dcae_stem:
            x = self.input_stem(x, c)
        x = self.x_embedder(x, self.bottleneck_gate(c)) if self.gated_bottleneck else self.x_embedder(x)
        x += self.pos_embed # in place, as JiT does: keeps the embedding's dtype (bf16 under autocast), the fp32 table would promote it
        return x

    def output(self, x, c):
        # final layer -> image (unpatchify, or overlap-add for patch_overlap) -> (DC-AE head)
        x = self.final_layer(x, c)
        if self.patch_overlap:
            o, p = self.patch_overlap, self.patch_size
            k, size = p + 2 * o, self.input_size
            cols = x.transpose(1, 2).float() # [N, C * k * k, L], channel-major patches as F.fold expects
            img = F.fold(cols, size, kernel_size=k, stride=p, padding=o)
            count = F.fold(torch.ones_like(cols[:1]), size, kernel_size=k, stride=p, padding=o)
            img = (img / count).to(x.dtype)
        else:
            img = self.unpatchify(x, self.patch_size)
        if self.dcae_head:
            img = self.output_head(img, c)
        return img


def JiT_B_16(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)

def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)

def JiT_L_16(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)

def JiT_L_32(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)

def JiT_H_16(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)

def JiT_H_32(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}
