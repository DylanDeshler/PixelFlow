# Vendored from https://github.com/LTH14/JiT (commit cbc743a, MIT license, see LICENSE_JiT).
# Local changes: attn_bias is created on query.device instead of via .cuda(), so the model runs on any device;
# optional Attention Residuals (attn_res, see AttnResState), off by default, in which case the model is unchanged.
# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


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
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

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
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, feat_rope=None, res=None, index=None):
        if res is None:
            return self.forward_residual(x, c, feat_rope)
        return self.forward_attn_res(x, c, feat_rope, res, index)

    @torch.compile
    def forward_residual(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    @torch.compile
    def attn_branch(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(6, dim=-1)[:3]
        return gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)

    @torch.compile
    def mlp_branch(self, x, c):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)[3:]
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
    def __init__(self, net, emb):
        self.queries, self.norms = net.attn_res_queries, net.attn_res_norms
        self.eps = self.norms[0].variance_epsilon
        self.group = 1 if net.attn_res == 'full' else 2 * net.attn_res_block_size # sublayers per source
        self.keys = torch.stack([n.weight * q for n, q in zip(self.norms, self.queries)]) # [2 * depth + 1, D]
        self.sources, self.logits = [], [] # completed sources and their logits for every sublayer [B, T, 2 * depth + 1]
        self.partial = None
        self.count = 0
        self._append(emb)

    def _logits(self, v, keys):
        # v: [B, T, D] -> w . RMSNorm(v) for each row of keys, fp32 [B, T, K]
        ms = torch.linalg.vecdot(v, v).unsqueeze(-1).float() / v.shape[-1] # mean square, [B, T, 1]
        return (v @ keys.t().to(v.dtype)).float() * torch.rsqrt(ms + self.eps)

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
            logits.append(self._logits(self.partial, self.keys[l:l + 1])[..., 0])
        if len(sources) == 1: # the first sublayer only sees the embedding (so its query never trains)
            return sources[0]
        alpha = torch.stack(logits).softmax(0)
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
        attn_res_block_size=0
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

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

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
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # attention residuals: a zero-initialized pseudo-query and a key RMSNorm per sublayer (2 per block) and the final layer
        if self.attn_res != 'none':
            self.attn_res_queries = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(2 * depth + 1)])
            self.attn_res_norms = nn.ModuleList([RMSNorm(hidden_size) for _ in range(2 * depth + 1)])

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

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

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

    def forward(self, x, t, y):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        """
        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        # forward JiT
        x = self.x_embedder(x)
        x += self.pos_embed

        res = AttnResState(self, x) if self.attn_res != 'none' else None
        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                if res is None:
                    x = torch.cat([in_context_tokens, x], dim=1)
                else:
                    res.prepend(in_context_tokens)
                    x = res.mix(2 * i)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext, res=res, index=i)

        x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        return output


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
