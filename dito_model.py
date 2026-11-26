"""
DiTo (Chen et al. 2025, "Diffusion Autoencoders are Scalable Image Tokenizers"), from the official code
(https://github.com/yinboc/dito, Apache-2.0). The network modules below are copied verbatim from it (the registry decorators
removed): the LDM/VQGAN convolutional encoder (models/ldm/vqgan/model.py: Encoder and what it uses) with its f8c4 config
(models/ldm/vqgan/utils.py), the Consistency-Decoder UNet (models/networks/consistency_decoder_unet.py), flow matching
(models/diffusion/fm.py) and its Euler sampler (models/diffusion/samplers.py). The DiTo class at the end re-assembles the
paper's tokenizer as models/ldm/dito.py + ldm_base.py do for the configs used here (z_layernorm, noise synchronization
zaug_p = 0.1 with the 'suffix' decoding loss, renderer = the UNet, 50 Euler steps), so its loss and decoding are the paper's.

Sizes: 'B' is the official dito-B-f8c4 config (encoder ch 128; UNet c0/c1/c2 128/256/512, pe 320, t 1280: 162.8M decoder
params at 256px). 'S' is not in the paper: the same architecture at half the channels (encoder ch 64; UNet 64/128/256), our
definition, for the tokenizer-compute axis.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---- models/ldm/vqgan/model.py (verbatim) ---------------------------------------------------------------------------------
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, kernel_size=3, conv_shortcut=False,
                 dropout, temb_channels=512, normalize=True):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels) if normalize else torch.nn.Identity()
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=(kernel_size - 1) // 2)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels) if normalize else torch.nn.Identity()
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=(kernel_size - 1) // 2)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=kernel_size,
                                                     stride=1,
                                                     padding=(kernel_size - 1) // 2)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    # print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)



class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# ---- models/ldm/vqgan/utils.py: the f8c4 encoder config (verbatim) -----------------------------------------------------------
default_configs = {
    'f8c4': dict(
        double_z=False,
        z_channels=4,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        give_pre_end=True,
    ),
    'f16c8': dict(
        double_z=False,
        z_channels=8,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        give_pre_end=True,
    ),
}


# ---- models/networks/consistency_decoder_unet.py (verbatim, registry removed; its Downsample / Upsample renamed CD* so they
#      don't shadow the encoder's classes of the same names in this one module) ---------------------------------------------------

class TimestepEmbedding(nn.Module):
    def __init__(self, n_time=1024, n_emb=320, n_out=1280) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_time, n_emb)
        self.f_1 = nn.Linear(n_emb, n_out)
        self.f_2 = nn.Linear(n_out, n_out)

    def forward(self, x) -> torch.Tensor:
        x = self.emb(x)
        x = self.f_1(x)
        x = F.silu(x)
        return self.f_2(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, pe_dim=320, out_dim=1280, max_positions=10000, endpoint=True):
        super().__init__()
        self.num_channels = pe_dim
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.f_1 = nn.Linear(pe_dim, out_dim)
        self.f_2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        
        x = self.f_1(x)
        x = F.silu(x)
        return self.f_2(x)


class ImageEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels=320) -> None:
        super().__init__()
        self.f = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        return self.f(x)


class ImageUnembedding(nn.Module):
    def __init__(self, in_channels=320, out_channels=3) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(32, in_channels)
        self.f = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        return self.f(F.silu(self.gn(x)))


class ConvResblock(nn.Module):
    def __init__(self, in_features, out_features, t_dim) -> None:
        super().__init__()
        self.f_t = nn.Linear(t_dim, out_features * 2)

        self.gn_1 = nn.GroupNorm(32, in_features)
        self.f_1 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)

        self.gn_2 = nn.GroupNorm(32, out_features)
        self.f_2 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1)

        skip_conv = in_features != out_features
        self.f_s = (
            nn.Conv2d(in_features, out_features, kernel_size=1, padding=0)
            if skip_conv
            else nn.Identity()
        )

    def forward(self, x, t):
        x_skip = x
        t = self.f_t(F.silu(t))
        t = t.chunk(2, dim=1)
        t_1 = t[0].unsqueeze(dim=2).unsqueeze(dim=3) + 1
        t_2 = t[1].unsqueeze(dim=2).unsqueeze(dim=3)

        gn_1 = F.silu(self.gn_1(x))
        f_1 = self.f_1(gn_1)

        gn_2 = self.gn_2(f_1)

        return self.f_s(x_skip) + self.f_2(F.silu(gn_2 * t_1 + t_2))


# Also ConvResblock
class CDDownsample(nn.Module):
    def __init__(self, in_channels, t_dim) -> None:
        super().__init__()
        self.f_t = nn.Linear(t_dim, in_channels * 2)

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2).unsqueeze(3) + 1
        t_2 = t_2.unsqueeze(2).unsqueeze(3)

        gn_1 = F.silu(self.gn_1(x))
        avg_pool2d = F.avg_pool2d(gn_1, kernel_size=(2, 2), stride=None)
        f_1 = self.f_1(avg_pool2d)
        gn_2 = self.gn_2(f_1)

        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + F.avg_pool2d(x_skip, kernel_size=(2, 2), stride=None)


# Also ConvResblock
class CDUpsample(nn.Module):
    def __init__(self, in_channels, t_dim) -> None:
        super().__init__()
        self.f_t = nn.Linear(t_dim, in_channels * 2)

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2).unsqueeze(3) + 1
        t_2 = t_2.unsqueeze(2).unsqueeze(3)

        gn_1 = F.silu(self.gn_1(x))
        upsample = F.upsample_nearest(gn_1, scale_factor=2)
        f_1 = self.f_1(upsample)
        gn_2 = self.gn_2(f_1)

        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + F.upsample_nearest(x_skip, scale_factor=2)


class ConsistencyDecoderUNet(nn.Module):
    def __init__(self, in_channels=3, z_dec_channels=None, c0=320, c1=640, c2=1024, pe_dim=320, t_dim=1280) -> None:
        super().__init__()
        if z_dec_channels is not None:
            in_channels += z_dec_channels
        self.embed_image = ImageEmbedding(in_channels=in_channels, out_channels=c0)
        self.embed_time = PositionalEmbedding(pe_dim=pe_dim, out_dim=t_dim)

        down_0 = nn.ModuleList([
            ConvResblock(c0, c0, t_dim),
            ConvResblock(c0, c0, t_dim),
            ConvResblock(c0, c0, t_dim),
            CDDownsample(c0, t_dim),
        ])
        down_1 = nn.ModuleList([
            ConvResblock(c0, c1, t_dim),
            ConvResblock(c1, c1, t_dim),
            ConvResblock(c1, c1, t_dim),
            CDDownsample(c1, t_dim),
        ])
        down_2 = nn.ModuleList([
            ConvResblock(c1, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
            CDDownsample(c2, t_dim),
        ])
        down_3 = nn.ModuleList([
            ConvResblock(c2, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
        ])
        self.down = nn.ModuleList([
            down_0,
            down_1,
            down_2,
            down_3,
        ])

        self.mid = nn.ModuleList([
            ConvResblock(c2, c2, t_dim),
            ConvResblock(c2, c2, t_dim),
        ])

        up_3 = nn.ModuleList([
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            CDUpsample(c2, t_dim),
        ])
        up_2 = nn.ModuleList([
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 * 2, c2, t_dim),
            ConvResblock(c2 + c1, c2, t_dim),
            CDUpsample(c2, t_dim),
        ])
        up_1 = nn.ModuleList([
            ConvResblock(c2 + c1, c1, t_dim),
            ConvResblock(c1 * 2, c1, t_dim),
            ConvResblock(c1 * 2, c1, t_dim),
            ConvResblock(c0 + c1, c1, t_dim),
            CDUpsample(c1, t_dim),
        ])
        up_0 = nn.ModuleList([
            ConvResblock(c0 + c1, c0, t_dim),
            ConvResblock(c0 * 2, c0, t_dim),
            ConvResblock(c0 * 2, c0, t_dim),
            ConvResblock(c0 * 2, c0, t_dim),
        ])
        self.up = nn.ModuleList([
            up_0,
            up_1,
            up_2,
            up_3,
        ])

        self.output = ImageUnembedding(in_channels=c0)
    
    def get_last_layer_weight(self):
        return self.output.f.weight

    def forward(self, x, t=None, z_dec=None) -> torch.Tensor:
        if z_dec is not None:
            if z_dec.shape[-2] != x.shape[-2] or z_dec.shape[-1] != x.shape[-1]:
                assert x.shape[-2] // z_dec.shape[-2] == x.shape[-1] // z_dec.shape[-1]
                z_dec = F.upsample_nearest(z_dec, scale_factor=x.shape[-2] // z_dec.shape[-2])
            x = torch.cat([x, z_dec], dim=1)
        
        x = self.embed_image(x)

        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)        
        t = self.embed_time(t)

        skips = [x]
        for down in self.down:
            for block in down:
                x = block(x, t)
                skips.append(x)

        for mid in self.mid:
            x = mid(x, t)

        for up in self.up[::-1]:
            for block in up:
                if isinstance(block, ConvResblock):
                    x = torch.concat([x, skips.pop()], dim=1)
                x = block(x, t)

        return self.output(x)


# ---- models/diffusion/fm.py and samplers.py (verbatim, registry removed) ------------------------------------------------------------
class FM:
    
    def __init__(self, sigma_min=1e-5, timescale=1.0):
        self.sigma_min = sigma_min
        self.prediction_type = None
        self.timescale = timescale
    
    def alpha(self, t):
        return 1.0 - t
    
    def sigma(self, t):
        return self.sigma_min + t * (1.0 - self.sigma_min)
    
    def A(self, t):
        return 1.0
    
    def B(self, t):
        return -(1.0 - self.sigma_min)
    
    def get_betas(self, n_timesteps):
        return torch.zeros(n_timesteps) # Not VP and not supported
    
    def add_noise(self, x, t, noise=None):
        noise = torch.randn_like(x) if noise is None else noise
        s = [x.shape[0]] + [1] * (x.dim() - 1)
        x_t = self.alpha(t).view(*s) * x + self.sigma(t).view(*s) * noise
        return x_t, noise
    
    def loss(self, net, x, t=None, net_kwargs=None, return_loss_unreduced=False, return_all=False):
        if net_kwargs is None:
            net_kwargs = {}
        
        if t is None:
            t = torch.rand(x.shape[0], device=x.device)
        x_t, noise = self.add_noise(x, t)
        
        pred = net(x_t, t=t * self.timescale, **net_kwargs)
        
        target = self.A(t) * x + self.B(t) * noise # -dxt/dt
        if return_loss_unreduced:
            loss = ((pred.float() - target.float()) ** 2).mean(dim=[1, 2, 3])
            if return_all:
                return loss, t, x_t, pred
            else:
                return loss, t
        else:
            loss = ((pred.float() - target.float()) ** 2).mean()
            if return_all:
                return loss, x_t, pred
            else:
                return loss
    
    def get_prediction(
        self,
        net,
        x_t,
        t,
        net_kwargs=None,
        uncond_net_kwargs=None,
        guidance=1.0,
    ):
        if net_kwargs is None:
            net_kwargs = {}
        pred = net(x_t, t=t * self.timescale, **net_kwargs)
        if guidance != 1.0:
            assert uncond_net_kwargs is not None
            uncond_pred = net(x_t, t=t * self.timescale, **uncond_net_kwargs)
            pred = uncond_pred + guidance * (pred - uncond_pred)
        return pred
    
    def convert_sample_prediction(self, x_t, t, pred):
        M = torch.tensor([
            [self.alpha(t), self.sigma(t)],
            [self.A(t), self.B(t)],
        ], dtype=torch.float64)
        M_inv = torch.linalg.inv(M)
        sample_pred = M_inv[0, 0].item() * x_t + M_inv[0, 1].item() * pred
        return sample_pred


class FMEulerSampler:

    def __init__(self, diffusion):
        self.diffusion = diffusion

    def sample(
        self,
        net,
        shape,
        n_steps,
        net_kwargs=None,
        uncond_net_kwargs=None,
        guidance=1.0,
        noise=None,
    ):
        device = next(net.parameters()).device
        x_t = torch.randn(shape, device=device) if noise is None else noise
        t_steps = torch.linspace(1, 0, n_steps + 1, device=device)

        with torch.no_grad():
            for i in range(n_steps):
                t = t_steps[i].repeat(x_t.shape[0])
                neg_v = self.diffusion.get_prediction(
                    net,
                    x_t,
                    t,
                    net_kwargs=net_kwargs,
                    uncond_net_kwargs=uncond_net_kwargs,
                    guidance=guidance,
                )
                x_t = x_t + neg_v * (t_steps[i] - t_steps[i + 1])
        return x_t


# ---- the tokenizer: models/ldm/dito.py + ldm_base.py, for the paper's f8c4 configs ----------------------------------------------
DITO_SIZES = { # encoder ch (f8c4: ch_mult 1, 2, 4, 4), UNet c0 / c1 / c2 (pe_dim 320, t_dim 1280 as the official configs)
    'B': dict(enc_ch=128, c0=128, c1=256, c2=512), # configs/experiments/dito-B-f8c4.yaml
    'L': dict(enc_ch=128, c0=192, c1=384, c2=768), # dito-L-f8c4.yaml
    'S': dict(enc_ch=64, c0=64, c1=128, c2=256),   # not in the paper: half of B's channels
}


class DiTo(nn.Module):
    """
    encode(x): x in [-1, 1] [B, 3, H, W] -> z [B, 4, H/8, W/8], LayerNorm'd over (C, H, W) without affine (0 mean, unit variance
    per image: ldm_base z_layernorm), the latent the generator models. loss(x): the diffusion decoder's flow-matching loss
    (v target (1 - sigma_min) e - x, t ~ U[0, 1], timescale 1000), with noise synchronization: with probability zaug_p the
    decoder sees z noised to a random tz by the same flow (zt = (1 - tz) z + sigma(tz) e) and its t is drawn from [tz, 1]
    ('suffix'), so the decoder learns to decode noisy latents too. decode(z): 50 Euler steps from noise.
    """
    def __init__(self, size='B', img_size=128, zaug_p=0.1, render_n_steps=50):
        super().__init__()
        cfg = DITO_SIZES[size]
        enc_kwargs = dict(default_configs['f8c4'], ch=cfg['enc_ch'])
        self.encoder = nn.Sequential(Encoder(**enc_kwargs), nn.Conv2d(4, 4, 1)) # make_vqgan_encoder('f8c4')
        self.z_shape = (4, img_size // 8, img_size // 8)
        self.z_layernorm = nn.LayerNorm(list(self.z_shape), elementwise_affine=False)
        self.renderer = ConsistencyDecoderUNet(in_channels=3, z_dec_channels=4, c0=cfg['c0'], c1=cfg['c1'], c2=cfg['c2'], pe_dim=320, t_dim=1280)
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
        self.zaug_p, self.render_n_steps, self.size = zaug_p, render_n_steps, size

    def encode(self, x):
        return self.z_layernorm(self.encoder(x))

    def loss(self, x):
        z = self.encode(x)
        t = torch.rand(x.shape[0], device=x.device)
        if self.zaug_p and self.training: # noise synchronization (zaug, decoding loss 'suffix', zaug_tmax 1)
            tz = torch.rand(z.shape[0], device=z.device)
            zt, _ = self.diffusion.add_noise(z, tz)
            mask = (torch.rand(z.shape[0], device=z.device) < self.zaug_p).float()
            z = mask.view(-1, 1, 1, 1) * zt + (1 - mask).view(-1, 1, 1, 1) * z
            t = mask * (tz + (1 - tz) * torch.rand_like(tz)) + (1 - mask) * t
        loss, t = self.diffusion.loss(self.renderer, x, t=t, net_kwargs={'z_dec': z}, return_loss_unreduced=True)
        return loss.mean(), loss.detach(), t

    @torch.no_grad()
    def decode(self, z, noise=None, n_steps=None):
        shape = (z.shape[0], 3, z.shape[2] * 8, z.shape[3] * 8)
        return self.sampler.sample(self.renderer, shape, n_steps or self.render_n_steps, net_kwargs={'z_dec': z}, noise=noise)
