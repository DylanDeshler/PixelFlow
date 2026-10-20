"""
Baseline ViT + DINO head/loss.

The VisionTransformer here mirrors facebookresearch/dino's vision_transformer.py
(same module names), so official DINO checkpoints load directly into it. That
lets you sanity check the eval pipeline against published numbers before
trusting it on your own encoders.

Encoder interface (anything you swap in for VisionTransformer needs this):
    embed_dim                       int, width of the token features
    num_prefix_tokens               int, tokens before the patch tokens (cls, registers, ...)
    forward(x) -> [B, D]            global embedding fed to the DINO head and k-NN eval
    get_intermediate_layers(x, n)   list of the last n blocks' normed outputs, each [B, num_prefix + N, D],
                                    with token 0 being the global/cls token (used by the linear probe)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=384,
                 depth=12,
                 num_heads=6,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prefix_tokens = 1
        norm_layer = lambda dim: nn.LayerNorm(dim, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def interpolate_pos_encoding(self, x, h, w):
        # lets the 224px-trained position embeddings work on 96px local crops (and other eval sizes)
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, :1]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        h0 = h // self.patch_embed.patch_size
        w0 = w // self.patch_embed.patch_size
        M = int(math.sqrt(N))
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode='bicubic',
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, H, W)
        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

# official ImageNet-1k self-supervised checkpoints that load directly into VisionTransformer:
# name -> (model_args, url, key of the state dict inside the checkpoint or None)
PRETRAINED = {
    'dino_vits16': (dict(patch_size=16, embed_dim=384, depth=12, num_heads=6), 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth', None),
    'dino_vits8': (dict(patch_size=8, embed_dim=384, depth=12, num_heads=6), 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth', None),
    'dino_vitb16': (dict(patch_size=16, embed_dim=768, depth=12, num_heads=12), 'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth', None),
    'dino_vitb8': (dict(patch_size=8, embed_dim=768, depth=12, num_heads=12), 'https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth', None),
    'mae_vitb16': (dict(patch_size=16, embed_dim=768, depth=12, num_heads=12), 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth', 'model'),
}

def load_pretrained(name):
    model_args, url, key = PRETRAINED[name]
    model = VisionTransformer(**model_args)
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    if key is not None:
        state_dict = state_dict[key]
    model.load_state_dict(state_dict, strict=True)
    return model

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        # weight-normed last layer, written out by hand instead of nn.utils.weight_norm (deprecated + awkward under compile)
        # with norm_last_layer=True the magnitude g is frozen at 1, which DINO found more stable
        last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.last_layer_v = nn.Parameter(last_layer.weight.data.clone())
        self.last_layer_g = nn.Parameter(torch.ones(out_dim, 1), requires_grad=not norm_last_layer)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        w = self.last_layer_g * F.normalize(self.last_layer_v, dim=1)
        return F.linear(x, w)

class MultiCropWrapper(nn.Module):
    """
    Runs the backbone once per crop resolution (all global crops batched together,
    then all local crops), then the head once on everything.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, crops):
        if not isinstance(crops, (list, tuple)):
            crops = [crops]
        # group consecutive crops with the same resolution
        groups = [[crops[0]]]
        for c in crops[1:]:
            if c.shape[-1] == groups[-1][-1].shape[-1]:
                groups[-1].append(c)
            else:
                groups.append([c])
        outputs = [self.backbone(torch.cat(g)) for g in groups]
        return self.head(torch.cat(outputs))

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output, teacher_temp):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_output: [ncrops * B, out_dim], teacher_output: [2 * B, out_dim] (global crops only)
        """
        student_out = F.log_softmax(student_output.float() / self.student_temp, dim=-1).chunk(self.ncrops)
        teacher_out = F.softmax((teacher_output.float() - self.center) / teacher_temp, dim=-1).detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # skip cases where student and teacher operate on the same view
                    continue
                total_loss += torch.sum(-q * student_out[v], dim=-1).mean()
                n_loss_terms += 1
        return total_loss / n_loss_terms

    @torch.no_grad()
    def update_center(self, batch_center):
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)
