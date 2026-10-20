"""
Representation alignment for JiT: align an intermediate JiT block's tokens (on the noisy input) with a frozen
encoder's patch features of the clean image.

Adapters (repa_adapter in train_jit.py):
    mlp   REPA (Yu et al. 2024): 3-layer SiLU MLP projector per token, negative cosine similarity. Shin et al. 2026
          ("Representation Alignment for Just Image Transformers is not Easier than You Think") show this hurts JiT
          (JiT-B/16, 200 epochs: FID 5.14 with REPA vs 4.37 without).
    mta   PixelREPA (Shin et al. 2026, https://github.com/kaist-cvml/PixelREPA): a Masked Transformer Adapter, i.e.
          2 JiT blocks with their own time / class conditioning, applied to the JiT tokens after random 20% token
          masking, trained with 1 - cosine similarity at weight 0.1. The adapter, not the JiT features, absorbs the
          mapping into the compressed semantic space.

Targets (repa_encoder in train_jit.py):
    dino_vits16, dino_vitb16, mae_vitb16, ...   any name in vit.PRETRAINED (ImageNet-1k only)
    dinov2_vits14, dinov2_vitb14, ...           torch.hub DINOv2 (LVD-142M), input resized so its 14px grid matches JiT's
    timm:<model>                                a timm model rebuilt at JiT's image and patch size, as PixelREPA does
                                                (timm:vit_base_patch14_dinov2.lvd142m = DINOv2-B with 16px patches at 256px)
    file:/path/to/ckpt.pth                      any DINO-format ViT checkpoint (e.g. iBOT IN-1k / IN-22k, DeiT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit import VisionTransformer, PRETRAINED, load_pretrained

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_vit_file(path):
    """
    Loads a DINO-format ViT checkpoint, inferring the architecture from the weights.
    Handles the usual wrappers ('teacher', 'state_dict', 'model') and prefixes ('module.', 'backbone.').
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('teacher', 'state_dict', 'model'):
        if isinstance(checkpoint, dict) and key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    state_dict = {}
    for k, v in checkpoint.items():
        for prefix in ('module.', 'backbone.'):
            if k.startswith(prefix):
                k = k[len(prefix):]
        state_dict[k] = v
    embed_dim = state_dict['pos_embed'].shape[-1]
    model_args = dict(
        patch_size=state_dict['patch_embed.proj.weight'].shape[-1],
        embed_dim=embed_dim,
        depth=len({k.split('.')[1] for k in state_dict if k.startswith('blocks.')}),
        num_heads={384: 6, 768: 12, 1024: 16}[embed_dim],
    )
    model = VisionTransformer(**model_args)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing, f'missing keys loading {path}: {missing}'
    print(f"loaded {path} as ViT {model_args}, ignored {len(unexpected)} extra keys (heads etc.)")
    return model

class FrozenEncoder(nn.Module):
    def __init__(self, name, img_size=None, patch_size=None):
        super().__init__()
        self.name = name
        if name.startswith('timm:'):
            # PixelREPA's setup: timm rebuilds the encoder at the JiT image size with the JiT patch size (resampling the
            # patch embedding and position embedding), so its token grid is JiT's by construction
            import timm
            self.model = timm.create_model(name[len('timm:'):], pretrained=True, img_size=img_size, patch_size=patch_size)
            self.patch_size = patch_size
            self.kind = 'timm'
        elif name.startswith('dinov2_'):
            self.model = torch.hub.load('facebookresearch/dinov2', name)
            self.patch_size = 14
            self.kind = 'dinov2'
        else:
            self.model = load_vit_file(name[len('file:'):]) if name.startswith('file:') else load_pretrained(name)
            self.patch_size = self.model.patch_embed.patch_size
            self.kind = 'vit'
        self.embed_dim = self.model.embed_dim
        self.model.eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        # always frozen in eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, x, grid):
        """
        x: clean images in [-1, 1], [B, 3, H, W]. Returns patch features [B, grid*grid, D].
        """
        x = ((x + 1) / 2 - self.mean) / self.std
        size = grid * self.patch_size
        if self.kind == 'timm':
            if x.shape[-1] != size: # PixelREPA resizes bilinearly when the sizes differ
                x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
            return self.model.forward_features(x)[:, self.model.num_prefix_tokens:]
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False)
        if self.kind == 'dinov2':
            return self.model.forward_features(x)['x_norm_patchtokens']
        tokens = self.model.get_intermediate_layers(x, 1)[0]
        return tokens[:, self.model.num_prefix_tokens:]

class Projector(nn.Module):
    # same 3-layer SiLU MLP as REPA
    def __init__(self, in_dim, out_dim, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)

class MaskedTransformerAdapter(nn.Module):
    """
    PixelREPA's Masked Transformer Adapter (following model_pixelREPA.py in their repo): JiT blocks with their own
    time / class embeddings and in-context class tokens, over the JiT tokens with a random fraction replaced by a mask
    token, then a projection head to the encoder width (only when the widths differ, as in their code).
    """
    def __init__(self, input_size, patch_size, out_dim, hidden_size, num_heads, num_classes,
                 depth=2, mlp_ratio=4.0, in_context_len=32, mask_ratio=0.2):
        super().__init__()
        from model_jit import JiTBlock, TimestepEmbedder, LabelEmbedder
        from util.model_util import VisionRotaryEmbeddingFast
        self.hidden_size, self.out_dim, self.in_context_len, self.mask_ratio = hidden_size, out_dim, in_context_len, mask_ratio
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.in_context_posemb = nn.Parameter(torch.zeros(1, in_context_len, hidden_size))
        nn.init.normal_(self.in_context_posemb, std=.02)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=in_context_len)
        self.blocks = nn.ModuleList([JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.projection_layer = nn.Sequential( # PixelREPA's ProjectionHead
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU(), nn.Linear(hidden_size, out_dim),
        )
        # in their code the mask token lives in the JiT model; it is only ever used on this adapter path
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, h, t, y):
        """
        h: JiT tokens [N, L, hidden] (no in-context tokens); t: [N]; y: [N] (the same dropped labels as the JiT forward).
        """
        if self.training and self.mask_ratio > 0:
            keep = torch.rand(h.shape[0], h.shape[1], 1, device=h.device) > self.mask_ratio # per-token, as in their code
            h = torch.where(keep, h, self.mask_token.to(h.dtype))
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        x = torch.cat([y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1) + self.in_context_posemb, h], dim=1)
        for block in self.blocks:
            x = block(x, c, self.feat_rope_incontext)
        x = x[:, self.in_context_len:]
        if self.hidden_size != self.out_dim:
            x = self.projection_layer(x)
        return x

def mta_loss(hidden, target, adapter, t, y):
    # PixelREPA: 1 - cosine similarity between adapter outputs and the encoder's patch features, averaged over tokens
    pred = F.normalize(adapter(hidden, t, y).float(), dim=-1)
    return (1 - (pred * F.normalize(target.float(), dim=-1)).sum(-1)).mean()

def repa_loss(hidden, target, projector):
    # mean negative cosine similarity over tokens and batch
    z = F.normalize(projector(hidden).float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return -(z * target).sum(-1).mean()

class BlockCapture:
    """
    Forward hook that stores the output tokens of one JiT block (dropping in-context class tokens if present).
    """
    def __init__(self, net, depth):
        self.block_idx = depth - 1
        self.n_strip = net.in_context_len if (net.in_context_len > 0 and self.block_idx >= net.in_context_start) else 0
        self.out = None
        net.blocks[self.block_idx].register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        self.out = output[:, self.n_strip:]
