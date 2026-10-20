"""
REPA (Yu et al. 2024) for JiT: align an intermediate JiT block's tokens (on the noisy input) with a
frozen encoder's patch features of the clean image, via an MLP projector and negative cosine similarity.

Targets (repa_encoder in train_jit.py):
    dino_vits16, dino_vitb16, mae_vitb16, ...   any name in vit.PRETRAINED (ImageNet-1k only)
    dinov2_vits14, dinov2_vitb14, ...           torch.hub DINOv2 (LVD-142M)
    file:/path/to/ckpt.pth                      any DINO-format ViT checkpoint (e.g. iBOT IN-1k / IN-22k, DeiT)

The encoder input is resized so its patch grid matches JiT's token grid (e.g. 16x16 for 256px JiT/16:
256px for /16 encoders, 224px for /14 encoders).
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
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name.startswith('dinov2_'):
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
