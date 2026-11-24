"""
The VAEs behind the latent baselines (prepare_latents.py, train_latent.py, latent_rfid.py, vae_cost.py), with one interface:
load_vae(name, device) returns an eval-mode module with
  encode_mean(x)  images in [-1, 1] [B, 3, H, W] -> the posterior mean [B, C, H/f, W/f] (unscaled)
  decode_img(z)   unscaled latents -> images in [-1, 1] (unclamped)
and the usual parameter names (encoder / quant_conv / decoder / post_quant_conv) for counting encoder and decoder params.
SCALE[name] is the factor latents are multiplied by for the diffusion model (about unit variance), as each VAE's own users set it.

  stabilityai/sd-vae-ft-ema   SD's kl-f8 (4 channels, stride 8): trained on OpenImages, decoder fine-tuned on LAION. Scale 0.18215
  stabilityai/sd-vae-ft-mse   (DiT / SiT / LDM). diffusers.
  mar-kl16                    MAR's KL-16 (16 channels, stride 16), trained on ImageNet-1k only, with LDM's kl-f8 recipe (L1 +
                              LPIPS + PatchGAN, KL weight 1e-6) at batch 256 / lr 1e-4, decoder without attention (Li et al. 2024,
                              Table 2: rFID 1.22 at 256; the author on github.com/LTH14/mar/issues/3). Scale 0.2325 (MAR's
                              hard-coded value). Code vendored in mar_vae.py; the checkpoint (MAR's util/download.py link) is
                              downloaded once to $MAR_KL16_CKPT or ~/.cache/cross_patch/kl16.ckpt.
"""

import os

import torch

SCALE = {'stabilityai/sd-vae-ft-ema': 0.18215, 'stabilityai/sd-vae-ft-mse': 0.18215, 'mar-kl16': 0.2325}
SHORT = {'stabilityai/sd-vae-ft-ema': 'sdvae-ft-ema', 'stabilityai/sd-vae-ft-mse': 'sdvae-ft-mse', 'mar-kl16': 'mar-kl16'} # latent dir names

MAR_KL16_URL = 'https://www.dropbox.com/scl/fi/hhmuvaiacrarfg28qxhwz/kl16.ckpt?rlkey=l44xipsezc8atcffdp4q7mwmh&dl=1'


def _mar_kl16_ckpt():
    path = os.environ.get('MAR_KL16_CKPT') or os.path.expanduser('~/.cache/cross_patch/kl16.ckpt')
    if not os.path.exists(path):
        import urllib.request
        from util.progress import bar
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(MAR_KL16_URL, headers={'User-Agent': 'Wget/1.16 (linux-gnu)'}) # Dropbox serves the file to wget-like clients
        with urllib.request.urlopen(req) as r, open(path + '.tmp', 'wb') as f:
            total = int(r.headers.get('Content-Length') or 0)
            for _ in bar(range(max(1, -(-total // 2 ** 20))), f'downloading MAR KL-16 to {path}'):
                chunk = r.read(2 ** 20)
                if not chunk:
                    break
                f.write(chunk)
            while chunk := r.read(2 ** 20): # if Content-Length was missing
                f.write(chunk)
        os.replace(path + '.tmp', path)
    return path


def load_vae(name, device):
    if name == 'mar-kl16':
        from mar_vae import AutoencoderKL as MarKL
        vae = MarKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4)) # as MAR's main_mar.py builds it
        sd = torch.load(_mar_kl16_ckpt(), map_location='cpu', weights_only=False)['model']
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        assert not missing, f'MAR KL-16 checkpoint is missing {missing}'
        if unexpected:
            print(f'MAR KL-16: ignored {len(unexpected)} checkpoint entries not in the autoencoder (e.g. {unexpected[0]})')
        vae.encode_mean = lambda x: vae.encode(x).mean
        vae.decode_img = lambda z: vae.decode(z)
    elif name in SCALE:
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(name)
        vae.encode_mean = lambda x: vae.encode(x).latent_dist.mean
        vae.decode_img = lambda z: vae.decode(z).sample
    else:
        raise ValueError(f'unknown VAE {name!r}: one of {list(SCALE)}')
    return vae.to(device).eval().requires_grad_(False)
