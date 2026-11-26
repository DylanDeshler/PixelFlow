"""
Training compute of the pretrained VAEs (latent_vae.py) that latent baselines (train_latent.py) depend on, so a latent pipeline's training
FLOPs = VAE training + trunk training, comparable with an end-to-end pixel model's.

Each VAE's training history is listed as stages with steps, batch size, crop resolution, which parts were trained and the
losses, from its model card / release config (sources below). The FLOPs per training image are measured with torch's FLOP
counter (FLOPs = 2 x multiply-adds) on the actual VAE at the stage's resolution, plus the loss networks:
  trained encoder + decoder   3 x (encoder + decoder forward)                 (forward + backward ~2x forward)
  decoder only                encoder forward + 3 x decoder forward          (frozen encoder: forward only)
  LPIPS (VGG16 to relu5_3)    3 x VGG forward: target + reconstruction forward, backward to the input (frozen weights:
                              no weight gradients, ~1x forward)
  PatchGAN discriminator      from disc_start on: generator term (forward + input backward, 2x) and the discriminator's own
                              step on real + fake (2 x 3x forward) = 8 x forward
VGG16 / PatchGAN forward at 256px measured once (torchvision vgg16 features[:30]; taming's NLayerDiscriminator, ndf 64,
3 layers): 40.09 / 6.29 GFLOPs, scaled with pixel count. Optimizer, EMA and KL terms are negligible and not counted.

Gaps, stated in the output: the original kl-f8 run's GPU count is not published (the release config gives 4 images per GPU;
counted as 1 GPU, a lower bound), and the fine-tuning stages' resolution isn't stated (256, as the original's crops and the
card's evaluation). Whether fine-tuning kept the adversarial term isn't stated ("L1 + LPIPS"); not counted (lower bound).

Sources:
  https://huggingface.co/stabilityai/sd-vae-ft-ema   (ft-EMA: 313,198 steps; ft-MSE: +280,000 steps from ft-EMA; batch 192 =
                                                      16 A100 x 12; decoder only; original: 246,803 steps)
  https://huggingface.co/stabilityai/sd-vae-ft-mse
  https://github.com/CompVis/latent-diffusion/blob/main/models/first_stage_models/kl-f8/config.yaml
                                                     (kl-f8: batch_size 4 per GPU, 256 crops of OpenImages, LPIPS + PatchGAN
                                                      discriminator from step 50,001)
"""

VGG16_GF_256 = 40.089157632
DISC_GF_256 = 6.293618688

_ORIGINAL_KLF8 = dict(name='kl-f8 original (CompVis, OpenImages)', steps=246803, batch=4, res=256, trains='encoder+decoder',
                      lpips=True, disc_start=50001, note='batch = 4 per GPU from the release config; GPU count unpublished, counted as 1 (lower bound)')
_FT_EMA = dict(name='ft-EMA (LAION-Aesthetics + LAION-Humans)', steps=313198, batch=192, res=256, trains='decoder',
               lpips=True, disc_start=None, note='resolution not stated (256 assumed); adversarial term not counted')
_FT_MSE = dict(name='ft-MSE (from ft-EMA, MSE + 0.1 LPIPS)', steps=280000, batch=192, res=256, trains='decoder',
               lpips=True, disc_start=None, note='resolution not stated (256 assumed)')

# MAR's KL-16 (ImageNet-1k only): LDM's kl-f8 recipe (LPIPS + PatchGAN from step 50,001) at batch 256 / lr 1e-4 (the author,
# github.com/LTH14/mar/issues/3); the number of steps was never published, so only the cost per training image is known and the
# total is not counted (vae_train_complete False: per-FLOP charts then understate this pipeline's training compute)
_MAR_KL16 = dict(name='MAR KL-16 (ImageNet-1k)', steps=None, batch=256, res=256, trains='encoder+decoder', lpips=True,
                 disc_start=50001, note='steps unpublished: total not counted (per-image cost only, without the discriminator)')

VAE_TRAINING = {
    'stabilityai/sd-vae-ft-ema': [_ORIGINAL_KLF8, _FT_EMA],
    'stabilityai/sd-vae-ft-mse': [_ORIGINAL_KLF8, _FT_EMA, _FT_MSE],
    'mar-kl16': [_MAR_KL16],
}

def vae_training_cost(vae, name, device):
    """
    {'vae_train_pflops': total, 'vae_train_complete': whether every stage's steps are known, 'vae_train_stages': [per stage:
    name, steps, batch, res, gflops_per_image, pflops, note]} or None for a VAE not in the table. A stage with unknown steps
    reports its per-image cost (without the discriminator) and pflops None, and adds nothing to the total. vae: a
    latent_vae.load_vae module (its forward FLOPs are measured per stage resolution).
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    if getattr(vae, 'dito_record', None) is not None: # our own DiTo: its measured training compute travels in the checkpoint
        r = vae.dito_record
        return {'vae_train_pflops': r['train_pflops'], 'vae_train_complete': True,
                'vae_train_stages': [dict(name=f"DiTo-{r['size']} (train_dito.py, {r['train_images']:,} images)", steps=r['iter_num'],
                                          batch=None, res=None, trains='encoder+decoder', gflops_per_image=r['train_gflops_per_image'],
                                          pflops=r['train_pflops'], note='measured (FLOP counter) x updates x batch')]}
    if name not in VAE_TRAINING:
        return None
    def flops(fn):
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            fn()
        return fc.get_total_flops() / 1e9
    stages, total, complete = [], 0.0, True
    measured = {}
    for st in VAE_TRAINING[name]:
        r = st['res']
        if r not in measured:
            x = torch.randn(1, 3, r, r, device=device)
            with torch.no_grad():
                z = vae.encode_mean(x)
            measured[r] = (flops(lambda: vae.encode_mean(x)), flops(lambda: vae.decode_img(z)))
        enc, dec = measured[r]
        scale = (r / 256) ** 2
        per = 3 * (enc + dec) if st['trains'] == 'encoder+decoder' else enc + 3 * dec
        if st['lpips']:
            per += 3 * VGG16_GF_256 * scale
        if st['steps'] is None:
            complete = False
            stages.append(dict(name=st['name'], steps=None, batch=st['batch'], res=r, trains=st['trains'],
                               gflops_per_image=per, pflops=None, note=st['note']))
            continue
        steps_gan = max(0, st['steps'] - st['disc_start'] + 1) if st['disc_start'] else 0
        gf_total = st['steps'] * st['batch'] * per + steps_gan * st['batch'] * 8 * DISC_GF_256 * scale
        pf = gf_total / 1e6
        total += pf
        stages.append(dict(name=st['name'], steps=st['steps'], batch=st['batch'], res=r, trains=st['trains'],
                           gflops_per_image=gf_total / (st['steps'] * st['batch']), pflops=pf, note=st['note']))
    return {'vae_train_pflops': total, 'vae_train_complete': complete, 'vae_train_stages': stages}
