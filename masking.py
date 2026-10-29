"""
MAE-style token masking for JiT training, with no extra loss term.

A fraction of the patch tokens is replaced by a learned mask token right after the patch embedding (JiT then adds the
position embedding as usual, so masked tokens still know where they are). The objective is unchanged: the same
flow-matching loss on every patch, so for masked patches the model has to predict the clean patch from the other
patches only. Masking is a training-time input corruption; sampling and evaluation never mask.

Loss weighting: JiT's v-loss is (x - x_pred)^2 / (1 - t)^2 per pixel, up to 400x near clean images. A masked patch
carries no information about itself, i.e. it is at noise level t=0 regardless of the sampled t, so with
weighting='effective_t' its error gets the t=0 weight (1). weighting='jit' keeps the sampled-t weight for every patch,
which lets the irreducible inpainting error of masked patches dominate at low noise.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn


class TokenMasker(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, net.hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)
        self.patch_size = net.patch_size
        self.num_patches = net.x_embedder.num_patches
        self.mask = None # [B, N] bool while active, else None
        net.x_embedder.register_forward_hook(self._hook)
        if getattr(net, 'skip_pathway', False): # masked patches must not reach the output through the skip pathway
            net.skip_embedder.register_forward_hook(self._skip_hook)
        # a DC-AE stem or overlapping patches let a token see its neighbours' pixels: blank masked patches in the image
        # itself, before either sees it, so their content can't leak into the unmasked tokens next to them
        if getattr(net, 'dcae_stem', False):
            net.input_stem.register_forward_pre_hook(self._image_pre_hook)
        elif getattr(net, 'patch_overlap', 0):
            net.x_embedder.register_forward_pre_hook(self._image_pre_hook)

    def _hook(self, module, inputs, output):
        if self.mask is None:
            return output
        return torch.where(self.mask.unsqueeze(-1), self.mask_token.to(output.dtype), output)

    def _image_pre_hook(self, module, args):
        if self.mask is None:
            return None
        img, *rest = args
        return (img * (1 - self.pixel_mask(self.mask).to(img.dtype)), *rest)

    def _skip_hook(self, module, inputs, output):
        if self.mask is None:
            return output
        return output.masked_fill(self.mask.unsqueeze(-1), 0)

    def sample(self, batch_size, ratio, prob, device):
        """
        For a random `prob` fraction of the batch, masks round(ratio * N) uniformly random patches; other samples are unmasked.
        """
        n_mask = int(round(ratio * self.num_patches))
        ranks = torch.rand(batch_size, self.num_patches, device=device).argsort(dim=1).argsort(dim=1)
        mask = ranks < n_mask
        mask &= (torch.rand(batch_size, 1, device=device) < prob)
        return mask

    @contextmanager
    def active(self, mask):
        self.mask = mask
        try:
            yield
        finally:
            self.mask = None

    def pixel_mask(self, mask):
        # [B, N] patch mask -> [B, 1, H, W] pixel mask (patches are row-major, as in JiT's patch embedding)
        g = int(round(self.num_patches ** 0.5))
        m = mask.view(-1, 1, g, g).float()
        return m.repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3)

def flow_loss(denoiser, x, labels, masker=None, mask=None, weighting='effective_t', return_aux=False, parts=None, weighter=None):
    """
    JiT's training loss (Denoiser.forward: label dropout, logit-normal t, x-prediction, v-loss), with optional token
    masking. With mask=None this is numerically the same computation as denoiser(x, labels).
    return_aux: also return the t (flattened) and the (possibly dropped) labels this forward used, for auxiliary heads
    that need the same conditioning (PixelREPA's adapter).
    Skip pathway with skip_stop_grad, in training: the loss is L(with skip) + L(trunk only), the same objective on both
    predictions (see JiT.forward's return_trunk). parts: optional dict, filled with the two terms for logging.
    weighter: optional loss_weighting.LossWeighting (frequency / learned per-noise-level weighting of the residual).
    """
    assert weighter is None or mask is None, 'loss weighting is not combined with token masking'
    labels = denoiser.drop_labels(labels) if denoiser.training else labels
    t = denoiser.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
    e = torch.randn_like(x) * denoiser.noise_scale
    z = t * x + (1 - t) * e
    two = denoiser.training and getattr(denoiser.net, 'skip_stop_grad', False)
    kwargs = {'return_trunk': True} if two else {}
    if mask is None:
        x_pred = denoiser.net(z, t.flatten(), labels, **kwargs)
    else:
        with masker.active(mask):
            x_pred = denoiser.net(z, t.flatten(), labels, **kwargs)

    def v_loss(x_pred):
        # v - v_pred = (x - x_pred) / (1 - t), with JiT's clamp
        if weighter is not None:
            return weighter((x - x_pred) / (1 - t).clamp_min(denoiser.t_eps), t.flatten())
        err = ((x - x_pred) / (1 - t).clamp_min(denoiser.t_eps)) ** 2
        if mask is not None and weighting == 'effective_t':
            pm = masker.pixel_mask(mask)
            err = err * (1 - pm) + (x - x_pred) ** 2 * pm # masked patches weighted as t=0
        return err.mean()

    if two:
        loss_skip, loss_trunk = v_loss(x_pred[0]), v_loss(x_pred[1])
        loss = loss_skip + loss_trunk
        if parts is not None:
            parts['skip'], parts['trunk'] = loss_skip, loss_trunk
    else:
        loss = v_loss(x_pred)
        if parts is not None:
            parts['skip'] = loss
    if return_aux:
        return loss, t.flatten(), labels
    return loss
