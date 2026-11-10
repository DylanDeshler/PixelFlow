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

def flow_loss(denoiser, x, labels, masker=None, mask=None, weighting='effective_t', return_aux=False, parts=None, weighter=None,
              backward=None, path_aux=None):
    """
    JiT's training loss (Denoiser.forward: label dropout, logit-normal t, x-prediction, v-loss), with optional token
    masking. With mask=None this is numerically the same computation as denoiser(x, labels).
    return_aux: also return the t (flattened) and the (possibly dropped) labels this forward used, for auxiliary heads
    that need the same conditioning (PixelREPA's adapter).
    Skip pathway with skip_stop_grad, in training: the loss is L(with skip) + L(trunk only), the same objective on both
    predictions (see JiT.forward's return_trunk). parts: optional dict, filled with the two terms for logging.
    weighter: optional loss_weighting.LossWeighting (frequency / learned per-noise-level weighting of the residual).
    backward: two-token-size models only (token_sizes): called on each path's share of the loss right after its forward, so
    that path's activations are freed before the next one runs (peak memory = the larger path, not the sum); the returned
    loss is then detached.
    path_aux: two-token-size models only: a callable returning an auxiliary loss for the forward that just ran (e.g. the
    Dispersive Loss on the blocks that forward's hooks captured), added to each path's loss with that path's share of the
    batch, before its backward; parts['aux'] gets the batch-weighted mean of the callable's values.
    """
    assert weighter is None or mask is None, 'loss weighting is not combined with token masking'
    if getattr(denoiser.net, 'token_sizes', ()) and denoiser.training:
        return _two_size_loss(denoiser, x, labels, return_aux, parts, weighter, mask, backward, path_aux)
    labels = denoiser.drop_labels(labels) if denoiser.training else labels
    t = denoiser.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
    e = torch.randn_like(x) * denoiser.noise_scale
    z = t * x + (1 - t) * perturbed(denoiser, e)
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


def perturbed(denoiser, e):
    """
    Input perturbation (DDPM-IP, Ning et al. ICML 2023, 'Input Perturbation Reduces Exposure Bias in Diffusion Models'), in
    training only: the network's input uses the noise e + gamma * noise_scale * xi (xi a fresh Gaussian) while the target stays
    the clean image, so it learns to map slightly-off inputs, like the ones its own earlier sampling steps produce, back to
    the data instead of following their error. Scaled by (1 - t) with the noise, as in the paper. gamma =
    denoiser.input_perturb (train_jit.py --input_perturb, 0 = off); evals and diagnostics never perturb.
    """
    gamma = getattr(denoiser, 'input_perturb', 0.0)
    if gamma <= 0 or not denoiser.training:
        return e
    return e + gamma * denoiser.noise_scale * torch.randn_like(e)


def _two_size_loss(denoiser, x, labels, return_aux, parts, weighter, mask, backward=None, path_aux=None):
    """
    flow_loss for a two-token-size JiT (token_sizes, model_jit.TokenAllocation): a fixed quota of the batch (fine_frac) goes
    through the fine tokens, drawn without replacement in proportion to the allocation's weights (Gumbel top-k: sampling
    proportional to the gain, not greedy), the rest through the coarse tokens; two forwards of fixed sizes. Each path's
    per-sample losses update the allocation's statistics. The loss is the per-sample mean over the whole batch, as usual.
    backward, path_aux: see flow_loss (each path backpropagated as soon as its loss exists, with its auxiliary loss).
    """
    net = denoiser.net
    assert mask is None and not getattr(net, 'skip_pathway', False), 'token_sizes: no token masking or skip pathway'
    labels = denoiser.drop_labels(labels)
    B = x.size(0)
    t = denoiser.sample_t(B, device=x.device).view(-1, *([1] * (x.ndim - 1)))
    z = t * x + (1 - t) * perturbed(denoiser, torch.randn_like(x) * denoiser.noise_scale)
    k = max(1, min(B - 1, round(net.alloc.fine_frac * B)))
    w = net.alloc.weights(t.flatten())
    gumbel = -torch.log(-torch.log(torch.rand(B, device=x.device).clamp(1e-20, 1.0)))
    fine = torch.zeros(B, dtype=torch.bool, device=x.device)
    fine[torch.topk(torch.log(w.clamp_min(1e-20)) + gumbel, k).indices] = True
    total, aux_total = 0.0, 0.0
    paths = (((~fine).nonzero().squeeze(1), net.token_sizes[0]), (fine.nonzero().squeeze(1), net.token_sizes[1]))
    for path, (sel, p) in enumerate(paths):
        tp = t[sel]
        x_pred = net(z[sel], tp.flatten(), labels[sel], patch=p)
        r = (x[sel] - x_pred) / (1 - tp).clamp_min(denoiser.t_eps) # v - v_pred, with JiT's clamp
        per = r.float().pow(2).mean(dim=(1, 2, 3))
        net.alloc.update(tp.flatten(), per.detach(), path)
        lp = (weighter(r, tp.flatten()) if weighter is not None else per.mean()) * (len(sel) / B)
        if path_aux is not None: # from this forward's activations (hooks), before its backward frees them
            aux = path_aux() * (len(sel) / B)
            aux_total = aux_total + aux.detach()
            lp = lp + aux
        if backward is not None:
            backward(lp)
            lp = lp.detach()
        total = total + lp
    loss = total
    if parts is not None:
        parts['skip'] = loss
        if path_aux is not None:
            parts['aux'] = aux_total
    if return_aux:
        return loss, t.flatten(), labels
    return loss
