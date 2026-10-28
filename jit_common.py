"""
Shared pieces for train_jit.py / eval_jit.py: building JiT's Denoiser from a flat config,
EMA weight swapping, and FID evaluation of generated samples.
"""

import os
from types import SimpleNamespace
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm

from denoiser import Denoiser
from fid import to_uint8
import model_jit
from model_pixeldit import PixelDiT, PIXELDIT_PRESETS
from model_quadtree import QuadtreeJiT, QUADTREE_PRESETS
from model_gauss import GaussJiT, GAUSS_PRESETS

# architecture of each JiT preset, as defined in model_jit.py (mlp_ratio is the JiT default for all of them)
JIT_PRESETS = {
    'JiT-B/16': dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16),
    'JiT-B/32': dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32),
    'JiT-L/16': dict(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16),
    'JiT-L/32': dict(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32),
    'JiT-H/16': dict(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16),
    'JiT-H/32': dict(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32),
}
# PixelDiT (model_pixeldit.py) shares the denoiser and training stack; registered under its own names
for _name, _arch in PIXELDIT_PRESETS.items():
    JIT_PRESETS[_name] = dict(_arch, in_context_len=0, in_context_start=0, family='pixeldit')
    model_jit.JiT_models[_name] = (lambda arch: lambda **kwargs: PixelDiT(**arch, **kwargs))(JIT_PRESETS[_name])
# fixed-budget quadtree tokenization (model_quadtree.py): same stack, its own network class
for _name, _arch in QUADTREE_PRESETS.items():
    JIT_PRESETS[_name] = dict(_arch, in_context_len=0, in_context_start=0, family='quadtree')
    model_jit.JiT_models[_name] = (lambda arch: lambda **kwargs: QuadtreeJiT(**arch, **kwargs))(JIT_PRESETS[_name])
# Gaussian-placed tokens (model_gauss.py): same stack, its own network class
for _name, _arch in GAUSS_PRESETS.items():
    JIT_PRESETS[_name] = dict(_arch, in_context_len=0, in_context_start=0, family='gauss')
    model_jit.JiT_models[_name] = (lambda arch: lambda **kwargs: GaussJiT(**arch, **kwargs))(JIT_PRESETS[_name])
for _name, _arch in JIT_PRESETS.items():
    assert _name in model_jit.JiT_models, f'{_name} missing from model_jit.JiT_models'

QUADTREE_DEFAULTS = dict(attn_res='none', adaln_single=False, router='uniform', route_baseline='', route_dino='dinov2_vitb14_reg',
                         route_t_min=0.0, route_gate_tau=1.0, pos_embed='integrated', rope='plain', area_bias=True, level_embed=True)

GAUSS_DEFAULTS = dict(attn_res='none', adaln_single=False, extent=0.75, basis=0, max_offset=0.25, scale_range=1.33,
                      max_rotation=0.0, sigma=0.3, placer_width=64, placer_depth=2, area_bias=True, shape_embed=True, fused='auto')

def resolve_arch(model, **overrides):
    """
    Preset architecture with any overrides applied (None means keep the preset value).
    Returns (name, arch): name is the preset itself if nothing changed, otherwise a descriptive custom name.
    """
    arch = dict(JIT_PRESETS[model], mlp_ratio=4.0)
    changed = {k: v for k, v in overrides.items() if v is not None and arch.get(k) != v}
    arch.update(changed)
    assert arch['hidden_size'] % arch['num_heads'] == 0, 'hidden_size must be divisible by num_heads'
    assert (arch['hidden_size'] // arch['num_heads']) % 4 == 0, 'head dim must be divisible by 4 (2D RoPE)'
    assert arch['in_context_len'] == 0 or arch['in_context_start'] < arch['depth'], 'in_context_start must be < depth'
    if not changed:
        return model, arch
    name = model + '(' + ','.join(f'{k}={v}' for k, v in sorted(changed.items())) + ')'
    return name, arch

def describe_arch(model, arch):
    """
    One line with every architecture field as actually built, marking the ones that differ from the preset, e.g.
    'arch: depth 12, hidden_size 768, ..., in_context_len 0* (preset 32), ... (preset JiT-B/16, * = overridden)'.
    """
    if JIT_PRESETS[model].get('family') == 'quadtree':
        preset = dict(QUADTREE_DEFAULTS, **JIT_PRESETS[model], mlp_ratio=4.0)
        full = dict(preset, **arch)
        keys = ('depth', 'hidden_size', 'num_heads', 'patch_size', 'levels', 'tokens', 'bottleneck_dim', 'mlp_ratio', 'attn_res', 'adaln_single',
                'router', 'route_t_min', 'route_gate_tau', 'pos_embed', 'rope', 'area_bias', 'level_embed')
        parts = [(f"{k} {full[k]:g}" if isinstance(full[k], float) else f"{k} {full[k]}") + (f"* (preset {preset[k]})" if full[k] != preset[k] else '')
                 for k in keys]
        if full['router'] in ('dino', 'pixel'):
            parts.append(f"route_baseline {full['route_baseline']}" + (f", route_dino {full['route_dino']}" if full['router'] == 'dino' else ''))
        return f"arch: {', '.join(parts)} (preset {model}, * = overridden)"
    if JIT_PRESETS[model].get('family') == 'gauss':
        preset = dict(GAUSS_DEFAULTS, **JIT_PRESETS[model], mlp_ratio=4.0)
        full = dict(preset, **arch)
        keys = ('depth', 'hidden_size', 'num_heads', 'patch_size', 'bottleneck_dim', 'mlp_ratio', 'attn_res', 'adaln_single', 'extent',
                'basis', 'max_offset', 'scale_range', 'max_rotation', 'sigma', 'placer_width', 'placer_depth', 'area_bias', 'shape_embed', 'fused')
        parts = [(f"{k} {full[k]:g}" if isinstance(full[k], float) else f"{k} {full[k]}") + (f"* (preset {preset[k]})" if full[k] != preset[k] else '')
                 for k in keys]
        return f"arch: {', '.join(parts)} (preset {model}, * = overridden)"
    if JIT_PRESETS[model].get('family') == 'pixeldit':
        preset = dict(JIT_PRESETS[model], mlp_ratio=4.0, prediction='x')
        full = dict(preset, **arch)
        parts = [f"{k} {full[k]:g}" if isinstance(full[k], float) else f"{k} {full[k]}" for k in
                 ('depth', 'pixel_depth', 'hidden_size', 'num_heads', 'pixel_dim', 'patch_size', 'mlp_ratio', 'prediction')]
        parts = [p_ + (f"* (preset {preset[k]})" if full[k] != preset[k] else '') for p_, k in
                 zip(parts, ('depth', 'pixel_depth', 'hidden_size', 'num_heads', 'pixel_dim', 'patch_size', 'mlp_ratio', 'prediction'))]
        return f"arch: {', '.join(parts)} (preset {model}, * = overridden)"
    preset = dict(JIT_PRESETS[model], mlp_ratio=4.0, attn_res='none', attn_res_block_size=0, attn_res_adaln=False, adaln_single=False, gated_bottleneck=False,
                  skip_pathway=False, skip_head_blocks=0, skip_inject_block=0, dcae_stem=False, dcae_head=False, dcae_width=128,
                  dcae_kernel2=1, patch_overlap=0, head_overlap=0, layernorm=False, skip_final_layer=False, skip_residual=False, dcae_trunk_head=False, gate_init='zero', dcae_norm='channel',
                  skip_decoder='linear', skip_decoder_dim=0, skip_decoder_blocks=2, skip_decoder_rank=0, skip_decoder_conv=False, skip_decoder_attnres=False,
                  skip_decoder_modulation=True)
    full = dict(preset, **arch)
    parts = []
    for k in ('depth', 'hidden_size', 'num_heads', 'patch_size', 'layernorm', 'bottleneck_dim', 'gated_bottleneck', 'patch_overlap', 'head_overlap', 'dcae_stem', 'dcae_head', 'dcae_trunk_head', 'dcae_norm', 'dcae_width', 'dcae_kernel2',
              'skip_pathway', 'skip_decoder', 'skip_decoder_dim', 'skip_decoder_blocks', 'skip_decoder_rank', 'skip_decoder_conv', 'skip_decoder_attnres',
              'skip_decoder_modulation', 'skip_head_blocks', 'skip_inject_block', 'skip_final_layer', 'skip_residual', 'mlp_ratio', 'in_context_len', 'in_context_start',
              'attn_res', 'attn_res_block_size', 'attn_res_adaln', 'adaln_single', 'gate_init'):
        v = full[k]
        if k == 'in_context_start' and full['in_context_len'] == 0:
            continue # unused without in-context tokens
        if (k == 'attn_res_block_size' and full['attn_res'] != 'block') or (k == 'attn_res_adaln' and full['attn_res'] == 'none') \
                or (k in ('skip_head_blocks', 'skip_inject_block', 'skip_final_layer', 'skip_residual', 'skip_decoder') and not full['skip_pathway']) \
                or (k.startswith('skip_decoder_') and full['skip_decoder'] == 'linear') \
                or (k in ('dcae_width', 'dcae_kernel2', 'dcae_norm') and not (full['dcae_stem'] or full['dcae_head'] or full['dcae_trunk_head'])) \
                or (k == 'dcae_trunk_head' and not full['skip_final_layer']):
            continue
        if k == 'skip_inject_block': # resolved as in model_jit.JiT
            parts.append(f"{k} {v}{'*' if v else ''} (after block {model_jit.resolve_skip_inject(v, full['depth'])} of {full['depth']})")
            continue
        if k == 'attn_res_block_size' and v == 0: # resolved as in model_jit.JiT
            parts.append(f"{k} {max(1, round(full['depth'] / 8))} (auto)")
            continue
        parts.append(f"{k} {v:g}" if isinstance(v, float) else f"{k} {v}")
        if v != preset[k]:
            parts[-1] += f"* (preset {preset[k]})"
    return f"arch: {', '.join(parts)} (preset {model}, * = overridden)"

def _register(name, arch):
    # JiT's Denoiser looks the network up in model_jit.JiT_models by name, so custom architectures get registered there
    if name not in model_jit.JiT_models:
        cls = {'pixeldit': PixelDiT, 'quadtree': QuadtreeJiT, 'gauss': GaussJiT}.get(arch.get('family'), model_jit.JiT)
        model_jit.JiT_models[name] = lambda **kwargs: cls(**arch, **kwargs)


class FastDenoiser(Denoiser):
    """
    JiT's Denoiser with a faster sampling step, same outputs:
      - cfg == 1, or t outside the CFG interval: only the conditional forward (the guided velocity equals it there)
      - otherwise the conditional and unconditional forwards run as one batch of 2B instead of two sequential passes
    """
    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        low, high = self.cfg_interval
        # same interval rule as JiT's Denoiser._forward_sample
        in_interval = (t < high) & ((low == 0) | (t > low))
        # contiguous: the timesteps are an expand (stride 0 over the batch), and a compiled network guards on strides, so
        # the conditional-only and the batched-CFG calls would otherwise need separate graphs
        t_flat = t.flatten().contiguous()
        # outside the guidance interval only the conditional forward is needed, unless every call must keep one batch
        # shape (fixed_shape, for a single compiled graph): then the batched call runs with scale 1 there
        if self.cfg_scale == 1.0 or (not self.fixed_shape and not bool(in_interval.any())):
            x_cond = self.net(z, t_flat, labels)
            return (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        x_both = self.net(torch.cat([z, z]), torch.cat([t_flat, t_flat]),
                          torch.cat([labels, torch.full_like(labels, self.num_classes)]))
        x_cond, x_uncond = x_both.chunk(2)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)
        scale = torch.where(in_interval, self.cfg_scale, 1.0)
        return v_uncond + scale * (v_cond - v_uncond)

    time_schedule = 'uniform'
    fixed_shape = False # set by generate_and_score(fixed_shape=True)

    def sampling_timesteps(self, device):
        """
        The steps+1 sampling times from 0 (noise) to 1 (clean).
          uniform         JiT's grid, linspace(0, 1)
          shift           FLUX / SD3-style: the uniform grid shifted by the training logit-normal mean,
                          logit(t') = logit(t) + P_mean (SD3's shift map with s = exp(-P_mean))
          train_quantile  quantiles of the training distribution, logit(t) = P_mean + P_std * Phi^-1(u)
        """
        u = torch.linspace(0.0, 1.0, self.steps + 1, device=device, dtype=torch.float64)
        if self.time_schedule == 'uniform':
            t = u
        elif self.time_schedule == 'shift':
            t = torch.sigmoid(torch.logit(u.clamp(1e-12, 1 - 1e-12)) + self.P_mean)
        elif self.time_schedule == 'train_quantile':
            t = torch.sigmoid(self.P_mean + self.P_std * torch.special.ndtri(u.clamp(1e-12, 1 - 1e-12)))
        else:
            raise ValueError(f"unknown time_schedule {self.time_schedule}")
        t[0], t[-1] = 0.0, 1.0
        return t.float()

    eta = 1.0 # method='ancestral': 1 = DDPM's ancestral sampler, 0 = deterministic (see _ancestral_step)

    @torch.no_grad()
    def _ancestral_step(self, z, t, t_next, labels):
        """
        Stochastic step: DDIM's eta family (Song et al. 2021) on JiT's path z = t x + s e, s = (1 - t) noise_scale.
        From the (guided) clean estimate x_hat and the noise it implies, e_hat = (z - t x_hat) / s:
            z_next = t_next x_hat + sqrt(s_next^2 - c^2) e_hat + c * fresh noise,  c = eta * s_next * s_{t|t_next} / s
        with s_{t|t_next}^2 = s^2 - (t / t_next)^2 s_next^2 the noise the forward process adds from t_next to t, so c is eta
        times the std of z_next given z and x_hat. eta = 1 is DDPM's ancestral sampler (the reverse SDE in the many-step
        limit); eta = 0 is deterministic DDIM, the same step as euler except above t = 1 - t_eps, where JiT's clamped
        velocity shortens euler's step. In theory its sample error is bounded by the prediction's L2 error, which an ODE's
        isn't (Song et al. 2021), but it doesn't undo a prediction that keeps too much of its input: that keeps the fresh
        noise too (diagnostics.input_sensitivity's Tweedie ratio measures it).
        """
        v = self._forward_sample(z, t, labels)
        x_hat = z + (1.0 - t).clamp_min(self.t_eps) * v # inverts the clamp: the network's (CFG-combined) x prediction
        s, s_next = (1.0 - t) * self.noise_scale, (1.0 - t_next) * self.noise_scale
        e_hat = (z - t * x_hat) / s
        s_fwd = (s ** 2 - (t / t_next) ** 2 * s_next ** 2).clamp_min(0).sqrt()
        c = (self.eta * s_next * s_fwd / s).clamp(max=s_next)
        return t_next * x_hat + (s_next ** 2 - c ** 2).clamp_min(0).sqrt() * e_hat + c * torch.randn_like(z)

    @torch.no_grad()
    def generate(self, labels):
        # JiT's Denoiser.generate with a configurable time grid (identical to JiT's for time_schedule='uniform') and an
        # optional stochastic sampler
        z = self.noise_scale * torch.randn(labels.size(0), 3, self.img_size, self.img_size, device=labels.device)
        return self._integrate(z, self.sampling_timesteps(labels.device), labels)

    @torch.no_grad()
    def sample_from(self, z, t0, labels):
        """
        Finish sampling from z at time t0 (e.g. a held-out image noised to t0): the sampling grid's times after t0, with
        the same method, so the steps are the ones generate takes from there.
        """
        ts = self.sampling_timesteps(z.device)
        return self._integrate(z, torch.cat([ts.new_tensor([t0]), ts[ts > t0 + 1e-6]]), labels)

    def _integrate(self, z, ts, labels):
        # steps along the 1-D grid ts (start time ... 1); the last step is euler for every method, as in JiT
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        elif self.method == "ancestral":
            stepper = self._ancestral_step
        else:
            raise NotImplementedError(f"unknown sampling method {self.method}: 'euler', 'heun' or 'ancestral'")
        timesteps = ts.view(-1, *([1] * z.ndim)).expand(-1, z.size(0), -1, -1, -1)
        hooks = hasattr(self.net, 'begin_sampling') # quadtree: layouts may be reused for route_every calls of one trajectory
        if hooks:
            self.net.begin_sampling()
        try:
            for i in range(len(ts) - 2):
                z = stepper(z, timesteps[i], timesteps[i + 1], labels)
            return self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        finally:
            if hooks:
                self.net.end_sampling()

    ema_warmup = False

    @torch.no_grad()
    def update_ema(self, step=None):
        """
        JiT's EMA update, optionally with warmup: decay = min(decay, (1 + step) / (10 + step)). JiT applies the full decay
        from step 0, so EMA1 (0.9999) is still 37% random init after 10k iterations; warmup removes that early lag and
        converges to JiT's update (the cap reaches 0.9999 at step ~90k, 0.9996 at ~22k).
        """
        d1, d2 = self.ema_decay1, self.ema_decay2
        if self.ema_warmup and step is not None:
            cap = (1 + step) / (10 + step)
            d1, d2 = min(d1, cap), min(d2, cap)
        params = list(self.parameters())
        torch._foreach_lerp_(self.ema_params1, params, 1 - d1) # ema = d * ema + (1 - d) * param
        torch._foreach_lerp_(self.ema_params2, params, 1 - d2)

def build_denoiser(model, img_size, num_classes, attn_dropout=0.0, proj_dropout=0.0, label_drop_prob=0.1,
                   P_mean=-0.8, P_std=0.8, t_eps=5e-2, noise_scale=1.0, ema_decay1=0.9999, ema_decay2=0.9996,
                   sampling_method='heun', num_sampling_steps=50, cfg=1.0, interval_min=0.0, interval_max=1.0, arch=None):
    """
    arch: full architecture dict (from resolve_arch) for non-preset models; stored in checkpoints so eval can rebuild them.
    """
    if arch is not None:
        _register(model, arch)
    args = SimpleNamespace(
        model=model, img_size=img_size, class_num=num_classes, attn_dropout=attn_dropout, proj_dropout=proj_dropout,
        label_drop_prob=label_drop_prob, P_mean=P_mean, P_std=P_std, t_eps=t_eps, noise_scale=noise_scale,
        ema_decay1=ema_decay1, ema_decay2=ema_decay2, sampling_method=sampling_method,
        num_sampling_steps=num_sampling_steps, cfg=cfg, interval_min=interval_min, interval_max=interval_max,
    )
    return FastDenoiser(args)

def load_denoiser(ckpt, ema=1, device='cpu'):
    """
    Denoiser from a train_jit.py checkpoint (ckpt.pt or a milestone ckpt_{iter}.pt) or a released JiT .pth, with the
    weights ema selects: 1 or 2 = that EMA, 0 = the raw weights (ckpt.pt only). Returns (denoiser in eval mode, checkpoint
    dict without the weights).
    """
    assert ema in (0, 1, 2), f'ema must be 0 (raw weights), 1 or 2, got {ema}'
    checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
    if 'denoiser_args' in checkpoint:
        denoiser_args = dict(checkpoint['denoiser_args'])
    else:
        # released JiT checkpoint: rebuild from its argparse namespace
        a = checkpoint['args']
        denoiser_args = dict(model=a.model, img_size=a.img_size, num_classes=a.class_num, P_mean=a.P_mean, P_std=a.P_std,
                             t_eps=a.t_eps, noise_scale=a.noise_scale)
    denoiser = build_denoiser(**denoiser_args).to(device)
    denoiser.load_state_dict(checkpoint['model' if ema == 0 else f'model_ema{ema}'])
    meta = {k: v for k, v in checkpoint.items() if not k.startswith('model') and k not in ('optimizer', 'projector', 'masker')}
    return denoiser.eval(), dict(meta, denoiser_args=denoiser_args)

def ema_state_dicts(model):
    # same format as JiT's checkpoints: full state dicts with parameters replaced by their EMA values
    sd1, sd2 = dict(model.state_dict()), dict(model.state_dict())
    for i, (name, _) in enumerate(model.named_parameters()):
        sd1[name] = model.ema_params1[i].detach().clone()
        sd2[name] = model.ema_params2[i].detach().clone()
    return sd1, sd2

@contextmanager
def ema_weights(model, which=1):
    """
    Temporarily load EMA parameters into the model (JiT samples with the first EMA by default).
    """
    ema = model.ema_params1 if which == 1 else model.ema_params2
    params = list(model.parameters())
    backup = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, e in zip(params, ema):
            p.copy_(e)
    try:
        yield model
    finally:
        with torch.no_grad():
            for p, b in zip(params, backup):
                p.copy_(b)

@torch.no_grad()
def generate_and_score(model, metrics, ref_stats, num_images, num_classes, device, ctx,
                       cfg=1.0, interval=(0.0, 1.0), steps=50, method='heun', batch_size=256, seed=0, on_batch=None,
                       fixed_shape=False, eta=1.0):
    """
    Class-balanced generation (num_images / num_classes per class, as in JiT) and FID/IS against ref_stats.
    A fixed seed gives every evaluation the same noise and labels, so FID differences between runs
    aren't just sampling noise. on_batch(x_uint8, labels) sees every generated batch (e.g. for spectra / saving).
    fixed_shape: every network call has batch_size samples, so a compiled network needs one graph (eval_jit.py):
    batch_size images per batch without CFG, batch_size / 2 with batched CFG (conditional + unconditional), and the last
    batch padded to full size with the extra samples discarded. batch_size must be even.
    Changes which noise each sample gets, so it's off for training-time evals (comparable across runs).
    Returns {'fid', 'is', 'features'} where features are the samples' Inception pool features.
    """
    assert num_images % num_classes == 0, 'number of images per class must be the same'
    was_training = model.training
    model.eval()
    model.cfg_scale, model.cfg_interval, model.steps, model.method, model.eta = cfg, interval, steps, method, eta
    labels_all = np.arange(num_classes).repeat(num_images // num_classes)
    if fixed_shape and cfg != 1.0: # batched CFG doubles every network call
        assert batch_size % 2 == 0, f'batch_size must be even with fixed_shape, got {batch_size}'
        batch_size //= 2
    model.fixed_shape = fixed_shape

    feats, logits = [], []
    with torch.random.fork_rng(devices=[device] if 'cuda' in str(device) else []):
        torch.manual_seed(seed)
        for start in tqdm(range(0, num_images, batch_size), desc=f'sampling (cfg {cfg})'):
            batch_labels = labels_all[start:start + batch_size]
            n = len(batch_labels)
            if fixed_shape and n < batch_size: # pad the last batch; the extra samples are discarded below
                batch_labels = np.concatenate([batch_labels, labels_all[:batch_size - n]])
            labels = torch.from_numpy(batch_labels).long().to(device)
            with ctx:
                x = model.generate(labels)
            x, labels = to_uint8(x[:n]), labels[:n]
            if on_batch is not None:
                on_batch(x, labels)
            f, l = metrics.features(x)
            feats.append(f)
            logits.append(l)
    model.train(was_training)
    model.fixed_shape = False
    feats = torch.cat(feats)
    fid, isc = metrics.compute(feats, torch.cat(logits), ref_stats)
    return {'fid': fid, 'is': isc, 'features': feats}

def save_labeled_grid(x, names, path, nrow=8):
    """
    Saves a grid of images ([B, 3, H, W] in [-1, 1]) with each image's class name written underneath.
    """
    from PIL import Image, ImageDraw
    imgs = to_uint8(x).permute(0, 2, 3, 1).cpu().numpy()
    B, H, W, _ = imgs.shape
    caption = 12
    ncol = min(nrow, B)
    nrows = (B + ncol - 1) // ncol
    canvas = Image.new('RGB', (ncol * W, nrows * (H + caption)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (img, name) in enumerate(zip(imgs, names)):
        r, c = divmod(i, ncol)
        canvas.paste(Image.fromarray(img), (c * W, r * (H + caption)))
        draw.text((c * W + 2, r * (H + caption) + H), name[: max(4, W // 6)], fill='black')
    canvas.save(path)

def save_layout_grids(net, x, names, prefix, history):
    """
    QuadJiT sample grids with the token layout drawn on: {prefix}_layout.png (the last routing call's layout on every
    sample) and {prefix}_layout_steps.png (the first 8 samples: the sample, then the layouts of the routing calls nearest
    t = 0.25, 0.5, 0.75 and the last one, drawn on the final sample). history: the router's recorded (layout, mean t)
    pairs from generating x; CFG batches carry the conditional half first, so the first len(x) rows are x's.
    """
    from model_quadtree import draw_layout
    if not history:
        return
    B, geo = x.shape[0], net.geo
    save_labeled_grid(draw_layout(x, history[-1][0][:B], geo), names, prefix + '_layout.png')
    picks = [min(range(len(history)), key=lambda i: abs(history[i][1] - t)) for t in (0.25, 0.5, 0.75)] + [len(history) - 1]
    n = min(8, B)
    cells, labels = [], []
    for j in range(n):
        cells.append(x[j:j + 1])
        labels.append(names[j])
        for i in picks:
            cells.append(draw_layout(x[j:j + 1], history[i][0][j:j + 1], geo))
            labels.append(f't={history[i][1]:.2f}')
    save_labeled_grid(torch.cat(cells), labels, prefix + '_layout_steps.png', nrow=1 + len(picks))

def save_geometry_grid(net, x, names, path, n=8):
    """
    GaussJiT samples with the geometry of the last network call (the end of sampling; CFG batches carry the conditional
    half first): every token's Gaussian, its displacement from its anchor, and the soft ownership (model_gauss.geometry_figure).
    """
    from model_gauss import geometry_figure
    if net.last_geometry is None:
        return
    n = min(n, x.shape[0])
    mu, h, theta = (g[:n] for g in net.last_geometry)
    geometry_figure(net, path, x[:n], [('end of sampling', mu, h, theta)], row_labels=names[:n],
                    title='geometry of the sampled images at the last sampling step (sampling weights)')

def plot_geometry_vs_t(stats, path, title, rotation=True):
    """
    Token geometry vs noise level from {t: stats} (keys offset_mean / _p95, scale_p5 / _p50 / _p95, rotation_p50 / _p95,
    outside_frame): one panel each for displacement, per-axis scale, |rotation| (if rotation) and the share of pixels
    outside their dominant token's frame.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from plot_style import SURFACE, INK_2, style_axis, line, CATEGORICAL
    tv = sorted(stats)
    panels = [('displacement (anchor spacings)', [('mean', 'offset_mean'), ('p95', 'offset_p95')]),
              ('per-axis scale (x initial size)', [('p5', 'scale_p5'), ('p50', 'scale_p50'), ('p95', 'scale_p95')])]
    panels.append(('token-to-token vs whole-image (std of log scale)', [('within images', 'scale_within'), ('between images', 'scale_between')]))
    if rotation:
        panels.append(('rotation (degrees)', [('|rot| p50', 'rotation_p50'), ('|rot| p95', 'rotation_p95'), ('signed mean', 'rotation_mean'),
                                              ('within-image std', 'rotation_within')]))
    panels.append(("pixels outside their dominant token's frame", [('share', 'outside_frame')]))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.4), facecolor=SURFACE, squeeze=False)
    for ax, (ptitle, series) in zip(axes[0], panels):
        for k, (lab, key) in enumerate(series):
            line(ax, tv, [stats[t][key] for t in tv], CATEGORICAL[k], 'o', lab)
        style_axis(ax, 't (1 = clean)', '', ptitle)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
        if series[0][1] == 'outside_frame': # a share: percent, from zero
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_ylim(0, max(0.05, 1.2 * max(stats[t]['outside_frame'] for t in tv)))
    fig.suptitle(title, fontsize=9, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)

def save_geometry_live(net, x, y, path, noise_scale, ts=(0.3, 0.6, 0.9), seed=0, t_grid=19, x_stats=None):
    """
    The geometry the network predicts now (call with the training weights) for held-out images x noised to t:
      {path}                      one column per t in ts (model_gauss.geometry_figure)
      {path stem}_vs_t.png        displacement / per-axis scale / |rotation| percentiles and the share of pixels outside
                                  their dominant token's frame vs t, over t_grid noise levels, from x_stats (default x)
      {path stem}.gif             t sweeping from noisy to clean: the noisy input (top) and the Gaussians on the clean image
    Returns {t: geometry stats} for logging. Shows what the placer is learning even while sampling weights (EMA) lag.
    """
    import math
    import numpy as np
    from model_gauss import geometry_figure, draw_gaussians
    g = torch.Generator(device=x.device).manual_seed(seed)
    e = torch.randn(x.shape, generator=g, device=x.device) * noise_scale
    def geo_at(t_val):
        t = torch.full((x.shape[0],), t_val, device=x.device)
        z = t_val * x + (1 - t_val) * e
        mu, h, theta, _ = net.geometry(z, net.t_embedder(t).float())
        return z, mu, h, theta
    xs = x if x_stats is None else x_stats
    es = torch.randn(xs.shape, generator=torch.Generator(device=xs.device).manual_seed(seed + 1), device=xs.device) * noise_scale
    def geo_stats_at(t_val, chunk=64):
        # geometry of every image in xs noised to t (placer forward only), in chunks
        out = []
        for i in range(0, xs.shape[0], chunk):
            t = torch.full((len(xs[i:i + chunk]),), t_val, device=xs.device)
            mu, h, theta, _ = net.geometry(t_val * xs[i:i + chunk] + (1 - t_val) * es[i:i + chunk], net.t_embedder(t).float())
            out.append((mu, h, theta))
        return [torch.cat(z) for z in zip(*out)]
    with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
        cols = [(f't={t_val:g}', *geo_at(t_val)[1:]) for t_val in ts]
        grid = [float(v) for v in np.linspace(0.05, 0.95, t_grid)]
        sweep = [(t_val, *geo_at(t_val)) for t_val in grid]
        stat_geo = {t_val: geo_stats_at(t_val) for t_val in grid}
    geometry_figure(net, path, x, cols, title='geometry predicted from held-out images noised to t (training weights)')
    stem = path.rsplit('.', 1)[0]
    # statistics vs t
    S, q = net.patch_size, torch.tensor([0.05, 0.5, 0.95], device=x.device)
    stats = {}
    for t_val in grid:
        mu, h, theta = stat_geo[t_val]
        off = ((mu - net.anchors) / S).norm(dim=-1).flatten()
        f = (h / (net.extent * S)).flatten()
        rot = theta.abs().flatten() * 180 / math.pi
        stats[t_val] = {'offset_mean': off.mean().item(), 'offset_p95': torch.quantile(off, 0.95).item(),
                        **{f'scale_{k}': v for k, v in zip(('p5', 'p50', 'p95'), torch.quantile(f, q).tolist())},
                        'rotation_p50': torch.quantile(rot, 0.5).item(), 'rotation_p95': torch.quantile(rot, 0.95).item(),
                        'outside_frame': net.outside_frame_share(mu, h, theta), **net.spread_stats(h, theta)}
    plot_geometry_vs_t(stats, stem + '_vs_t.png', f'token geometry vs noise level ({xs.shape[0]} held-out images x {net.tokens} tokens, '
                       f'training weights)', net.max_rotation > 0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_style import SURFACE
    # animation over t
    from PIL import Image
    n = min(3, x.shape[0])
    frames = []
    for t_val, z, mu, h, theta in sweep:
        fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 4.9), facecolor=SURFACE, squeeze=False, dpi=72)
        for j_ in range(n):
            for ax in axes[:, j_]:
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            axes[0, j_].imshow(((z[j_] / max(1e-6, z[j_].abs().max().item())).clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy())
            draw_gaussians(axes[1, j_], net, x[j_], mu[j_], h[j_], theta[j_])
        axes[0, 0].set_ylabel('noisy input', fontsize=8); axes[1, 0].set_ylabel('Gaussians (arrows 4x)', fontsize=8)
        fig.suptitle(f't = {t_val:.2f}  (1 = clean)', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    frames = [f.quantize(colors=128) for f in frames] # palette frames: a few hundred KB
    frames[0].save(stem + '.gif', save_all=True, append_images=frames[1:] + frames[::-1][1:-1], duration=180, loop=0, optimize=True)
    return stats

# settings that don't change what is trained (budget, evaluation, logging, system), left out of run directory names;
# diag* (diagnostics) settings are excluded too
_NAME_EXCLUDE = {
    'out_dir', 'runs_root', 'run_name', 'data_root', 'data_dir', 'dataset', 'model', 'depth', 'hidden_size', 'num_heads',
    'patch_size', 'bottleneck_dim', 'in_context_len', 'in_context_start', 'mlp_ratio', 'pixel_depth', 'pixel_dim',
    'epochs', 'batch_size', 'gradient_accumulation_steps', 'eval_every_epochs', 'save_every_epochs', 'keep_ckpt_every_epochs', 'log_interval',
    'eval_only', 'init_from', 'wandb_log', 'wandb_project', 'wandb_run_name', 'fid_num_images', 'fid_cfg',
    'fid_interval_min', 'fid_interval_max', 'sample_cfg', 'sample_interval_min', 'sample_interval_max',
    'sampling_method', 'sampling_eta', 'num_sampling_steps', 'time_schedule', 'gen_batch_size', 'eval_loss_images',
    'device', 'dtype', 'compile', 'trend_plots', 'progress_gif', 'data_workers', 'data_prefetch', 'quad_route_baseline', 'quad_route_every',
}
_ARCH_SHORT = {'depth': 'd', 'hidden_size': 'w', 'num_heads': 'h', 'patch_size': 'p', 'bottleneck_dim': 'bn',
               'in_context_len': 'icl', 'in_context_start': 'ics', 'mlp_ratio': 'mlp', 'pixel_depth': 'pd', 'pixel_dim': 'pdim'}

# short, shell-friendly name tokens for the settings that change training. A value is appended to the prefix
# ('noise' + 0.75 -> noise0.75); string settings use their value ('muon', 'repa-dino_vitb16'); booleans name the change
# from the default ('no-emawarmup'). Settings missing here still get a readable '{key}{value}' token.
_NAME_TOKENS = {
    'optimizer_type': '', 'muon_adjust_lr': 'muon-', 'muon_lr': 'muonlr', 'muon_weight_decay': 'muonwd',
    'muon_momentum': 'muonmom', 'muon_ns_steps': 'muonns', 'muon_split_fused': 'muonsplit', 'muon_adaln': 'muonadaln',
    'noise_scale': 'noise', 'P_mean': 'pmean', 'P_std': 'pstd', 't_eps': 'teps', 'label_drop_prob': 'labeldrop',
    'blr': 'blr', 'lr_schedule': 'lr-', 'min_lr': 'minlr', 'warmup_epochs': 'warmup', 'weight_decay': 'wd',
    'beta1': 'beta1-', 'beta2': 'beta2-', 'grad_clip': 'clip', 'ema_decay1': 'ema', 'ema_decay2': 'ema2-', 'ema_warmup': 'emawarmup',
    'attn_dropout': 'attndrop', 'proj_dropout': 'projdrop', 'attn_res': 'attnres-', 'attn_res_block_size': 'attnresbs',
    'adaln_single': 'adalnsingle', 'attn_res_adaln': 'attnresadaln', 'gated_bottleneck': 'gatedbn',
    'skip_pathway': 'skip', 'skip_head_blocks': 'skiphead', 'skip_inject_block': 'skipat', 'skip_pathway_drop_prob': 'skipdrop', 'skip_stop_grad': 'skipsg', 'skip_final_layer': 'skipfl', 'skip_residual': 'skipres',
    'dcae_stem': 'dcaestem', 'dcae_head': 'dcaehead', 'dcae_trunk_head': 'dcaetrunkhead', 'gate_init': 'gateinit-', 'dcae_norm': 'dcaenorm-', 'pixeldit_prediction': 'pred-',
    'skip_decoder': 'skipdec-', 'skip_decoder_dim': 'skipdecdim', 'skip_decoder_blocks': 'skipdecblocks', 'skip_decoder_rank': 'skipdecrank',
    'skip_decoder_conv': 'skipdecconv', 'skip_decoder_attnres': 'skipdecattnres', 'skip_decoder_modulation': 'skipdecmod', 'dcae_width': 'dcaew', 'dcae_kernel2': 'dcaek', 'patch_overlap': 'overlap', 'head_overlap': 'headoverlap', 'layernorm': 'layernorm',
    'repa_encoder': 'repa-', 'repa_adapter': 'adapter-', 'repa_depth': 'repadepth', 'repa_coeff': 'repacoeff', 'repa_proj_dim': 'repaproj',
    'repa_mask_ratio': 'repamask', 'repa_adapter_depth': 'adapterdepth',
    'mask_ratio': 'mask', 'mask_prob': 'maskprob', 'mask_weighting': 'maskweight-',
    'quad_tokens': 'K', 'quad_levels': 'levels', 'quad_router': 'route-', 'quad_route_dino': 'dino-', 'quad_route_t_min': 'routetmin',
    'quad_gate_tau': 'gatetau', 'quad_pos': 'pos-', 'quad_rope': 'rope-', 'quad_area_bias': 'areabias', 'quad_level_embed': 'levelemb',
    'gauss_extent': 'gext', 'gauss_basis': 'gbasis', 'gauss_sigma': 'gsigma', 'gauss_max_offset': 'goff', 'gauss_scale_range': 'gscale', 'gauss_max_rotation': 'grot',
    'gauss_placer_width': 'gplacerw', 'gauss_placer_depth': 'gplacerd', 'gauss_area_bias': 'gareabias', 'gauss_shape_embed': 'gshape', 'gauss_fused': 'gfused-',
}

def _fmt_value(v):
    if isinstance(v, str):
        v = os.path.splitext(os.path.basename(v.rstrip('/')))[0] if v.startswith(('file:', '/')) else v
        v = v[len('timm:'):] if v.startswith('timm:') else v
        return v.replace('/', '-').replace(':', '-').replace('=', '-').replace(' ', '-')
    return f'{v:g}' if isinstance(v, float) else str(v)

def _name_token(key, value):
    prefix = _NAME_TOKENS.get(key, key.replace('_', '') + ('-' if isinstance(value, str) else ''))
    if isinstance(value, bool):
        return prefix if value else f'no-{prefix}'
    return prefix + _fmt_value(value)

def _model_part(model, arch):
    preset = dict(JIT_PRESETS[model], mlp_ratio=4.0)
    return model.lower().replace('/', '') + ''.join(f'_{_ARCH_SHORT[k]}{arch[k]:g}' for k in _ARCH_SHORT if k in arch and arch[k] != preset.get(k))

def _changed(config, defaults):
    # in _NAME_TOKENS order (optimizer first, then noise schedule, ...), settings not in the table after, alphabetically
    order = {k: i for i, k in enumerate(_NAME_TOKENS)}
    changed = [(k, v) for k, v in config.items() if k not in _NAME_EXCLUDE and not k.startswith('diag') and v != defaults[k]]
    return sorted(changed, key=lambda kv: (order.get(kv[0], len(order)), kv[0]))

def run_dir(runs_root, dataset, model, arch, config, defaults, run_name=''):
    """
    {runs_root}/{dataset}/{model}/{changes}[_{run_name}], e.g. runs/in100_128/jit-b16_d6_w384_h6/muon_noise0.75.
    model: preset short name plus architecture fields that differ from the preset. changes: a short token for every other
    setting that differs from the script defaults and affects training ('base' if none). Budget / eval / logging / system
    settings are excluded, so extending --epochs and resuming lands in the same directory.
    """
    leaf = '_'.join(_name_token(k, v) for k, v in _changed(config, defaults)) or 'base'
    if run_name:
        leaf += f'_{run_name}'
    return os.path.join(runs_root, dataset, _model_part(model, arch), leaf)

def legacy_run_dir(runs_root, dataset, model, arch, config, defaults, run_name=''):
    # the earlier key=value naming (alphabetical, paths kept their extension), so runs created before the rename can still be resumed
    def fmt(v):
        if isinstance(v, str):
            v = os.path.basename(v.rstrip('/')) if v.startswith(('file:', '/')) else v
            return v.replace('/', '-').replace(':', '-')
        return f'{v:g}' if isinstance(v, float) else str(v)
    changed = sorted((k, v) for k, v in config.items() if k not in _NAME_EXCLUDE and not k.startswith('diag') and v != defaults[k])
    leaf = '_'.join(f'{k}={fmt(v)}' for k, v in changed) or 'base'
    if run_name:
        leaf += f'_{run_name}'
    return os.path.join(runs_root, dataset, _model_part(model, arch), leaf)
