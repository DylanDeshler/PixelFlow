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

# architecture of each JiT preset, as defined in model_jit.py (mlp_ratio is the JiT default for all of them)
JIT_PRESETS = {
    'JiT-B/16': dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16),
    'JiT-B/32': dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32),
    'JiT-L/16': dict(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16),
    'JiT-L/32': dict(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32),
    'JiT-H/16': dict(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16),
    'JiT-H/32': dict(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32),
}
for _name, _arch in JIT_PRESETS.items():
    assert _name in model_jit.JiT_models, f'{_name} missing from model_jit.JiT_models'

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

def _register(name, arch):
    # JiT's Denoiser looks the network up in model_jit.JiT_models by name, so custom architectures get registered there
    if name not in model_jit.JiT_models:
        model_jit.JiT_models[name] = lambda **kwargs: model_jit.JiT(**arch, **kwargs)


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
        if self.cfg_scale == 1.0 or not bool(in_interval.any()):
            x_cond = self.net(z, t.flatten(), labels)
            return (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        x_both = self.net(torch.cat([z, z]), torch.cat([t.flatten(), t.flatten()]),
                          torch.cat([labels, torch.full_like(labels, self.num_classes)]))
        x_cond, x_uncond = x_both.chunk(2)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)
        scale = torch.where(in_interval, self.cfg_scale, 1.0)
        return v_uncond + scale * (v_cond - v_uncond)

    time_schedule = 'uniform'

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

    @torch.no_grad()
    def generate(self, labels):
        # JiT's Denoiser.generate with a configurable time grid (identical to JiT's for time_schedule='uniform')
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = self.sampling_timesteps(device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError
        for i in range(self.steps - 1):
            z = stepper(z, timesteps[i], timesteps[i + 1], labels)
        # last step euler
        return self._euler_step(z, timesteps[-2], timesteps[-1], labels)

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
                       cfg=1.0, interval=(0.0, 1.0), steps=50, method='heun', batch_size=256, seed=0, on_batch=None):
    """
    Class-balanced generation (num_images / num_classes per class, as in JiT) and FID/IS against ref_stats.
    A fixed seed gives every evaluation the same noise and labels, so FID differences between runs
    aren't just sampling noise. on_batch(x_uint8, labels) sees every generated batch (e.g. for spectra / saving).
    Returns {'fid', 'is', 'features'} where features are the samples' Inception pool features.
    """
    assert num_images % num_classes == 0, 'number of images per class must be the same'
    was_training = model.training
    model.eval()
    model.cfg_scale, model.cfg_interval, model.steps, model.method = cfg, interval, steps, method
    labels_all = np.arange(num_classes).repeat(num_images // num_classes)

    feats, logits = [], []
    with torch.random.fork_rng(devices=[device] if 'cuda' in str(device) else []):
        torch.manual_seed(seed)
        for start in tqdm(range(0, num_images, batch_size), desc=f'sampling (cfg {cfg})'):
            labels = torch.from_numpy(labels_all[start:start + batch_size]).long().to(device)
            with ctx:
                x = model.generate(labels)
            x = to_uint8(x)
            if on_batch is not None:
                on_batch(x, labels)
            f, l = metrics.features(x)
            feats.append(f)
            logits.append(l)
    model.train(was_training)
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

# settings that don't change what is trained (budget, evaluation, logging, system), left out of run directory names;
# diag* (diagnostics) settings are excluded too
_NAME_EXCLUDE = {
    'out_dir', 'runs_root', 'run_name', 'data_root', 'data_dir', 'dataset', 'model', 'depth', 'hidden_size', 'num_heads',
    'patch_size', 'bottleneck_dim', 'in_context_len', 'in_context_start', 'mlp_ratio',
    'epochs', 'batch_size', 'gradient_accumulation_steps', 'eval_every_epochs', 'save_every_epochs', 'log_interval',
    'eval_only', 'init_from', 'wandb_log', 'wandb_project', 'wandb_run_name', 'fid_num_images', 'fid_cfg',
    'fid_interval_min', 'fid_interval_max', 'sample_cfg', 'sample_interval_min', 'sample_interval_max',
    'sampling_method', 'num_sampling_steps', 'time_schedule', 'gen_batch_size', 'eval_loss_images',
    'device', 'dtype', 'compile', 'trend_plots',
}
_ARCH_SHORT = {'depth': 'd', 'hidden_size': 'w', 'num_heads': 'h', 'patch_size': 'p', 'bottleneck_dim': 'bn',
               'in_context_len': 'icl', 'in_context_start': 'ics', 'mlp_ratio': 'mlp'}

# short, shell-friendly name tokens for the settings that change training. A value is appended to the prefix
# ('noise' + 0.75 -> noise0.75); string settings use their value ('muon', 'repa-dino_vitb16'); booleans name the change
# from the default ('no-emawarmup'). Settings missing here still get a readable '{key}{value}' token.
_NAME_TOKENS = {
    'optimizer_type': '', 'muon_adjust_lr': 'muon-', 'muon_lr': 'muonlr', 'muon_weight_decay': 'muonwd',
    'muon_momentum': 'muonmom', 'muon_ns_steps': 'muonns', 'muon_split_fused': 'muonsplit', 'muon_adaln': 'muonadaln',
    'noise_scale': 'noise', 'P_mean': 'pmean', 'P_std': 'pstd', 't_eps': 'teps', 'label_drop_prob': 'labeldrop',
    'blr': 'blr', 'lr_schedule': 'lr-', 'min_lr': 'minlr', 'warmup_epochs': 'warmup', 'weight_decay': 'wd',
    'beta1': 'beta1-', 'beta2': 'beta2-', 'grad_clip': 'clip', 'ema_decay1': 'ema', 'ema_decay2': 'ema2-', 'ema_warmup': 'emawarmup',
    'attn_dropout': 'attndrop', 'proj_dropout': 'projdrop',
    'repa_encoder': 'repa-', 'repa_depth': 'repadepth', 'repa_coeff': 'repacoeff', 'repa_proj_dim': 'repaproj',
    'mask_ratio': 'mask', 'mask_prob': 'maskprob', 'mask_weighting': 'maskweight-',
}

def _fmt_value(v):
    if isinstance(v, str):
        v = os.path.splitext(os.path.basename(v.rstrip('/')))[0] if v.startswith(('file:', '/')) else v
        return v.replace('/', '-').replace(':', '-').replace('=', '-').replace(' ', '-')
    return f'{v:g}' if isinstance(v, float) else str(v)

def _name_token(key, value):
    prefix = _NAME_TOKENS.get(key, key.replace('_', '') + ('-' if isinstance(value, str) else ''))
    if isinstance(value, bool):
        return prefix if value else f'no-{prefix}'
    return prefix + _fmt_value(value)

def _model_part(model, arch):
    preset = dict(JIT_PRESETS[model], mlp_ratio=4.0)
    return model.lower().replace('/', '') + ''.join(f'_{_ARCH_SHORT[k]}{arch[k]:g}' for k in _ARCH_SHORT if arch[k] != preset[k])

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
