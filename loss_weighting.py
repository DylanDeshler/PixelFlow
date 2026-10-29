"""
Loss weightings for JiT's training loss (x-prediction, v-loss), discoverable from the data rather than tuned per dataset.
Used by masking.flow_loss when train_jit.py runs with --loss_freq_weight / --loss_t_weight; off by default.

freq='wiener'   within each noise level, weight every frequency f of the residual r = (x - x_pred) / (1 - t) by its
                Wiener gain SNR / (1 + SNR), SNR = t^2 P(f) / ((1 - t)^2 noise_scale^2), P the data's power spectrum: the
                share of that frequency the noisy input still carries. Fine detail destroyed by the noise (unpredictable,
                its loss mostly gradient noise) is down-weighted; at low noise every gain is ~1. A weighted L2 in an
                orthonormal (Fourier) basis has the same minimizer, the posterior mean, so this changes how efficiently the
                model learns, not what it converges to. Each noise level's weighted loss is rescaled to the plain loss's
                running mean at that level (per logit(t) bin), so the weighting only redistributes effort among frequencies
                within a level; how much each level counts is still set by the t distribution (P_mean / P_std). At high
                noise only the coarsest frequencies survive, so the factor can be large (it hands that level's whole
                budget to its predictable part); it's capped at norm_cap for stability and logged (wiener_norm_t*).
t_weight='learned'
                EDM2's uncertainty weighting (Karras et al. 2024): a learned log-variance u(t) per noise level (random Fourier
                features of logit(t) -> linear, zero init), loss L / e^u + u, so e^u tracks E[L | t] and every level's loss
                is normalized to ~1. It balances loss scales across levels; the allocation is then set by the t
                distribution. With freq='wiener', the normalized loss reflects the predictable part of the error.
"""

import math

import torch
import torch.nn as nn


def image_power_spectrum(data, n_images=2048, seed=0):
    """
    Radially averaged power spectrum of a MemmapImages dataset in model units ([-1, 1]): E |FFT(x)|^2 / (H W) averaged over
    images and channels (white noise of variance s^2 has s^2 at every frequency) -> [H, W] float32 (CPU).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(data), min(n_images, len(data)), replace=False))
    x, _ = data.get(idx)
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 127.5 - 1
    H, W = x.shape[-2:]
    p = sum(torch.fft.fft2(x[i:i + 256]).abs().pow(2).sum(dim=(0, 1)) for i in range(0, len(x), 256)) / (x.shape[0] * x.shape[1] * H * W)
    fy, fx = torch.fft.fftfreq(H) * H, torch.fft.fftfreq(W) * W
    r = torch.sqrt(fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2).round().long()
    radial = torch.zeros(int(r.max()) + 1).scatter_add_(0, r.flatten(), p.flatten())
    count = torch.zeros_like(radial).scatter_add_(0, r.flatten(), torch.ones_like(p).flatten())
    return (radial / count.clamp_min(1))[r]


class LossWeighting(nn.Module):
    def __init__(self, noise_scale, power=None, freq='none', t_weight='none', bins=48, logit_range=8.0, decay=0.99, fourier=16,
                 norm_cap=100.0):
        super().__init__()
        assert freq in ('none', 'wiener'), f"freq must be 'none' or 'wiener', got {freq}"
        assert t_weight in ('none', 'learned'), f"t_weight must be 'none' or 'learned', got {t_weight}"
        assert freq == 'none' or power is not None, "freq='wiener' needs the data's power spectrum"
        self.freq, self.t_weight, self.noise_scale = freq, t_weight, noise_scale
        self.bins, self.logit_range, self.decay, self.norm_cap = bins, logit_range, decay, norm_cap
        if freq == 'wiener':
            self.register_buffer('power', power.float())
            # running mean of the plain and weighted loss per logit(t) bin, for the per-level normalization
            self.register_buffer('ema_plain', torch.zeros(bins))
            self.register_buffer('ema_weighted', torch.zeros(bins))
            self.register_buffer('seen', torch.zeros(bins, dtype=torch.bool))
        if t_weight == 'learned':
            g = torch.Generator().manual_seed(0)
            self.register_buffer('fourier_freqs', torch.randn(fourier, generator=g))
            self.register_buffer('fourier_phases', torch.rand(fourier, generator=g) * 2 * math.pi)
            self.logvar = nn.Linear(fourier, 1)
            nn.init.zeros_(self.logvar.weight)
            nn.init.zeros_(self.logvar.bias)

    def _logit(self, t):
        t = t.float().clamp(1e-6, 1 - 1e-6)
        return torch.log(t / (1 - t))

    def log_variance(self, t):
        # u(t), the learned log-variance of the loss at noise level t [B]
        f = torch.cos(self._logit(t)[:, None] * self.fourier_freqs + self.fourier_phases) * math.sqrt(2)
        return self.logvar(f).squeeze(-1)

    def wiener_gain(self, t):
        # [B, H, W]: SNR / (1 + SNR) per frequency at each sample's noise level
        tt = t.float().clamp(1e-6, 1 - 1e-6)[:, None, None]
        snr = tt ** 2 * self.power / ((1 - tt) ** 2 * self.noise_scale ** 2)
        return snr / (1 + snr)

    def _bin(self, t):
        lt = self._logit(t).clamp(-self.logit_range, self.logit_range - 1e-6)
        return ((lt + self.logit_range) / (2 * self.logit_range) * self.bins).long()

    @torch.no_grad()
    def _update(self, b, plain, weighted):
        cnt = torch.zeros(self.bins, device=b.device).scatter_add_(0, b, torch.ones_like(plain))
        mp = torch.zeros(self.bins, device=b.device).scatter_add_(0, b, plain) / cnt.clamp_min(1)
        mw = torch.zeros(self.bins, device=b.device).scatter_add_(0, b, weighted) / cnt.clamp_min(1)
        hit = cnt > 0
        d = self.decay
        self.ema_plain.copy_(torch.where(hit, torch.where(self.seen, d * self.ema_plain + (1 - d) * mp, mp), self.ema_plain))
        self.ema_weighted.copy_(torch.where(hit, torch.where(self.seen, d * self.ema_weighted + (1 - d) * mw, mw), self.ema_weighted))
        self.seen |= hit

    def forward(self, r, t):
        """
        r: the v-loss residual (x - x_pred) / (1 - t) [B, C, H, W]; t: noise levels [B] (1 = clean). Returns the scalar loss.
        """
        plain = r.float().pow(2).mean(dim=(1, 2, 3))
        per = plain
        if self.freq == 'wiener':
            H, W = r.shape[-2:]
            R2 = torch.fft.fft2(r.float()).abs().pow(2).mean(dim=1) / (H * W) ** 2 # [B, H, W]; sums over f to plain
            weighted = (self.wiener_gain(t) * R2).sum(dim=(1, 2))
            b = self._bin(t)
            if self.training:
                self._update(b, plain.detach(), weighted.detach())
            per = weighted * (self.ema_plain / self.ema_weighted.clamp_min(1e-12)).clamp(max=self.norm_cap)[b]
        if self.t_weight == 'learned':
            u = self.log_variance(t)
            per = per / torch.exp(u) + u
        return per.mean()

    @torch.no_grad()
    def summary(self, ts=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)):
        # for logging: the learned weight e^-u(t) and the normalization factor (plain / weighted) at a few noise levels
        dev = next(iter(self.buffers()), torch.zeros(1)).device
        t = torch.tensor(ts, device=dev)
        out = {}
        if self.t_weight == 'learned':
            out.update({f'loss_weight_t{v:.2f}': w for v, w in zip(ts, torch.exp(-self.log_variance(t)).tolist())})
        if self.freq == 'wiener':
            b = self._bin(t)
            ratio = (self.ema_plain / self.ema_weighted.clamp_min(1e-12)).clamp(max=self.norm_cap)[b]
            out.update({f'wiener_norm_t{v:.2f}': (q if s else float('nan')) for v, q, s in zip(ts, ratio.tolist(), self.seen[b].tolist())})
        return out
