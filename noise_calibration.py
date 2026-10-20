"""
Data-driven calibration of the noise scale / logit-normal timestep distribution for flow matching.

Model: z_t = t * x + (1 - t) * s * eps (JiT's convention, t=1 clean). The log-SNR is
    u = log(t^2 / ((1 - t)^2 s^2)) = 2 * (logit(t) - log s),
so the noise scale s and a shift of the logit-normal mean are interchangeable as far as noise levels go:
    noise scale s at P_mean mu  <=>  noise scale 1 at P_mean mu - log(s).
(They aren't identical for the network: input magnitudes and JiT's v-loss weighting differ.)

The data is approximated as a stationary Gaussian: channels are decorrelated (PCA, orthonormal so noise stays
white) and each Fourier coefficient k gets its measured variance p_k. Under that model, closed forms as a function of u:
    R^2(u)       explained variance of the optimal (Wiener) denoiser,  1 - sum_k p_k / (1 + e^u p_k) / sum_k p_k
                 -> power weighted, dominated by coarse structure
    info(u)      fraction of the information about the quantized signal that z_t reveals (I-MMSE / Gaussian channel),
                 sum_k [log(1 + e^u p_k) - log(1 + e^u q)]_+ / sum_k log(p_k / q)_+,  q = quantization noise variance
                 -> counts every coefficient equally, dominated by fine detail
Both rise from 0 (pure noise) to 1 (clean), so their derivatives are densities over noise levels that say where the
data's content is revealed. The script reports, for the current setting:
    - where the training t distribution sits on both curves (how much training goes to trivial / hopeless noise levels)
    - logit-normal (P_mean, P_std) fits to both transition densities. These bracket a good setting rather than determine
      it: on real ImageNet photos at 256px (noise scale 1) the R^2 fit gives P_mean ~ -3.0 and the info fit ~ +4.3, and
      JiT's tuned -0.8 sits ~30% of the way from the first to the second (it also trains a narrower P_std, 0.8 vs ~1.7-1.9).
      Without a reference, use them as bounds for a sweep.
    - per-frequency-band SNR crossings (the coarse-to-fine order in which structure appears)
    - with --ref_data, matched against a known-good reference setting (e.g. JiT's ImageNet-256 recipe):
        shared-content match (same modality, any resolution): both datasets are average-pooled to their common size, which
          divides the noise std by sqrt(#values pooled), and the coarse content's R^2 is matched there. For the same content
          at a different resolution this reproduces the sqrt(m/n) rule (JiT's img_size/256, SD3's shift); for different
          content it adds a data-dependent correction. This is the one to use within a modality.
        full-spectrum matches (any modality): put the training distribution at the same expected R^2 / info as the reference.
          Heuristic: the two datasets' curves can differ in shape (e.g. a higher resolution adds an octave of fine detail),
          so no single shift matches everything; R^2 and info bracket the coarse vs fine trade-off.

The Gaussian model captures redundancy / resolution / spectral shape, which is what noise schedules compensate for,
but not semantic structure. Treat the outputs as a starting point and confirm with a short sweep (judged on samples,
not training loss, which isn't comparable across noise settings).

$ python noise_calibration.py --data=/data/jit/in100_128
$ python noise_calibration.py --data=/data/jit/in100_128 --ref_data=/data/jit/in1k_256 --ref_noise_scale=1.0
$ python noise_calibration.py --data=audio_clips.npy --quant_bits=16 --noise_scale=1.0   # generic [N, C, *spatial] arrays
"""

import os
import json
import math
import argparse

import numpy as np
import torch


# -----------------------------------------------------------------------------
# data

def load_samples(path, max_samples, seed=0):
    """
    Returns float tensor [N, C, *spatial] in model units. Prepared JiT dirs (meta.json) are converted to [-1, 1];
    .npy files are used as is (shape [N, C, *spatial], already scaled the way the model sees them).
    """
    if os.path.isdir(path):
        from data_jit import MemmapImages
        data = MemmapImages(path)
        idx = np.linspace(0, len(data) - 1, min(max_samples, len(data))).round().astype(np.int64)
        x, _ = data.get(idx)
        return torch.from_numpy(x).permute(0, 3, 1, 2).float() / 127.5 - 1
    arr = np.load(path, mmap_mode='r')
    idx = np.sort(np.random.default_rng(seed).choice(len(arr), min(max_samples, len(arr)), replace=False))
    return torch.from_numpy(np.asarray(arr[idx], dtype=np.float32))

def coefficient_variances(x):
    """
    Variance of every Fourier coefficient after decorrelating channels: [C, *spatial] flattened to [D].
    Orthonormal transforms (PCA over channels, ortho FFT) keep the injected noise white with unit variance per coefficient.
    """
    N, C = x.shape[:2]
    spatial_dims = tuple(range(2, x.ndim))
    chunk = max(1, int(2e7 // math.prod(x.shape[1:]))) # bounds the float64 / complex128 buffers to ~300MB
    # the mean is known to the denoiser, so only fluctuations around it count
    mean = sum(x[i:i + chunk].double().sum(0, keepdim=True) for i in range(0, N, chunk)) / N
    cov = sum(torch.einsum('nc...,nd...->cd', x[i:i + chunk].double() - mean, x[i:i + chunk].double() - mean)
              for i in range(0, N, chunk)) / (N * math.prod(x.shape[2:]))
    _, V = torch.linalg.eigh(cov)
    power = 0
    for i in range(0, N, chunk):
        xc = torch.einsum('cd,nc...->nd...', V, x[i:i + chunk].double() - mean)
        power = power + torch.fft.fftn(xc, dim=spatial_dims, norm='ortho').abs().pow(2).sum(0)
    return (power / N).flatten()

def pool_to(x, size):
    """
    Area-average x [N, C, *spatial] down to spatial size `size` (tuple). Returns pooled x and the number of values
    averaged into each output value (the noise std shrinks by its square root).
    """
    factor = math.prod(a / b for a, b in zip(x.shape[2:], size))
    return torch.nn.functional.interpolate(x, size=size, mode='area'), factor

def radial_frequency(shape):
    # normalized radial frequency (1 = Nyquist) of every coefficient in a [C, *spatial] layout, flattened
    C, spatial = shape[0], shape[1:]
    grids = torch.meshgrid(*[torch.fft.fftfreq(n, dtype=torch.float64) for n in spatial], indexing='ij')
    r = torch.sqrt(sum(g ** 2 for g in grids)) / 0.5
    return r.unsqueeze(0).expand(C, *spatial).flatten()

# -----------------------------------------------------------------------------
# Gaussian-model curves over log-SNR u

def _over_u(fn, u, n_coeffs):
    # evaluates fn on chunks of u so the [len(u), n_coeffs] intermediate stays ~100MB
    step = max(1, int(1.2e7 // n_coeffs))
    return torch.cat([fn(u[i:i + step]) for i in range(0, len(u), step)])

def r2_curve(p, u):
    def mmse(uc):
        return (p.unsqueeze(0) / (1 + torch.exp(uc).unsqueeze(1) * p.unsqueeze(0))).sum(1)
    return 1 - _over_u(mmse, u, len(p)) / p.sum()

def info_curve(p, u, q):
    p = p[p > q] # coefficients below the quantization floor carry no information
    def info(uc):
        snr = torch.exp(uc).unsqueeze(1)
        return (torch.log1p(snr * p.unsqueeze(0)) - torch.log1p(snr * q)).clamp_min(0).sum(1)
    return _over_u(info, u, len(p)) / torch.log(p / q).sum()

# -----------------------------------------------------------------------------
# training distribution: logit(t) ~ N(P_mean, P_std)  =>  u ~ N(2 (P_mean - log s), 2 P_std)

def train_u_params(P_mean, P_std, s):
    return 2 * (P_mean - math.log(s)), 2 * P_std

def gaussian_pdf(u, m, sd):
    return torch.exp(-0.5 * ((u - m) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))

def expectation(curve, u, m, sd):
    w = gaussian_pdf(u, m, sd)
    return (curve * w).sum() / w.sum()

def fit_logit_normal(cdf, u, s):
    """
    Moment-matched logit-normal (P_mean, P_std) for the density d cdf / du, at noise scale s.
    """
    density = torch.diff(cdf).clamp_min(0)
    uc = 0.5 * (u[1:] + u[:-1])
    w = density / density.sum()
    mean_u = (w * uc).sum()
    sd_u = torch.sqrt((w * (uc - mean_u) ** 2).sum())
    return (mean_u / 2 + math.log(s)).item(), (sd_u / 2).item()

def crossing(curve, u, level):
    # u where a monotone curve first reaches level
    i = int(torch.searchsorted(curve.contiguous(), torch.tensor([level], dtype=curve.dtype)).item())
    i = min(max(i, 1), len(u) - 1)
    c0, c1 = curve[i - 1], curve[i]
    return (u[i - 1] + (level - c0) / (c1 - c0 + 1e-30) * (u[i] - u[i - 1])).item()

def match_shift(curve_new, curve_ref, u, m_new, sd_new, m_ref, sd_ref):
    """
    Shift delta (in u) of the new training distribution so that the expected curve value under it equals the
    reference's: E_{u ~ N(m_new + delta, sd_new)}[curve_new] = E_{u ~ N(m_ref, sd_ref)}[curve_ref]. Bisection (monotone in delta).
    """
    target = expectation(curve_ref, u, m_ref, sd_ref)
    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if expectation(curve_new, u, m_new + mid, sd_new) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), target.item()

# -----------------------------------------------------------------------------

def analyze(name, x, noise_scale, P_mean, P_std, quant_bits, value_range, n_bands=8):
    p = coefficient_variances(x)
    q = (value_range / (2 ** quant_bits - 1)) ** 2 / 12 if quant_bits else 1e-12
    u = torch.linspace(-30, 30, 6001, dtype=torch.float64)
    r2, info = r2_curve(p, u), info_curve(p, u, q)
    m, sd = train_u_params(P_mean, P_std, noise_scale)
    to_t = lambda uu: 1 / (1 + math.exp(-(uu / 2 + math.log(noise_scale)))) # u -> t at this noise scale

    print(f"\n=== {name}: {tuple(x.shape[1:])}, {len(x):,} samples, noise scale {noise_scale}, logit-normal({P_mean}, {P_std})")
    print(f"total variance {p.sum().item():.4f} over {len(p):,} coefficients, quantization floor q={q:.2e}")
    for label, curve in (('R^2 (coarse, power weighted)', r2), ('info (fine, count weighted)', info)):
        u_half = crossing(curve, u, 0.5)
        print(f"{label:32s} half-way at t={to_t(u_half):.3f} (logit {u_half / 2 + math.log(noise_scale):+.2f}); "
              f"E_train={expectation(curve, u, m, sd).item():.3f}")
    w = gaussian_pdf(u, m, sd); w = w / w.sum()
    print(f"training mass on near-trivial levels (R^2 > 0.99): {(w * (r2 > 0.99)).sum().item():.1%}, "
          f"on near-hopeless levels (info < 0.01): {(w * (info < 0.01)).sum().item():.1%}")
    for label, curve in (('R^2', r2), ('info', info)):
        mu_fit, sd_fit = fit_logit_normal(curve, u, noise_scale)
        print(f"logit-normal fit to the {label:4s} transition density: P_mean {mu_fit:+.2f}, P_std {sd_fit:.2f} (at noise scale {noise_scale}; a bound, not a recommendation)")

    # coarse-to-fine: where each radial frequency band's SNR crosses 1
    r = radial_frequency((x.shape[1],) + tuple(x.shape[2:]))
    print("band (radial freq / Nyquist)   SNR=1 at t    training mass noisier than that")
    for b in range(n_bands):
        lo, hi = b / n_bands, (b + 1) / n_bands
        sel = (r >= lo) & ((r < hi) if b < n_bands - 1 else torch.ones_like(r, dtype=torch.bool))
        if sel.sum() == 0:
            continue
        u_cross = -math.log(p[sel].mean().item()) # e^u * mean power = 1
        frac = 0.5 * (1 + math.erf((u_cross - m) / (sd * math.sqrt(2))))
        print(f"  [{lo:.3f}, {hi:.3f}{')' if b < n_bands - 1 else '+'}               {to_t(u_cross):.3f}        {frac:.1%}")
    return {'u': u, 'r2': r2, 'info': info, 'm': m, 'sd': sd}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='prepared JiT dataset dir, or .npy [N, C, *spatial] in model units')
    parser.add_argument('--noise_scale', type=float, default=0.0, help='0 -> img_size/256 for image dirs, 1.0 otherwise')
    parser.add_argument('--P_mean', type=float, default=-0.8)
    parser.add_argument('--P_std', type=float, default=0.8)
    parser.add_argument('--ref_data', type=str, default='', help='reference dataset with a known-good setting, e.g. in1k_256')
    parser.add_argument('--ref_noise_scale', type=float, default=1.0)
    parser.add_argument('--ref_P_mean', type=float, default=-0.8)
    parser.add_argument('--ref_P_std', type=float, default=0.8)
    parser.add_argument('--quant_bits', type=int, default=8, help='bit depth of the stored data (0 = continuous)')
    parser.add_argument('--value_range', type=float, default=2.0, help='range of the data in model units ([-1, 1] -> 2)')
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--plot', type=str, default='', help='optional path for a PNG of the curves')
    args = parser.parse_args()

    def default_scale(path, s):
        if s:
            return s
        if os.path.isdir(path):
            with open(os.path.join(path, 'meta.json')) as f:
                return json.load(f)['img_size'] / 256
        return 1.0

    s = default_scale(args.data, args.noise_scale)
    x = load_samples(args.data, args.max_samples)
    new = analyze(args.data, x, s, args.P_mean, args.P_std, args.quant_bits, args.value_range)

    if args.ref_data:
        xr = load_samples(args.ref_data, args.max_samples)
        ref = analyze(args.ref_data + ' (reference)', xr, args.ref_noise_scale, args.ref_P_mean, args.ref_P_std, args.quant_bits, args.value_range)
        print(f"\n=== matching {args.data} to the reference setting (P_std kept at {args.P_std})")
        if x.ndim == xr.ndim and x.shape[1] == xr.shape[1]:
            # shared-content match at the common resolution
            common = tuple(min(a, b) for a, b in zip(x.shape[2:], xr.shape[2:]))
            xp, f_new = pool_to(x, common)
            xrp, f_ref = pool_to(xr, common)
            u = torch.linspace(-30, 30, 6001, dtype=torch.float64)
            r2p_new, r2p_ref = r2_curve(coefficient_variances(xp), u), r2_curve(coefficient_variances(xrp), u)
            # pooled noise scales: s / sqrt(f). solve on the pooled curves, then map back to the original resolution
            m_new, sd_new = train_u_params(args.P_mean, args.P_std, s / math.sqrt(f_new))
            m_ref, sd_ref = train_u_params(args.ref_P_mean, args.ref_P_std, args.ref_noise_scale / math.sqrt(f_ref))
            delta, target = match_shift(r2p_new, r2p_ref, u, m_new, sd_new, m_ref, sd_ref)
            print(f"shared-content match at {common} (E_train R^2 = {target:.3f}): noise_scale {s * math.exp(-delta / 2):.3f} at P_mean {args.P_mean}, "
                  f"or P_mean {args.P_mean + delta / 2:+.2f} at noise_scale {s}   <- use this within a modality")
        for label in ('r2', 'info'):
            delta, target = match_shift(new[label], ref[label], new['u'], new['m'], new['sd'], ref['m'], ref['sd'])
            # shifting u by delta: P_mean -> P_mean + delta/2 at fixed s, or s -> s * exp(-delta/2) at fixed P_mean
            print(f"full-spectrum {'R^2' if label == 'r2' else 'info':4s} match (E_train = {target:.3f}): noise_scale {s * math.exp(-delta / 2):.3f} at P_mean {args.P_mean}, "
                  f"or P_mean {args.P_mean + delta / 2:+.2f} at noise_scale {s}")
        print("full-spectrum matches are heuristics (curve shapes differ); if they disagree, sweep between them")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        lt = new['u'] / 2 + math.log(s)
        ax.plot(lt, new['r2'], label='R^2 (coarse)')
        ax.plot(lt, new['info'], label='info revealed (fine)')
        dens = gaussian_pdf(new['u'], new['m'], new['sd'])
        ax.plot(lt, dens / dens.max(), '--', label='training density (scaled)')
        ax.set_xlim(-8, 8)
        ax.set_xlabel('logit(t)   (t=1 clean)')
        ax.legend()
        ax.set_title(f'{os.path.basename(args.data.rstrip("/"))}, noise scale {s}')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"plot saved to {args.plot}")
