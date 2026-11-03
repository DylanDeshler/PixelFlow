"""
Post-hoc EMA (Karras et al. 2024, "Analyzing and Improving the Training Dynamics of Diffusion Models", EDM2, section 3).

A power-function EMA averages the parameter trajectory p(tau), tau in [0, t] (t = optimizer steps so far), with weights
proportional to tau^gamma: its window is a fixed fraction of the training so far, not a fixed number of steps like JiT's
EMA (decay 0.9999 = ~10k steps whatever the run length). The window is described by sigma_rel, the standard deviation of
the weights over tau divided by t: sigma_rel^2 = (gamma + 1) / ((gamma + 2)^2 (gamma + 3)). Per step, with t counting
from 1: ema <- beta ema + (1 - beta) p, beta = (1 - 1 / t)^(gamma + 1).

Post-hoc: training keeps two such EMAs (e.g. sigma_rel 0.05 and 0.10) and saves snapshots of them every so often. Any
other power-function EMA (any sigma_rel, at any snapshot time) is then a linear combination of the snapshots: the
coefficients minimize the L2 distance between the combined weight profile and the target profile, a least-squares
problem in the profiles' inner products, which have a closed form (power_function_correlation). Snapshots already
integrate every step, so they can be sparse.

Training side: PowerEMA (train_jit.py --phema_stds / --phema_every_epochs). Post-hoc side: solve_coefficients, used by
posthoc_ema.py.
"""

import numpy as np
import torch


def std_to_exp(std):
    # sigma_rel -> gamma: the real root of gamma^3 + 7 gamma^2 + (16 - s) gamma + (12 - s) = 0, s = sigma_rel^-2
    std = np.asarray(std, dtype=np.float64)
    out = [max(r.real for r in np.roots([1, 7, 16 - s, 12 - s]) if abs(r.imag) < 1e-9) for s in (std.flatten() ** -2)]
    return np.array(out).reshape(std.shape)


def exp_to_std(gamma):
    gamma = np.asarray(gamma, dtype=np.float64)
    return np.sqrt((gamma + 1) / ((gamma + 2) ** 2 * (gamma + 3)))


def power_function_correlation(a_t, a_std, b_t, b_std):
    # inner product of two normalized power-function weight profiles (densities over tau): profile a ends at a_t with
    # sigma_rel a_std, b at b_t with b_std; integral over [0, min(a_t, b_t)] of (a+1) tau^a / a_t^(a+1) * (b+1) tau^b / b_t^(b+1)
    a = std_to_exp(a_std)
    b = std_to_exp(b_std)
    ratio = a_t / b_t
    e = np.where(a_t < b_t, b, -a)
    return (a + 1) * (b + 1) * ratio ** e / ((a + b + 1) * np.maximum(a_t, b_t))


def solve_coefficients(in_t, in_std, out_t, out_std):
    """
    Coefficients [len(in), len(out)]: out_k ~ sum_i X[i, k] * in_i, for snapshots taken at steps in_t of EMAs with
    sigma_rel in_std, and targets at steps out_t with sigma_rel out_std (broadcast against each other).
    """
    in_t, in_std = np.broadcast_arrays(np.float64(in_t), np.float64(in_std))
    out_t, out_std = np.broadcast_arrays(np.float64(out_t), np.float64(out_std))
    col, row = (lambda x: x.reshape(-1, 1)), (lambda x: x.reshape(1, -1))
    A = power_function_correlation(col(in_t), col(in_std), row(in_t), row(in_std))
    B = power_function_correlation(col(in_t), col(in_std), row(out_t), row(out_std))
    return np.linalg.lstsq(A, B, rcond=None)[0]


def fixed_ema_std(decay, t):
    # the sigma_rel a fixed-decay EMA (JiT's) amounts to after t steps: its weights over age are geometric with
    # standard deviation sqrt(decay) / (1 - decay) steps (ignoring the truncation at the start of training)
    return np.sqrt(decay) / (1 - decay) / t


class PowerEMA:
    """
    Power-function EMAs of a list of tensors, one per sigma_rel. update(params, t) after optimizer step t (t >= 1, global:
    a resumed run continues the count). state() / load_state() for checkpoints.
    """
    def __init__(self, params, stds):
        params = list(params) # (a generator would be used up by the first EMA)
        self.stds = [float(s) for s in stds]
        self.exps = [float(g) for g in std_to_exp(self.stds)]
        self.emas = [[p.detach().clone() for p in params] for _ in self.stds]

    @torch.no_grad()
    def update(self, params, t):
        params = list(params)
        for gamma, ema in zip(self.exps, self.emas):
            beta = (1 - 1 / t) ** (gamma + 1)
            torch._foreach_lerp_(ema, params, 1 - beta) # ema = beta * ema + (1 - beta) * param

    def state(self):
        return {'stds': self.stds, 'emas': [[e.detach().cpu() for e in ema] for ema in self.emas]}

    def load_state(self, state, device):
        assert [round(s, 6) for s in state['stds']] == [round(s, 6) for s in self.stds], \
            f"checkpoint has power EMAs {state['stds']}, run asks for {self.stds}"
        self.emas = [[e.to(device) for e in ema] for ema in state['emas']]
