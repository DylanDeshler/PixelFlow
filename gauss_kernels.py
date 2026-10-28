"""
Fused Triton kernels for GaussJiT's rotated frames (model_gauss.py). With rotation, a token's 2D cosine basis
psi_q(v) psi_a(u) isn't separable over the pixel grid, so the PyTorch path evaluates it at every pixel of every token's
window, materializing [tokens, pixels, n] tables (and their derivatives in the backward pass). These kernels evaluate the
basis in registers instead; everything cheap (frame coordinates, masks, window indices, Gaussian weights, gather /
scatter) stays in PyTorch, and autograd carries the geometry gradients from u, v back to the placer.

Two kernels, each used in a forward and a backward pass:
  project    out[t, c, q, a] = scale[t] * sum_p f[t, c, p] * psi_q(v[t, p]) * psi_a(u[t, p])
             (the encoder; also the gradient of evaluate w.r.t. its coefficients)
  evaluate   E[t, c, p] = sum_{q, a} coef[t, c, q, a] psi_q(v) psi_a(u), and optionally dE/du, dE/dv at each pixel
             (the decoder; also the gradients of project w.r.t. u, v and f)
with psi_a(z) = s_a cos(pi a (z + 1) / 2), s_0 = 1, s_a = sqrt 2 (model_gauss.GaussJiT.basis).

project / evaluate are registered as custom ops (gaussjit::project, gaussjit::evaluate, gaussjit::evaluate_deriv) with their
backward passes, so torch.compile treats each kernel as one opaque call and compiles everything around it; the reference
implementations (ref_project, ref_evaluate) are the plain PyTorch definitions, for testing and as the fallback when Triton
or CUDA isn't available.

$ python gauss_kernels.py        # checks the kernels (forward and gradients) against the references; on a GPU, also timings
                                  # (GPU dots run in TF32: expect ~1e-3 relative error there, ~1e-7 in the interpreter)
$ TRITON_INTERPRET=1 python gauss_kernels.py --device=cpu   # correctness on CPU through Triton's interpreter
"""

import math
import os

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError: # macOS / no Triton: the PyTorch path only
    HAVE_TRITON = False

INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'
SQRT2 = math.sqrt(2.0)
HALF_PI = math.pi / 2


def kernels_available(device):
    return HAVE_TRITON and (str(device).startswith('cuda') or INTERPRET)


# -----------------------------------------------------------------------------
# references (and the fallback)

def basis(z, n):
    a = torch.arange(n, device=z.device, dtype=torch.float32)
    s = torch.cat([torch.ones(1, device=z.device), torch.full((n - 1,), SQRT2, device=z.device)])
    return torch.cos(HALF_PI * a * (z[..., None] + 1)) * s

def ref_project(f, u, v, scale, n):
    # f [T, C, P], u / v [T, P], scale [T] -> [T, C, n, n] (rows q: v, columns a: u)
    return torch.einsum('tcp,tpq,tpa->tcqa', f, basis(v, n), basis(u, n)) * scale[:, None, None, None]

def ref_evaluate(coef, u, v, scale):
    # coef [T, C, n, n], u / v [T, P], scale [T] -> [T, C, P]
    n = coef.shape[-1]
    return torch.einsum('tcqa,tpq,tpa->tcp', coef, basis(v, n), basis(u, n)) * scale[:, None, None]


# -----------------------------------------------------------------------------
# kernels

if HAVE_TRITON:
    @triton.jit
    def _basis_tile(z, freq, s, valid):
        # psi_a(z) for a pixel block z [BP] and frequencies [BN] -> [BP, BN] (zero beyond n)
        b = tl.cos(1.5707963267948966 * freq[None, :] * (z[:, None] + 1.0)) * s[None, :]
        return tl.where(valid[None, :], b, 0.0)

    @triton.jit
    def _dbasis_tile(z, freq, s, valid):
        # d psi_a / dz -> [BP, BN]
        b = -tl.sin(1.5707963267948966 * freq[None, :] * (z[:, None] + 1.0)) * (1.5707963267948966 * freq[None, :]) * s[None, :]
        return tl.where(valid[None, :], b, 0.0)

    @triton.jit
    def _project_kernel(F, U, V, S, OUT, P, n, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        # one program per token, all three channels (the basis is evaluated once per pixel block)
        t = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        valid = offs_n < n
        freq = offs_n.to(tl.float32)
        s = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        acc0 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        for p0 in range(0, P, BLOCK_P):
            offs_p = p0 + tl.arange(0, BLOCK_P)
            mp = offs_p < P
            u = tl.load(U + t * P + offs_p, mask=mp, other=0.0)
            v = tl.load(V + t * P + offs_p, mask=mp, other=0.0)
            pu = _basis_tile(u, freq, s, valid)
            pv = tl.trans(_basis_tile(v, freq, s, valid))
            f0 = tl.load(F + (t * 3 + 0) * P + offs_p, mask=mp, other=0.0)
            f1 = tl.load(F + (t * 3 + 1) * P + offs_p, mask=mp, other=0.0)
            f2 = tl.load(F + (t * 3 + 2) * P + offs_p, mask=mp, other=0.0)
            acc0 += tl.dot(pv, pu * f0[:, None], input_precision=PREC)
            acc1 += tl.dot(pv, pu * f1[:, None], input_precision=PREC)
            acc2 += tl.dot(pv, pu * f2[:, None], input_precision=PREC)
        sc = tl.load(S + t)
        q = offs_n[:, None]
        a = offs_n[None, :]
        msk = (q < n) & (a < n)
        tl.store(OUT + ((t * 3 + 0) * n + q) * n + a, acc0 * sc, mask=msk)
        tl.store(OUT + ((t * 3 + 1) * n + q) * n + a, acc1 * sc, mask=msk)
        tl.store(OUT + ((t * 3 + 2) * n + q) * n + a, acc2 * sc, mask=msk)

    @triton.jit
    def _evaluate_channel(COEF, E, EU, EV, t, c, P, n, offs_n, offs_p, mp, pu, pv, dpu, dpv, DERIV: tl.constexpr, PREC: tl.constexpr):
        q = offs_n[:, None]
        a = offs_n[None, :]
        cm = tl.load(COEF + ((t * 3 + c) * n + q) * n + a, mask=(q < n) & (a < n), other=0.0) # [BN (q), BN (a)]
        tq = tl.dot(pv, cm, input_precision=PREC) # [BP, BN]: sum_q psi_q(v) coef[q, a]
        base = (t * 3 + c) * P + offs_p
        tl.store(E + base, tl.sum(tq * pu, axis=1), mask=mp)
        if DERIV:
            tl.store(EU + base, tl.sum(tq * dpu, axis=1), mask=mp)
            tl.store(EV + base, tl.sum(tl.dot(dpv, cm, input_precision=PREC) * pu, axis=1), mask=mp)

    @triton.jit
    def _evaluate_kernel(COEF, U, V, E, EU, EV, P, n, DERIV: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        # one program per (token, pixel block), all three channels (the basis is evaluated once)
        t = tl.program_id(0)
        pb = tl.program_id(1)
        offs_n = tl.arange(0, BLOCK_N)
        valid = offs_n < n
        freq = offs_n.to(tl.float32)
        s = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        offs_p = pb * BLOCK_P + tl.arange(0, BLOCK_P)
        mp = offs_p < P
        u = tl.load(U + t * P + offs_p, mask=mp, other=0.0)
        v = tl.load(V + t * P + offs_p, mask=mp, other=0.0)
        pu = _basis_tile(u, freq, s, valid)
        pv = _basis_tile(v, freq, s, valid)
        dpu = _dbasis_tile(u, freq, s, valid) if DERIV else pu
        dpv = _dbasis_tile(v, freq, s, valid) if DERIV else pv
        for c in tl.static_range(3):
            _evaluate_channel(COEF, E, EU, EV, t, c, P, n, offs_n, offs_p, mp, pu, pv, dpu, dpv, DERIV, PREC)

    # TF32 dots on GPU (the rest of the model trains in bf16); exact fp32 in the interpreter, or with GAUSS_KERNEL_PRECISION=ieee
    PRECISION = os.environ.get('GAUSS_KERNEL_PRECISION', 'ieee' if INTERPRET else 'tf32')

    def _launch_project(f, u, v, scale, n):
        T, C, P = f.shape
        assert C == 3, 'the kernels handle three channels per program'
        out = torch.empty(T, C, n, n, device=f.device, dtype=torch.float32)
        BN = max(16, triton.next_power_of_2(n))
        _project_kernel[(T,)](f.contiguous(), u.contiguous(), v.contiguous(), scale.contiguous(), out, P, n, BLOCK_P=64, BLOCK_N=BN, PREC=PRECISION)
        return out

    def _launch_evaluate(coef, u, v, deriv):
        T, C, n, _ = coef.shape
        P = u.shape[1]
        E = torch.empty(T, C, P, device=coef.device, dtype=torch.float32)
        EU = torch.empty_like(E) if deriv else E
        EV = torch.empty_like(E) if deriv else E
        assert C == 3, 'the kernels handle three channels per program'
        BN = max(16, triton.next_power_of_2(n))
        BP = 64
        _evaluate_kernel[(T, triton.cdiv(P, BP))](coef.contiguous(), u.contiguous(), v.contiguous(), E, EU, EV, P, n,
                                                   DERIV=deriv, BLOCK_P=BP, BLOCK_N=BN, PREC=PRECISION)
        return (E, EU, EV) if deriv else E


# custom ops (opaque to torch.compile, which compiles and fuses everything around them), with their backward passes

if HAVE_TRITON:
    @torch.library.custom_op('gaussjit::project', mutates_args=())
    def project_op(f: torch.Tensor, u: torch.Tensor, v: torch.Tensor, scale: torch.Tensor, n: int) -> torch.Tensor:
        return _launch_project(f.float(), u.float(), v.float(), scale.float(), n)

    @project_op.register_fake
    def _(f, u, v, scale, n):
        return f.new_empty((f.shape[0], f.shape[1], n, n), dtype=torch.float32)

    @torch.library.custom_op('gaussjit::evaluate', mutates_args=())
    def evaluate_op(coef: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return _launch_evaluate(coef.float(), u.float(), v.float(), False)

    @evaluate_op.register_fake
    def _(coef, u, v):
        return coef.new_empty((coef.shape[0], coef.shape[1], u.shape[1]), dtype=torch.float32)

    @torch.library.custom_op('gaussjit::evaluate_deriv', mutates_args=())
    def evaluate_deriv_op(coef: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _launch_evaluate(coef.float(), u.float(), v.float(), True)

    @evaluate_deriv_op.register_fake
    def _(coef, u, v):
        shape = (coef.shape[0], coef.shape[1], u.shape[1])
        return coef.new_empty(shape, dtype=torch.float32), coef.new_empty(shape, dtype=torch.float32), coef.new_empty(shape, dtype=torch.float32)

    def _project_setup(ctx, inputs, output):
        f, u, v, scale, n = inputs
        ctx.save_for_backward(f, u, v, scale)

    def _project_backward(ctx, g):
        f, u, v, scale = ctx.saved_tensors
        E, EU, EV = evaluate_deriv_op(g.contiguous(), u, v) # sum_{q,a} g psi(v) psi(u) and its u / v derivatives per pixel
        sc = scale[:, None]
        return E * scale[:, None, None], (f * EU).sum(1) * sc, (f * EV).sum(1) * sc, None, None

    project_op.register_autograd(_project_backward, setup_context=_project_setup)

    def _evaluate_setup(ctx, inputs, output):
        coef, u, v = inputs
        ctx.save_for_backward(coef, u, v)

    def _evaluate_backward(ctx, g):
        coef, u, v = ctx.saved_tensors
        g = g.contiguous()
        dcoef = project_op(g, u, v, torch.ones(u.shape[0], device=u.device), coef.shape[-1])
        _, EU, EV = evaluate_deriv_op(coef, u, v)
        return dcoef, (g * EU).sum(1), (g * EV).sum(1)

    evaluate_op.register_autograd(_evaluate_backward, setup_context=_evaluate_setup)


def project(f, u, v, scale, n):
    # [T, C, P] pixels -> [T, C, n, n] coefficients (scaled per token); differentiable in f, u, v
    return project_op(f, u, v, scale, n) if kernels_available(f.device) else ref_project(f, u, v, scale, n)

def evaluate(coef, u, v, scale):
    # [T, C, n, n] coefficients -> [T, C, P] values (scaled per token); differentiable in coef, u, v
    return evaluate_op(coef, u, v) * scale[:, None, None] if kernels_available(coef.device) else ref_evaluate(coef, u, v, scale)


# -----------------------------------------------------------------------------
# fully fused input / output layers: per-token geometry in, per-token geometry gradients out, nothing per pixel stored
#
# A token's geometry is 5 numbers g = (mu_x, mu_y, h_x, h_y, theta). For each pixel of its W x W window (whole pixels
# around floor(mu)), the kernels compute the frame coordinates u, v, the masks, the pixel reads (reflected past the image
# borders), the Gaussian weights and the image accumulation themselves, and recompute them in the backward pass:
#   encode   coef[t, c] = sum_{p in frame} x[c, p] psi(v_p) psi(u_p) / sqrt(N_t)          (N_t: pixels in the frame)
#   render   num[c, p] += w_tp val_tc(p), den[p] += w_tp,  val = sum coef psi(v) psi(u) / n, w = exp(-(u^2+v^2) / 2 sigma^2)
# Gradients: encode w.r.t. g (x is the noisy input: no gradient); render w.r.t. coef and g. The frame mask and the window
# placement are piecewise constant in g (no gradient), as in the PyTorch path.

if HAVE_TRITON:
    @triton.jit
    def _frame(t, l, W, Hh, G):
        # window pixel l of token t: frame coordinates u, v [BP], pixel indices ix, iy, and the token's geometry
        mux = tl.load(G + t * 5 + 0); muy = tl.load(G + t * 5 + 1)
        hx = tl.load(G + t * 5 + 2); hy = tl.load(G + t * 5 + 3); th = tl.load(G + t * 5 + 4)
        c = tl.cos(th); s = tl.sin(th)
        R = (W - 1) // 2
        ix = tl.floor(mux).to(tl.int32) + (l % W) - R
        iy = tl.floor(muy).to(tl.int32) + (l // W) - R
        dx = ix.to(tl.float32) + 0.5 - mux
        dy = iy.to(tl.float32) + 0.5 - muy
        u = (c * dx + s * dy) / hx
        v = (-s * dx + c * dy) / hy
        return u, v, ix, iy, hx, hy, c, s

    @triton.jit
    def _geom_grads(du, dv, u, v, hx, hy, c, s):
        # per-pixel dL/du, dL/dv -> this block's contributions to dL/d(mu_x, mu_y, h_x, h_y, theta)
        gmx = tl.sum(du * (-c / hx) + dv * (s / hy))
        gmy = tl.sum(du * (-s / hx) + dv * (-c / hy))
        ghx = tl.sum(du * (-u / hx))
        ghy = tl.sum(dv * (-v / hy))
        gth = tl.sum(du * (v * hy / hx) - dv * (u * hx / hy))
        return gmx, gmy, ghx, ghy, gth

    @triton.jit
    def _reflect(i, Hh):
        i = tl.where(i < 0, -i - 1, i)
        i = tl.where(i >= Hh, 2 * Hh - 1 - i, i)
        return tl.minimum(tl.maximum(i, 0), Hh - 1)

    @triton.jit
    def _encode_fwd_kernel(X, G, OUT, NCOUNT, K, Hh, W, n, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        acc0 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        cnt = tl.zeros((BLOCK_P,), dtype=tl.float32)
        for l0 in range(0, W * W, BLOCK_P):
            l = l0 + tl.arange(0, BLOCK_P)
            u, v, ix, iy, hx, hy, c, s = _frame(t, l, W, Hh, G)
            m = (l < W * W) & (tl.abs(u) < 1.0) & (tl.abs(v) < 1.0)
            off = _reflect(iy, Hh) * Hh + _reflect(ix, Hh)
            base = X + b * 3 * Hh * Hh
            x0 = tl.load(base + off, mask=m, other=0.0)
            x1 = tl.load(base + Hh * Hh + off, mask=m, other=0.0)
            x2 = tl.load(base + 2 * Hh * Hh + off, mask=m, other=0.0)
            pu = _basis_tile(u, freq, sn, valid_n)
            pv = tl.trans(_basis_tile(v, freq, sn, valid_n))
            acc0 += tl.dot(pv, pu * x0[:, None], input_precision=PREC)
            acc1 += tl.dot(pv, pu * x1[:, None], input_precision=PREC)
            acc2 += tl.dot(pv, pu * x2[:, None], input_precision=PREC)
            cnt += m.to(tl.float32)
        N = tl.maximum(tl.sum(cnt), 1.0)
        tl.store(NCOUNT + t, N)
        sc = 1.0 / tl.sqrt(N)
        q = offs_n[:, None]
        a = offs_n[None, :]
        msk = (q < n) & (a < n)
        tl.store(OUT + ((t * 3 + 0) * n + q) * n + a, acc0 * sc, mask=msk)
        tl.store(OUT + ((t * 3 + 1) * n + q) * n + a, acc1 * sc, mask=msk)
        tl.store(OUT + ((t * 3 + 2) * n + q) * n + a, acc2 * sc, mask=msk)

    @triton.jit
    def _load_tile(PTR, t, ch, n, offs_n):
        q = offs_n[:, None]
        a = offs_n[None, :]
        return tl.load(PTR + ((t * 3 + ch) * n + q) * n + a, mask=(q < n) & (a < n), other=0.0)

    @triton.jit
    def _encode_bwd_kernel(X, G, GC, NCOUNT, DG, K, Hh, W, n, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        sc = 1.0 / tl.sqrt(tl.load(NCOUNT + t))
        g0 = _load_tile(GC, t, 0, n, offs_n) * sc
        g1 = _load_tile(GC, t, 1, n, offs_n) * sc
        g2 = _load_tile(GC, t, 2, n, offs_n) * sc
        gmx = 0.0; gmy = 0.0; ghx = 0.0; ghy = 0.0; gth = 0.0
        for l0 in range(0, W * W, BLOCK_P):
            l = l0 + tl.arange(0, BLOCK_P)
            u, v, ix, iy, hx, hy, c, s = _frame(t, l, W, Hh, G)
            m = (l < W * W) & (tl.abs(u) < 1.0) & (tl.abs(v) < 1.0)
            off = _reflect(iy, Hh) * Hh + _reflect(ix, Hh)
            base = X + b * 3 * Hh * Hh
            x0 = tl.load(base + off, mask=m, other=0.0)
            x1 = tl.load(base + Hh * Hh + off, mask=m, other=0.0)
            x2 = tl.load(base + 2 * Hh * Hh + off, mask=m, other=0.0)
            pu = _basis_tile(u, freq, sn, valid_n); pv = _basis_tile(v, freq, sn, valid_n)
            dpu = _dbasis_tile(u, freq, sn, valid_n); dpv = _dbasis_tile(v, freq, sn, valid_n)
            # dL/du_p = sum_c x_c sum_{q,a} G_c psi_q(v) psi_a'(u); likewise v
            du = x0 * tl.sum(tl.dot(pv, g0, input_precision=PREC) * dpu, axis=1)
            du += x1 * tl.sum(tl.dot(pv, g1, input_precision=PREC) * dpu, axis=1)
            du += x2 * tl.sum(tl.dot(pv, g2, input_precision=PREC) * dpu, axis=1)
            dv = x0 * tl.sum(tl.dot(dpv, g0, input_precision=PREC) * pu, axis=1)
            dv += x1 * tl.sum(tl.dot(dpv, g1, input_precision=PREC) * pu, axis=1)
            dv += x2 * tl.sum(tl.dot(dpv, g2, input_precision=PREC) * pu, axis=1)
            a0, a1, a2, a3, a4 = _geom_grads(du, dv, u, v, hx, hy, c, s)
            gmx += a0; gmy += a1; ghx += a2; ghy += a3; gth += a4
        tl.store(DG + t * 5 + 0, gmx); tl.store(DG + t * 5 + 1, gmy)
        tl.store(DG + t * 5 + 2, ghx); tl.store(DG + t * 5 + 3, ghy); tl.store(DG + t * 5 + 4, gth)

    @triton.jit
    def _render_fwd_kernel(COEF, G, NUM, DEN, K, Hh, W, n, inv2s2, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        c0 = _load_tile(COEF, t, 0, n, offs_n); c1 = _load_tile(COEF, t, 1, n, offs_n); c2 = _load_tile(COEF, t, 2, n, offs_n)
        inv_n = 1.0 / n
        for l0 in range(0, W * W, BLOCK_P):
            l = l0 + tl.arange(0, BLOCK_P)
            u, v, ix, iy, hx, hy, c, s = _frame(t, l, W, Hh, G)
            inside = (l < W * W) & (ix >= 0) & (ix < Hh) & (iy >= 0) & (iy < Hh)
            w = tl.exp(-(u * u + v * v) * inv2s2)
            pu = _basis_tile(u, freq, sn, valid_n); pv = _basis_tile(v, freq, sn, valid_n)
            off = iy * Hh + ix
            nb = NUM + b * 3 * Hh * Hh
            tl.atomic_add(nb + off, w * tl.sum(tl.dot(pv, c0, input_precision=PREC) * pu, axis=1) * inv_n, mask=inside)
            tl.atomic_add(nb + Hh * Hh + off, w * tl.sum(tl.dot(pv, c1, input_precision=PREC) * pu, axis=1) * inv_n, mask=inside)
            tl.atomic_add(nb + 2 * Hh * Hh + off, w * tl.sum(tl.dot(pv, c2, input_precision=PREC) * pu, axis=1) * inv_n, mask=inside)
            tl.atomic_add(DEN + b * Hh * Hh + off, w, mask=inside)

    @triton.jit
    def _render_bwd_kernel(COEF, G, DNUM, DDEN, DCOEF, DG, K, Hh, W, n, inv2s2, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        c0 = _load_tile(COEF, t, 0, n, offs_n); c1 = _load_tile(COEF, t, 1, n, offs_n); c2 = _load_tile(COEF, t, 2, n, offs_n)
        inv_n = 1.0 / n
        acc0 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        gmx = 0.0; gmy = 0.0; ghx = 0.0; ghy = 0.0; gth = 0.0
        for l0 in range(0, W * W, BLOCK_P):
            l = l0 + tl.arange(0, BLOCK_P)
            u, v, ix, iy, hx, hy, c, s = _frame(t, l, W, Hh, G)
            inside = (l < W * W) & (ix >= 0) & (ix < Hh) & (iy >= 0) & (iy < Hh)
            w = tl.where(inside, tl.exp(-(u * u + v * v) * inv2s2), 0.0)
            pu = _basis_tile(u, freq, sn, valid_n); pv = _basis_tile(v, freq, sn, valid_n)
            dpu = _dbasis_tile(u, freq, sn, valid_n); dpv = _dbasis_tile(v, freq, sn, valid_n)
            off = iy * Hh + ix
            nb = DNUM + b * 3 * Hh * Hh
            n0 = tl.load(nb + off, mask=inside, other=0.0)
            n1 = tl.load(nb + Hh * Hh + off, mask=inside, other=0.0)
            n2 = tl.load(nb + 2 * Hh * Hh + off, mask=inside, other=0.0)
            dd = tl.load(DDEN + b * Hh * Hh + off, mask=inside, other=0.0)
            t0 = tl.dot(pv, c0, input_precision=PREC); t1 = tl.dot(pv, c1, input_precision=PREC); t2 = tl.dot(pv, c2, input_precision=PREC)
            # dL/dw_p = sum_c dnum_c val_c + dden
            dw = dd + inv_n * (n0 * tl.sum(t0 * pu, axis=1) + n1 * tl.sum(t1 * pu, axis=1) + n2 * tl.sum(t2 * pu, axis=1))
            # through the values: dL/du_p = sum_c dnum_c w dval_c/du; then through the weights: dw/du = -u w / sigma^2
            du = w * inv_n * (n0 * tl.sum(t0 * dpu, axis=1) + n1 * tl.sum(t1 * dpu, axis=1) + n2 * tl.sum(t2 * dpu, axis=1))
            dv = w * inv_n * (n0 * tl.sum(tl.dot(dpv, c0, input_precision=PREC) * pu, axis=1)
                              + n1 * tl.sum(tl.dot(dpv, c1, input_precision=PREC) * pu, axis=1)
                              + n2 * tl.sum(tl.dot(dpv, c2, input_precision=PREC) * pu, axis=1))
            du += dw * (-2.0 * inv2s2) * u * w
            dv += dw * (-2.0 * inv2s2) * v * w
            # dL/dcoef_c = sum_p dnum_c w psi(v) psi(u) / n
            pvt = tl.trans(pv)
            acc0 += tl.dot(pvt, pu * (n0 * w * inv_n)[:, None], input_precision=PREC)
            acc1 += tl.dot(pvt, pu * (n1 * w * inv_n)[:, None], input_precision=PREC)
            acc2 += tl.dot(pvt, pu * (n2 * w * inv_n)[:, None], input_precision=PREC)
            a0, a1, a2, a3, a4 = _geom_grads(du, dv, u, v, hx, hy, c, s)
            gmx += a0; gmy += a1; ghx += a2; ghy += a3; gth += a4
        q = offs_n[:, None]
        a = offs_n[None, :]
        msk = (q < n) & (a < n)
        tl.store(DCOEF + ((t * 3 + 0) * n + q) * n + a, acc0, mask=msk)
        tl.store(DCOEF + ((t * 3 + 1) * n + q) * n + a, acc1, mask=msk)
        tl.store(DCOEF + ((t * 3 + 2) * n + q) * n + a, acc2, mask=msk)
        tl.store(DG + t * 5 + 0, gmx); tl.store(DG + t * 5 + 1, gmy)
        tl.store(DG + t * 5 + 2, ghx); tl.store(DG + t * 5 + 3, ghy); tl.store(DG + t * 5 + 4, gth)

    BP_FUSED = 64

    @torch.library.custom_op('gaussjit::encode', mutates_args=())
    def encode_op(x: torch.Tensor, geom: torch.Tensor, K: int, n: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, geom = x.float().contiguous(), geom.float().contiguous()
        T, Hh = geom.shape[0], x.shape[-1]
        out = torch.empty(T, 3, n, n, device=x.device, dtype=torch.float32)
        N = torch.empty(T, device=x.device, dtype=torch.float32)
        _encode_fwd_kernel[(T,)](x, geom, out, N, K, Hh, W, n, BLOCK_P=BP_FUSED, BLOCK_N=max(16, triton.next_power_of_2(n)), PREC=PRECISION)
        return out, N

    @encode_op.register_fake
    def _(x, geom, K, n, W):
        return x.new_empty((geom.shape[0], 3, n, n), dtype=torch.float32), x.new_empty((geom.shape[0],), dtype=torch.float32)

    @torch.library.custom_op('gaussjit::encode_bwd', mutates_args=())
    def encode_bwd_op(x: torch.Tensor, geom: torch.Tensor, gcoef: torch.Tensor, N: torch.Tensor, K: int, n: int, W: int) -> torch.Tensor:
        x, geom = x.float().contiguous(), geom.float().contiguous()
        T, Hh = geom.shape[0], x.shape[-1]
        dg = torch.empty(T, 5, device=x.device, dtype=torch.float32)
        _encode_bwd_kernel[(T,)](x, geom, gcoef.float().contiguous(), N, dg, K, Hh, W, n, BLOCK_P=BP_FUSED,
                                 BLOCK_N=max(16, triton.next_power_of_2(n)), PREC=PRECISION)
        return dg

    @encode_bwd_op.register_fake
    def _(x, geom, gcoef, N, K, n, W):
        return geom.new_empty((geom.shape[0], 5), dtype=torch.float32)

    def _encode_setup(ctx, inputs, output):
        x, geom, K, n, W = inputs
        ctx.save_for_backward(x, geom, output[1])
        ctx.K, ctx.n, ctx.W = K, n, W

    def _encode_backward(ctx, gcoef, gN):
        x, geom, N = ctx.saved_tensors
        return None, encode_bwd_op(x, geom, gcoef, N, ctx.K, ctx.n, ctx.W), None, None, None

    encode_op.register_autograd(_encode_backward, setup_context=_encode_setup)

    @torch.library.custom_op('gaussjit::render', mutates_args=())
    def render_op(coef: torch.Tensor, geom: torch.Tensor, B: int, Hh: int, W: int, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
        coef, geom = coef.float().contiguous(), geom.float().contiguous()
        T, n = geom.shape[0], coef.shape[-1]
        num = torch.zeros(B, 3, Hh, Hh, device=coef.device, dtype=torch.float32)
        den = torch.zeros(B, 1, Hh, Hh, device=coef.device, dtype=torch.float32)
        _render_fwd_kernel[(T,)](coef, geom, num, den, T // B, Hh, W, n, 1.0 / (2 * sigma * sigma), BLOCK_P=BP_FUSED,
                                 BLOCK_N=max(16, triton.next_power_of_2(n)), PREC=PRECISION)
        return num, den

    @render_op.register_fake
    def _(coef, geom, B, Hh, W, sigma):
        return coef.new_empty((B, 3, Hh, Hh), dtype=torch.float32), coef.new_empty((B, 1, Hh, Hh), dtype=torch.float32)

    @torch.library.custom_op('gaussjit::render_bwd', mutates_args=())
    def render_bwd_op(coef: torch.Tensor, geom: torch.Tensor, dnum: torch.Tensor, dden: torch.Tensor, W: int, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
        coef, geom = coef.float().contiguous(), geom.float().contiguous()
        T, n = geom.shape[0], coef.shape[-1]
        B, Hh = dnum.shape[0], dnum.shape[-1]
        dcoef = torch.empty_like(coef)
        dg = torch.empty(T, 5, device=coef.device, dtype=torch.float32)
        _render_bwd_kernel[(T,)](coef, geom, dnum.float().contiguous(), dden.float().contiguous(), dcoef, dg, T // B, Hh, W, n,
                                 1.0 / (2 * sigma * sigma), BLOCK_P=BP_FUSED, BLOCK_N=max(16, triton.next_power_of_2(n)), PREC=PRECISION)
        return dcoef, dg

    @render_bwd_op.register_fake
    def _(coef, geom, dnum, dden, W, sigma):
        # contiguous, as the real op returns (empty_like would copy a permuted input's strides)
        return coef.new_empty(coef.shape, dtype=torch.float32), geom.new_empty((geom.shape[0], 5), dtype=torch.float32)

    def _render_setup(ctx, inputs, output):
        coef, geom, B, Hh, W, sigma = inputs
        ctx.save_for_backward(coef, geom)
        ctx.W, ctx.sigma = W, sigma

    def _render_backward(ctx, dnum, dden):
        coef, geom = ctx.saved_tensors
        dnum = torch.zeros_like(coef[:0].new_empty(0)) if dnum is None else dnum
        dcoef, dg = render_bwd_op(coef, geom, dnum, dden, ctx.W, ctx.sigma)
        return dcoef, dg, None, None, None, None

    render_op.register_autograd(_render_backward, setup_context=_render_setup)


def ref_encode(x, geom, K, n, W):
    """
    PyTorch reference for encode_op: x [B, 3, H, H], geom [T = B K, 5] -> (coef [T, 3, n, n], N [T]).
    """
    B, C, Hh = x.shape[0], x.shape[1], x.shape[-1]
    T = geom.shape[0]
    mux, muy, hx, hy, th = geom.unbind(-1)
    R = (W - 1) // 2
    l = torch.arange(W * W, device=x.device)
    ix = torch.floor(mux).long()[:, None] + (l % W) - R
    iy = torch.floor(muy).long()[:, None] + (l // W) - R
    dx, dy = ix + 0.5 - mux[:, None], iy + 0.5 - muy[:, None]
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
    u, v = (c * dx + s * dy) / hx[:, None], (-s * dx + c * dy) / hy[:, None]
    m = ((u.abs() < 1) & (v.abs() < 1)).float()
    refl = lambda i: torch.where(i >= Hh, 2 * Hh - 1 - torch.where(i < 0, -i - 1, i), torch.where(i < 0, -i - 1, i)).clamp(0, Hh - 1)
    off = refl(iy) * Hh + refl(ix)
    b = torch.arange(T, device=x.device) // K
    vals = x.reshape(B, C, -1)[b[:, None], :, off].permute(0, 2, 1) * m[:, None] # [T, C, P]
    N = m.sum(-1).clamp_min(1)
    return ref_project(vals, u, v, N.rsqrt(), n), N

def ref_render(coef, geom, B, Hh, W, sigma):
    """
    PyTorch reference for render_op: coef [T, 3, n, n], geom [T, 5] -> (num [B, 3, H, H], den [B, 1, H, H]).
    """
    T, n = geom.shape[0], coef.shape[-1]
    mux, muy, hx, hy, th = geom.unbind(-1)
    R = (W - 1) // 2
    l = torch.arange(W * W, device=coef.device)
    ix = torch.floor(mux).long()[:, None] + (l % W) - R
    iy = torch.floor(muy).long()[:, None] + (l // W) - R
    dx, dy = ix + 0.5 - mux[:, None], iy + 0.5 - muy[:, None]
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
    u, v = (c * dx + s * dy) / hx[:, None], (-s * dx + c * dy) / hy[:, None]
    inside = ((ix >= 0) & (ix < Hh) & (iy >= 0) & (iy < Hh)).float()
    w = torch.exp(-(u ** 2 + v ** 2) / (2 * sigma ** 2)) * inside
    vals = ref_evaluate(coef, u, v, torch.full((T,), 1.0 / n, device=coef.device)) # [T, 3, P]
    K = T // B
    off = (iy.clamp(0, Hh - 1) * Hh + ix.clamp(0, Hh - 1)).view(B, K * W * W)
    num = torch.zeros(B, 3, Hh * Hh, device=coef.device).scatter_add_(2, off[:, None].expand(B, 3, -1), (vals * w[:, None]).view(B, K, 3, -1).transpose(1, 2).reshape(B, 3, -1))
    den = torch.zeros(B, 1, Hh * Hh, device=coef.device).scatter_add_(2, off[:, None], w.view(B, 1, -1))
    return num.view(B, 3, Hh, Hh), den.view(B, 1, Hh, Hh)

def fused_available(device):
    return kernels_available(device)


# -----------------------------------------------------------------------------
# self-test

if __name__ == '__main__':
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--tokens', type=int, default=24)
    ap.add_argument('--pixels', type=int, default=300)
    ap.add_argument('--n', type=int, default=24)
    args = ap.parse_args()
    assert kernels_available(args.device), 'needs Triton and CUDA (or TRITON_INTERPRET=1 for CPU)'
    torch.manual_seed(0)
    T, C, P, n, dev = args.tokens, 3, args.pixels, args.n, args.device
    u = (torch.rand(T, P, device=dev) * 2.4 - 1.2).requires_grad_(True)
    v = (torch.rand(T, P, device=dev) * 2.4 - 1.2).requires_grad_(True)
    f = torch.randn(T, C, P, device=dev, requires_grad=True)
    coef = torch.randn(T, C, n, n, device=dev, requires_grad=True)
    scale = torch.rand(T, device=dev) + 0.5
    def check(name, out_k, out_r, inputs):
        g = torch.randn_like(out_r)
        gk = torch.autograd.grad(out_k, inputs, g, allow_unused=True)
        gr = torch.autograd.grad(out_r, inputs, g, allow_unused=True)
        rel = lambda a, b: ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()
        print(f'{name}: forward rel err {rel(out_k, out_r):.2e} | ' + ', '.join(
            f'grad {nm} rel err {rel(a, b):.2e}' for nm, a, b in zip(('1st', '2nd', '3rd'), gk, gr) if a is not None and b is not None))
    check('project ', project(f, u, v, scale, n), ref_project(f, u, v, scale, n), (f, u, v))
    check('evaluate', evaluate(coef, u, v, scale), ref_evaluate(coef, u, v, scale), (coef, u, v))
    # fully fused encode / render vs their references (geometry gradients included)
    B_, K_, Hh, Wd = 2, 4, 32, 21
    x_img = torch.randn(B_, 3, Hh, Hh, device=dev)
    anchors = torch.tensor([[8., 8.], [24., 8.], [8., 24.], [24., 24.]], device=dev).repeat(B_, 1)
    geom = torch.cat([anchors + torch.rand(B_ * K_, 2, device=dev) * 4 - 2, 6 + torch.rand(B_ * K_, 2, device=dev) * 3,
                      torch.rand(B_ * K_, 1, device=dev) * 1.4 - 0.7], 1).requires_grad_(True)
    ck, Nk = encode_op(x_img, geom, K_, n, Wd)
    cr, Nr = ref_encode(x_img, geom, K_, n, Wd)
    print('encode  : N match', torch.equal(Nk, Nr), end=' | ')
    check('encode  ', ck, cr, (geom,))
    cf = torch.randn(B_ * K_, 3, n, n, device=dev, requires_grad=True)
    nk, dk = render_op(cf, geom, B_, Hh, Wd, 0.3)
    nr, dr = ref_render(cf, geom, B_, Hh, Wd, 0.3)
    img_k, img_r = nk / dk.clamp_min(1e-30), nr / dr.clamp_min(1e-30) # the model's use: num / den
    check('render  ', img_k, img_r, (cf, geom))
    if dev.startswith('cuda') and not INTERPRET: # timings at GaussJiT-B/16's sizes (batch 64: 4096 tokens, 2401 / 1521 pixels)
        T, P = 64 * 64, 2401
        u, v = (torch.rand(T, P, device=dev) * 2 - 1).requires_grad_(True), (torch.rand(T, P, device=dev) * 2 - 1).requires_grad_(True)
        f = torch.randn(T, C, P, device=dev)
        coef = torch.randn(T, C, n, n, device=dev, requires_grad=True)
        scale = torch.ones(T, device=dev)
        for name, fn in (('kernel', lambda: project(f, u, v, scale, n)), ('torch ', lambda: ref_project(f, u, v, scale, n))):
            for label, run in (('fwd', lambda: fn()), ('fwd+bwd', lambda: torch.autograd.grad(fn().sum(), (u, v)))):
                run(); torch.cuda.synchronize(); t0 = time.time()
                for _ in range(5):
                    run()
                torch.cuda.synchronize()
                print(f'project  {name} {label}: {(time.time() - t0) / 5 * 1000:.1f} ms (batch 64)')
