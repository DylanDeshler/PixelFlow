"""
Fused Triton kernels for GaussJiT's input and output layers (model_gauss.py).

The model uses the fully fused ops (encode_op, render_op, below): per token, geometry in and geometry gradients out, the
window, masks, pixel reads, basis, Gaussian weights and image accumulation all in registers, over the token's own bounding
box, with separable kernels for axis-aligned frames (row then column matmuls) and per-pixel 2D basis evaluation for rotated
ones. Nothing per pixel is stored for the backward pass.

The basis kernels (project, evaluate) are the earlier, partly fused version: they evaluate the 2D basis in registers while
everything cheap (frame coordinates, masks, window indices, Gaussian weights, gather / scatter) stays in PyTorch, and
autograd carries the geometry gradients from u, v back to the placer. Two kernels, each used in a forward and a backward
pass:
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
                                  # (GPU dots run in TF32: expect ~1e-3 relative error there, ~1e-7 in the interpreter;
                                  # GAUSS_KERNEL_PRECISION=bf16 is ~5-10% faster, ~1e-2 error)
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
    def _mm(a, b, PREC: tl.constexpr):
        # a @ b with fp32 accumulation: TF32 / IEEE fp32 inputs, or bf16 inputs (PREC='bf16': twice the TF32 tensor-core rate)
        if PREC == 'bf16':
            return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        else:
            return tl.dot(a, b, input_precision=PREC)

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
            acc0 += _mm(pv, pu * f0[:, None], PREC)
            acc1 += _mm(pv, pu * f1[:, None], PREC)
            acc2 += _mm(pv, pu * f2[:, None], PREC)
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
        tq = _mm(pv, cm, PREC) # [BP, BN]: sum_q psi_q(v) coef[q, a]
        base = (t * 3 + c) * P + offs_p
        tl.store(E + base, tl.sum(tq * pu, axis=1), mask=mp)
        if DERIV:
            tl.store(EU + base, tl.sum(tq * dpu, axis=1), mask=mp)
            tl.store(EV + base, tl.sum(_mm(dpv, cm, PREC) * pu, axis=1), mask=mp)

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

    # TF32 dots on GPU (the rest of the model trains in bf16); exact fp32 in the interpreter, or with GAUSS_KERNEL_PRECISION=ieee;
    # GAUSS_KERNEL_PRECISION=bf16: bf16 inputs (fp32 accumulation)
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
# A token's geometry is 5 numbers g = (mu_x, mu_y, h_x, h_y, theta). For each pixel of its window (whole pixels around
# floor(mu)), the kernels compute the frame coordinates u, v, the masks, the pixel reads (reflected past the image
# borders), the Gaussian weights and the image accumulation themselves, and recompute them in the backward pass:
#   encode   coef[t, c] = sum_{p in frame} x[c, p] psi(v_p) psi(u_p) / sqrt(N_t)          (N_t: pixels in the frame)
#   render   num[c, p] += w_tp val_tc(p), den[p] += w_tp,  val = sum coef psi(v) psi(u) / n, w = exp(-(u^2+v^2) / 2 sigma^2)
# Gradients: encode w.r.t. g (x is the noisy input: no gradient); render w.r.t. coef and g. The frame mask and the window
# placement are piecewise constant in g (no gradient), as in the PyTorch path.
#
# Each token visits only its own bounding box (radii rx, ry around floor(mu), at most the model's window radius R):
#   encode   the frame's box: r = ceil(e) + 1, e = |cos| h_x + |sin| h_y (x; likewise y)  -- every pixel with |u|, |v| < 1
#   render   the Gaussian's box out to u^2 + v^2 = u_cut^2 (weights below 1e-3), at least r_own (the distance to the pixels
#            the token may own, so every pixel stays covered)                              -- render_radius
# Axis-aligned frames (theta = 0, axis=True) are separable: the window is processed in tiles [BY rows, BX columns] with the
# 1D bases psi(u) [BX, n], psi(v) [BY, n], so a tile costs matmuls of [BY, BX] x [BX, n] instead of n^2 per pixel.

def render_cut(sigma):
    # the frame radius past which the blending weights are dropped (exp(-u_cut^2 / 2 sigma^2) = 1e-3)
    return sigma * math.sqrt(2 * math.log(1e3))

def render_radius(geom, sigma, r_own, R):
    # per token, the render's box radii (rx, ry) [T] around floor(mu) (the kernels compute the same in-kernel)
    hx, hy, th = geom[:, 2], geom[:, 3], geom[:, 4]
    c, s, uc = torch.cos(th), torch.sin(th), render_cut(sigma)
    ex = uc * torch.sqrt((hx * c) ** 2 + (hy * s) ** 2)
    ey = uc * torch.sqrt((hx * s) ** 2 + (hy * c) ** 2)
    rad = lambda e: (torch.ceil(e) + 1).clamp(min=r_own, max=R)
    return rad(ex), rad(ey)

if HAVE_TRITON:
    @triton.jit
    def _radius(e, rmin, R):
        return tl.minimum(tl.maximum(tl.ceil(e).to(tl.int32) + 1, rmin), R)

    @triton.jit
    def _render_radii(G, t, uc, r_own, R):
        hx = tl.load(G + t * 5 + 2); hy = tl.load(G + t * 5 + 3); th = tl.load(G + t * 5 + 4)
        c = tl.cos(th); s = tl.sin(th)
        rx = _radius(uc * tl.sqrt((hx * c) * (hx * c) + (hy * s) * (hy * s)), r_own, R)
        ry = _radius(uc * tl.sqrt((hx * s) * (hx * s) + (hy * c) * (hy * c)), r_own, R)
        return rx, ry

    @triton.jit
    def _encode_radii(G, t, R):
        hx = tl.load(G + t * 5 + 2); hy = tl.load(G + t * 5 + 3); th = tl.load(G + t * 5 + 4)
        c = tl.abs(tl.cos(th)); s = tl.abs(tl.sin(th))
        return _radius(c * hx + s * hy, 0, R), _radius(s * hx + c * hy, 0, R)

    @triton.jit
    def _frame(t, l, Wx, rx, ry, G):
        # pixel l of token t's (2 rx + 1) x (2 ry + 1) box: frame coordinates u, v [BP], pixel indices ix, iy, and the geometry
        mux = tl.load(G + t * 5 + 0); muy = tl.load(G + t * 5 + 1)
        hx = tl.load(G + t * 5 + 2); hy = tl.load(G + t * 5 + 3); th = tl.load(G + t * 5 + 4)
        c = tl.cos(th); s = tl.sin(th)
        ix = tl.floor(mux).to(tl.int32) + (l % Wx) - rx
        iy = tl.floor(muy).to(tl.int32) + (l // Wx) - ry
        dx = ix.to(tl.float32) + 0.5 - mux
        dy = iy.to(tl.float32) + 0.5 - muy
        u = (c * dx + s * dy) / hx
        v = (-s * dx + c * dy) / hy
        return u, v, ix, iy, hx, hy, c, s

    @triton.jit
    def _axis(t, o, r, G, d: tl.constexpr):
        # axis-aligned frames, axis d (0: x, 1: y): offsets o [B] from floor(mu) within radius r -> pixel indices, frame
        # coordinates, and whether the offset is in the box
        mu = tl.load(G + t * 5 + d); h = tl.load(G + t * 5 + 2 + d)
        i = tl.floor(mu).to(tl.int32) + o
        z = (i.to(tl.float32) + 0.5 - mu) / h
        return i, z, o <= r

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
    def _geom_grads_axis(du, dv, u, v, hx, hy):
        # the same for a tile of an axis-aligned frame: du, dv [BY, BX], u [BX], v [BY]
        gmx = tl.sum(tl.sum(du, axis=1)) * (-1.0 / hx)
        gmy = tl.sum(tl.sum(dv, axis=1)) * (-1.0 / hy)
        ghx = tl.sum(tl.sum(du * u[None, :], axis=1)) * (-1.0 / hx)
        ghy = tl.sum(tl.sum(dv * v[:, None], axis=1)) * (-1.0 / hy)
        gth = tl.sum(tl.sum(du * v[:, None], axis=1)) * (hy / hx) - tl.sum(tl.sum(dv * u[None, :], axis=1)) * (hx / hy)
        return gmx, gmy, ghx, ghy, gth

    @triton.jit
    def _window(z, inv2s2):
        # the encoder's window along one frame axis and its derivative: exp(-z^2 / 2 sigma_e^2) inside the frame (|z| < 1),
        # 0 outside; inv2s2 = 1 / (2 sigma_e^2), 0 for the box window (1 inside the frame). The 2D window w(u) w(v) factorizes,
        # so it's folded into the 1D bases: phi(z) = w(z) psi(z), phi'(z) = w(z) psi'(z) + w'(z) psi(z)
        w = tl.where(tl.abs(z) < 1.0, tl.exp(-z * z * inv2s2), 0.0)
        return w, -2.0 * inv2s2 * z * w

    @triton.jit
    def _reflect(i, Hh):
        i = tl.where(i < 0, -i - 1, i)
        i = tl.where(i >= Hh, 2 * Hh - 1 - i, i)
        return tl.minimum(tl.maximum(i, 0), Hh - 1)

    @triton.jit
    def _store_dg(DG, t, gmx, gmy, ghx, ghy, gth):
        tl.store(DG + t * 5 + 0, gmx); tl.store(DG + t * 5 + 1, gmy)
        tl.store(DG + t * 5 + 2, ghx); tl.store(DG + t * 5 + 3, ghy); tl.store(DG + t * 5 + 4, gth)

    @triton.jit
    def _load_tile(PTR, t, ch, n, offs_n):
        q = offs_n[:, None]
        a = offs_n[None, :]
        return tl.load(PTR + ((t * 3 + ch) * n + q) * n + a, mask=(q < n) & (a < n), other=0.0)

    @triton.jit
    def _store_tile(PTR, t, ch, n, offs_n, val):
        q = offs_n[:, None]
        a = offs_n[None, :]
        tl.store(PTR + ((t * 3 + ch) * n + q) * n + a, val, mask=(q < n) & (a < n))

    # --- rotated frames: pixel blocks of the token's box, the 2D basis evaluated per pixel

    @triton.jit
    def _encode_fwd_kernel(X, G, OUT, NCOUNT, K, Hh, R, n, inv2s2, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        rx, ry = _encode_radii(G, t, R)
        Wx = 2 * rx + 1
        P = Wx * (2 * ry + 1)
        acc0 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        cnt = tl.zeros((BLOCK_P,), dtype=tl.float32)
        base = X + b * 3 * Hh * Hh
        for l0 in range(0, P, BLOCK_P):
            l = l0 + tl.arange(0, BLOCK_P)
            u, v, ix, iy, hx, hy, c, s = _frame(t, l, Wx, rx, ry, G)
            m = (l < P) & (tl.abs(u) < 1.0) & (tl.abs(v) < 1.0)
            off = _reflect(iy, Hh) * Hh + _reflect(ix, Hh)
            x0 = tl.load(base + off, mask=m, other=0.0)
            x1 = tl.load(base + Hh * Hh + off, mask=m, other=0.0)
            x2 = tl.load(base + 2 * Hh * Hh + off, mask=m, other=0.0)
            wu, _ = _window(u, inv2s2); wv, _ = _window(v, inv2s2)
            pu = _basis_tile(u, freq, sn, valid_n) * wu[:, None]
            pv = tl.trans(_basis_tile(v, freq, sn, valid_n) * wv[:, None])
            acc0 += _mm(pv, pu * x0[:, None], PREC)
            acc1 += _mm(pv, pu * x1[:, None], PREC)
            acc2 += _mm(pv, pu * x2[:, None], PREC)
            w = tl.where(m, wu * wv, 0.0)
            cnt += w * w
        N = tl.maximum(tl.sum(cnt), 1e-6) # S = sum of squared window weights (the pixel count for the box)
        tl.store(NCOUNT + t, N)
        sc = 1.0 / tl.sqrt(N)
        _store_tile(OUT, t, 0, n, offs_n, acc0 * sc)
        _store_tile(OUT, t, 1, n, offs_n, acc1 * sc)
        _store_tile(OUT, t, 2, n, offs_n, acc2 * sc)

    @triton.jit
    def _encode_bwd_kernel(X, G, GC, NCOUNT, DS, DG, K, Hh, R, n, inv2s2, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        # DS: dL/dS per token (S = sum w^2, the normalizer), from the backward's PyTorch side
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        rx, ry = _encode_radii(G, t, R)
        Wx = 2 * rx + 1
        P = Wx * (2 * ry + 1)
        sc = 1.0 / tl.sqrt(tl.load(NCOUNT + t))
        ds = tl.load(DS + t)
        g0 = _load_tile(GC, t, 0, n, offs_n) * sc
        g1 = _load_tile(GC, t, 1, n, offs_n) * sc
        g2 = _load_tile(GC, t, 2, n, offs_n) * sc
        gmx = 0.0; gmy = 0.0; ghx = 0.0; ghy = 0.0; gth = 0.0
        base = X + b * 3 * Hh * Hh
        for l0 in range(0, P, BLOCK_P):
            l = l0 + tl.arange(0, BLOCK_P)
            u, v, ix, iy, hx, hy, c, s = _frame(t, l, Wx, rx, ry, G)
            m = (l < P) & (tl.abs(u) < 1.0) & (tl.abs(v) < 1.0)
            off = _reflect(iy, Hh) * Hh + _reflect(ix, Hh)
            x0 = tl.load(base + off, mask=m, other=0.0)
            x1 = tl.load(base + Hh * Hh + off, mask=m, other=0.0)
            x2 = tl.load(base + 2 * Hh * Hh + off, mask=m, other=0.0)
            wu, dwu = _window(u, inv2s2); wv, dwv = _window(v, inv2s2)
            bu = _basis_tile(u, freq, sn, valid_n); bv = _basis_tile(v, freq, sn, valid_n)
            pu = bu * wu[:, None]; pv = bv * wv[:, None]
            dpu = _dbasis_tile(u, freq, sn, valid_n) * wu[:, None] + bu * dwu[:, None]
            dpv = _dbasis_tile(v, freq, sn, valid_n) * wv[:, None] + bv * dwv[:, None]
            # dL/du_p = sum_c x_c sum_{q,a} G_c phi_q(v) phi_a'(u) (windowed bases); likewise v
            du = x0 * tl.sum(_mm(pv, g0, PREC) * dpu, axis=1)
            du += x1 * tl.sum(_mm(pv, g1, PREC) * dpu, axis=1)
            du += x2 * tl.sum(_mm(pv, g2, PREC) * dpu, axis=1)
            dv = x0 * tl.sum(_mm(dpv, g0, PREC) * pu, axis=1)
            dv += x1 * tl.sum(_mm(dpv, g1, PREC) * pu, axis=1)
            dv += x2 * tl.sum(_mm(dpv, g2, PREC) * pu, axis=1)
            # through the normalizer: dS/du_p = 2 w(v)^2 w(u) w'(u) (0 for the box)
            fm = tl.where(m, 1.0, 0.0)
            du += ds * 2.0 * wv * wv * wu * dwu * fm
            dv += ds * 2.0 * wu * wu * wv * dwv * fm
            a0, a1, a2, a3, a4 = _geom_grads(du, dv, u, v, hx, hy, c, s)
            gmx += a0; gmy += a1; ghx += a2; ghy += a3; gth += a4
        _store_dg(DG, t, gmx, gmy, ghx, ghy, gth)

    @triton.jit
    def _render_fwd_kernel(COEF, G, NUM, DEN, K, Hh, R, r_own, n, inv2s2, uc,
                           BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        rx, ry = _render_radii(G, t, uc, r_own, R)
        Wx = 2 * rx + 1
        P = Wx * (2 * ry + 1)
        c0 = _load_tile(COEF, t, 0, n, offs_n); c1 = _load_tile(COEF, t, 1, n, offs_n); c2 = _load_tile(COEF, t, 2, n, offs_n)
        inv_n = 1.0 / n
        nb = NUM + b * 3 * Hh * Hh
        for l0 in range(0, P, BLOCK_P):
            l = l0 + tl.arange(0, BLOCK_P)
            u, v, ix, iy, hx, hy, c, s = _frame(t, l, Wx, rx, ry, G)
            inside = (l < P) & (ix >= 0) & (ix < Hh) & (iy >= 0) & (iy < Hh)
            w = tl.exp(-(u * u + v * v) * inv2s2)
            pu = _basis_tile(u, freq, sn, valid_n); pv = _basis_tile(v, freq, sn, valid_n)
            off = iy * Hh + ix
            tl.atomic_add(nb + off, w * tl.sum(_mm(pv, c0, PREC) * pu, axis=1) * inv_n, mask=inside)
            tl.atomic_add(nb + Hh * Hh + off, w * tl.sum(_mm(pv, c1, PREC) * pu, axis=1) * inv_n, mask=inside)
            tl.atomic_add(nb + 2 * Hh * Hh + off, w * tl.sum(_mm(pv, c2, PREC) * pu, axis=1) * inv_n, mask=inside)
            tl.atomic_add(DEN + b * Hh * Hh + off, w, mask=inside)

    @triton.jit
    def _render_bwd_kernel(COEF, G, DNUM, DDEN, DCOEF, DG, K, Hh, R, r_own, n, inv2s2, uc,
                           BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        rx, ry = _render_radii(G, t, uc, r_own, R)
        Wx = 2 * rx + 1
        P = Wx * (2 * ry + 1)
        c0 = _load_tile(COEF, t, 0, n, offs_n); c1 = _load_tile(COEF, t, 1, n, offs_n); c2 = _load_tile(COEF, t, 2, n, offs_n)
        inv_n = 1.0 / n
        acc0 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        gmx = 0.0; gmy = 0.0; ghx = 0.0; ghy = 0.0; gth = 0.0
        nb = DNUM + b * 3 * Hh * Hh
        for l0 in range(0, P, BLOCK_P):
            l = l0 + tl.arange(0, BLOCK_P)
            u, v, ix, iy, hx, hy, c, s = _frame(t, l, Wx, rx, ry, G)
            inside = (l < P) & (ix >= 0) & (ix < Hh) & (iy >= 0) & (iy < Hh)
            w = tl.where(inside, tl.exp(-(u * u + v * v) * inv2s2), 0.0)
            pu = _basis_tile(u, freq, sn, valid_n); pv = _basis_tile(v, freq, sn, valid_n)
            dpu = _dbasis_tile(u, freq, sn, valid_n); dpv = _dbasis_tile(v, freq, sn, valid_n)
            off = iy * Hh + ix
            n0 = tl.load(nb + off, mask=inside, other=0.0)
            n1 = tl.load(nb + Hh * Hh + off, mask=inside, other=0.0)
            n2 = tl.load(nb + 2 * Hh * Hh + off, mask=inside, other=0.0)
            dd = tl.load(DDEN + b * Hh * Hh + off, mask=inside, other=0.0)
            t0 = _mm(pv, c0, PREC); t1 = _mm(pv, c1, PREC); t2 = _mm(pv, c2, PREC)
            # dL/dw_p = sum_c dnum_c val_c + dden
            dw = dd + inv_n * (n0 * tl.sum(t0 * pu, axis=1) + n1 * tl.sum(t1 * pu, axis=1) + n2 * tl.sum(t2 * pu, axis=1))
            # through the values: dL/du_p = sum_c dnum_c w dval_c/du; then through the weights: dw/du = -u w / sigma^2
            du = w * inv_n * (n0 * tl.sum(t0 * dpu, axis=1) + n1 * tl.sum(t1 * dpu, axis=1) + n2 * tl.sum(t2 * dpu, axis=1))
            dv = w * inv_n * (n0 * tl.sum(_mm(dpv, c0, PREC) * pu, axis=1)
                              + n1 * tl.sum(_mm(dpv, c1, PREC) * pu, axis=1)
                              + n2 * tl.sum(_mm(dpv, c2, PREC) * pu, axis=1))
            du += dw * (-2.0 * inv2s2) * u * w
            dv += dw * (-2.0 * inv2s2) * v * w
            # dL/dcoef_c = sum_p dnum_c w psi(v) psi(u) / n
            pvt = tl.trans(pv)
            acc0 += _mm(pvt, pu * (n0 * w * inv_n)[:, None], PREC)
            acc1 += _mm(pvt, pu * (n1 * w * inv_n)[:, None], PREC)
            acc2 += _mm(pvt, pu * (n2 * w * inv_n)[:, None], PREC)
            a0, a1, a2, a3, a4 = _geom_grads(du, dv, u, v, hx, hy, c, s)
            gmx += a0; gmy += a1; ghx += a2; ghy += a3; gth += a4
        _store_tile(DCOEF, t, 0, n, offs_n, acc0)
        _store_tile(DCOEF, t, 1, n, offs_n, acc1)
        _store_tile(DCOEF, t, 2, n, offs_n, acc2)
        _store_dg(DG, t, gmx, gmy, ghx, ghy, gth)

    # --- axis-aligned frames: tiles [BY, BX] of the token's box, separable 1D bases

    @triton.jit
    def _encode_fwd_axis_kernel(X, G, OUT, NCOUNT, K, Hh, R, n, inv2s2, BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr,
                                BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        # coef_c = phi(v)^T X_c phi(u) / sqrt(S_x S_y), the windows folded into the 1D bases (S: sums of squared weights)
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        rx, ry = _encode_radii(G, t, R)
        nx = 0.0
        for x0 in range(0, 2 * rx + 1, BLOCK_X):
            ix, u, ok = _axis(t, x0 - rx + tl.arange(0, BLOCK_X), rx, G, 0)
            wx, _ = _window(u, inv2s2)
            wx = tl.where(ok, wx, 0.0)
            nx += tl.sum(wx * wx)
        acc0 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        ny = 0.0
        base = X + b * 3 * Hh * Hh
        for y0 in range(0, 2 * ry + 1, BLOCK_Y):
            iy, v, ok_y = _axis(t, y0 - ry + tl.arange(0, BLOCK_Y), ry, G, 1)
            my = ok_y & (tl.abs(v) < 1.0)
            wy, _ = _window(v, inv2s2)
            wy = tl.where(ok_y, wy, 0.0)
            ny += tl.sum(wy * wy)
            pv = _basis_tile(v, freq, sn, valid_n) * wy[:, None]
            row = _reflect(iy, Hh) * Hh
            r0 = tl.zeros((BLOCK_Y, BLOCK_N), dtype=tl.float32)
            r1 = tl.zeros((BLOCK_Y, BLOCK_N), dtype=tl.float32)
            r2 = tl.zeros((BLOCK_Y, BLOCK_N), dtype=tl.float32)
            for x0 in range(0, 2 * rx + 1, BLOCK_X):
                ix, u, ok_x = _axis(t, x0 - rx + tl.arange(0, BLOCK_X), rx, G, 0)
                mx = ok_x & (tl.abs(u) < 1.0)
                wx, _ = _window(u, inv2s2)
                pu = _basis_tile(u, freq, sn, valid_n) * tl.where(ok_x, wx, 0.0)[:, None]
                off = row[:, None] + _reflect(ix, Hh)[None, :]
                m2 = my[:, None] & mx[None, :]
                r0 += _mm(tl.load(base + off, mask=m2, other=0.0), pu, PREC)
                r1 += _mm(tl.load(base + Hh * Hh + off, mask=m2, other=0.0), pu, PREC)
                r2 += _mm(tl.load(base + 2 * Hh * Hh + off, mask=m2, other=0.0), pu, PREC)
            pvt = tl.trans(pv)
            acc0 += _mm(pvt, r0, PREC)
            acc1 += _mm(pvt, r1, PREC)
            acc2 += _mm(pvt, r2, PREC)
        N = tl.maximum(nx * ny, 1e-6)
        tl.store(NCOUNT + t, N)
        sc = 1.0 / tl.sqrt(N)
        _store_tile(OUT, t, 0, n, offs_n, acc0 * sc)
        _store_tile(OUT, t, 1, n, offs_n, acc1 * sc)
        _store_tile(OUT, t, 2, n, offs_n, acc2 * sc)

    @triton.jit
    def _encode_bwd_axis_kernel(X, G, GC, NCOUNT, DS, DG, K, Hh, R, n, inv2s2, BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr,
                                BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        # dL/du at each pixel = sum_c x_c (phi(v) G_c phi'(u)^T) + the normalizer's term; likewise v
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        rx, ry = _encode_radii(G, t, R)
        hx = tl.load(G + t * 5 + 2); hy = tl.load(G + t * 5 + 3)
        sc = 1.0 / tl.sqrt(tl.load(NCOUNT + t))
        ds = tl.load(DS + t)
        g0 = _load_tile(GC, t, 0, n, offs_n) * sc
        g1 = _load_tile(GC, t, 1, n, offs_n) * sc
        g2 = _load_tile(GC, t, 2, n, offs_n) * sc
        gmx = 0.0; gmy = 0.0; ghx = 0.0; ghy = 0.0; gth = 0.0
        base = X + b * 3 * Hh * Hh
        for y0 in range(0, 2 * ry + 1, BLOCK_Y):
            iy, v, ok_y = _axis(t, y0 - ry + tl.arange(0, BLOCK_Y), ry, G, 1)
            my = ok_y & (tl.abs(v) < 1.0)
            wy, dwy = _window(v, inv2s2)
            wy = tl.where(ok_y, wy, 0.0); dwy = tl.where(ok_y, dwy, 0.0)
            bv = _basis_tile(v, freq, sn, valid_n)
            pv = bv * wy[:, None]
            dpv = _dbasis_tile(v, freq, sn, valid_n) * wy[:, None] + bv * dwy[:, None]
            a0 = _mm(pv, g0, PREC); a1 = _mm(pv, g1, PREC); a2 = _mm(pv, g2, PREC)
            d0 = _mm(dpv, g0, PREC); d1 = _mm(dpv, g1, PREC); d2 = _mm(dpv, g2, PREC)
            row = _reflect(iy, Hh) * Hh
            for x0 in range(0, 2 * rx + 1, BLOCK_X):
                ix, u, ok_x = _axis(t, x0 - rx + tl.arange(0, BLOCK_X), rx, G, 0)
                mx = ok_x & (tl.abs(u) < 1.0)
                wx, dwx = _window(u, inv2s2)
                wx = tl.where(ok_x, wx, 0.0); dwx = tl.where(ok_x, dwx, 0.0)
                bu = _basis_tile(u, freq, sn, valid_n)
                put = tl.trans(bu * wx[:, None])
                dput = tl.trans(_dbasis_tile(u, freq, sn, valid_n) * wx[:, None] + bu * dwx[:, None])
                off = row[:, None] + _reflect(ix, Hh)[None, :]
                m2 = my[:, None] & mx[None, :]
                x0t = tl.load(base + off, mask=m2, other=0.0)
                x1t = tl.load(base + Hh * Hh + off, mask=m2, other=0.0)
                x2t = tl.load(base + 2 * Hh * Hh + off, mask=m2, other=0.0)
                du = x0t * _mm(a0, dput, PREC) + x1t * _mm(a1, dput, PREC) \
                    + x2t * _mm(a2, dput, PREC)
                dv = x0t * _mm(d0, put, PREC) + x1t * _mm(d1, put, PREC) \
                    + x2t * _mm(d2, put, PREC)
                # through the normalizer S = S_x S_y: dS/du = 2 w(v)^2 w(u) w'(u) (0 for the box)
                du += ds * 2.0 * (wy * wy)[:, None] * (wx * dwx)[None, :]
                dv += ds * 2.0 * (wy * dwy)[:, None] * (wx * wx)[None, :]
                e0, e1, e2, e3, e4 = _geom_grads_axis(du, dv, u, v, hx, hy)
                gmx += e0; gmy += e1; ghx += e2; ghy += e3; gth += e4
        _store_dg(DG, t, gmx, gmy, ghx, ghy, gth)

    @triton.jit
    def _render_fwd_axis_kernel(COEF, G, NUM, DEN, K, Hh, R, r_own, n, inv2s2, uc, BLOCK_Y: tl.constexpr,
                                BLOCK_X: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        # a tile's values (psi(v) coef_c psi(u)^T / n) times the separable weights g(v) g(u), accumulated into the image
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        rx, ry = _render_radii(G, t, uc, r_own, R)
        c0 = _load_tile(COEF, t, 0, n, offs_n); c1 = _load_tile(COEF, t, 1, n, offs_n); c2 = _load_tile(COEF, t, 2, n, offs_n)
        inv_n = 1.0 / n
        nb = NUM + b * 3 * Hh * Hh
        db = DEN + b * Hh * Hh
        for y0 in range(0, 2 * ry + 1, BLOCK_Y):
            iy, v, ok_y = _axis(t, y0 - ry + tl.arange(0, BLOCK_Y), ry, G, 1)
            in_y = ok_y & (iy >= 0) & (iy < Hh)
            gy = tl.where(in_y, tl.exp(-v * v * inv2s2), 0.0)
            pv = _basis_tile(v, freq, sn, valid_n) * (gy * inv_n)[:, None]
            a0 = _mm(pv, c0, PREC); a1 = _mm(pv, c1, PREC); a2 = _mm(pv, c2, PREC)
            for x0 in range(0, 2 * rx + 1, BLOCK_X):
                ix, u, ok_x = _axis(t, x0 - rx + tl.arange(0, BLOCK_X), rx, G, 0)
                in_x = ok_x & (ix >= 0) & (ix < Hh)
                gx = tl.where(in_x, tl.exp(-u * u * inv2s2), 0.0)
                put = tl.trans(_basis_tile(u, freq, sn, valid_n) * gx[:, None])
                off = iy[:, None] * Hh + ix[None, :]
                m2 = in_y[:, None] & in_x[None, :]
                tl.atomic_add(nb + off, _mm(a0, put, PREC), mask=m2)
                tl.atomic_add(nb + Hh * Hh + off, _mm(a1, put, PREC), mask=m2)
                tl.atomic_add(nb + 2 * Hh * Hh + off, _mm(a2, put, PREC), mask=m2)
                tl.atomic_add(db + off, gy[:, None] * gx[None, :], mask=m2)

    @triton.jit
    def _render_bwd_axis_kernel(COEF, G, DNUM, DDEN, DCOEF, DG, K, Hh, R, r_own, n, inv2s2, uc, BLOCK_Y: tl.constexpr,
                                BLOCK_X: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        rx, ry = _render_radii(G, t, uc, r_own, R)
        hx = tl.load(G + t * 5 + 2); hy = tl.load(G + t * 5 + 3)
        c0 = _load_tile(COEF, t, 0, n, offs_n); c1 = _load_tile(COEF, t, 1, n, offs_n); c2 = _load_tile(COEF, t, 2, n, offs_n)
        inv_n = 1.0 / n
        acc0 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
        gmx = 0.0; gmy = 0.0; ghx = 0.0; ghy = 0.0; gth = 0.0
        nb = DNUM + b * 3 * Hh * Hh
        db = DDEN + b * Hh * Hh
        for y0 in range(0, 2 * ry + 1, BLOCK_Y):
            iy, v, ok_y = _axis(t, y0 - ry + tl.arange(0, BLOCK_Y), ry, G, 1)
            in_y = ok_y & (iy >= 0) & (iy < Hh)
            gy = tl.where(in_y, tl.exp(-v * v * inv2s2), 0.0)
            dgy = (-2.0 * inv2s2) * v * gy
            pv = _basis_tile(v, freq, sn, valid_n) * inv_n
            dpv = _dbasis_tile(v, freq, sn, valid_n) * inv_n
            a0 = _mm(pv, c0, PREC); a1 = _mm(pv, c1, PREC); a2 = _mm(pv, c2, PREC)
            d0 = _mm(dpv, c0, PREC); d1 = _mm(dpv, c1, PREC); d2 = _mm(dpv, c2, PREC)
            r0 = tl.zeros((BLOCK_Y, BLOCK_N), dtype=tl.float32)
            r1 = tl.zeros((BLOCK_Y, BLOCK_N), dtype=tl.float32)
            r2 = tl.zeros((BLOCK_Y, BLOCK_N), dtype=tl.float32)
            for x0 in range(0, 2 * rx + 1, BLOCK_X):
                ix, u, ok_x = _axis(t, x0 - rx + tl.arange(0, BLOCK_X), rx, G, 0)
                in_x = ok_x & (ix >= 0) & (ix < Hh)
                gx = tl.where(in_x, tl.exp(-u * u * inv2s2), 0.0)
                dgx = (-2.0 * inv2s2) * u * gx
                pu = _basis_tile(u, freq, sn, valid_n)
                put = tl.trans(pu)
                dput = tl.trans(_dbasis_tile(u, freq, sn, valid_n))
                off = iy[:, None] * Hh + ix[None, :]
                m2 = in_y[:, None] & in_x[None, :]
                n0 = tl.load(nb + off, mask=m2, other=0.0)
                n1 = tl.load(nb + Hh * Hh + off, mask=m2, other=0.0)
                n2 = tl.load(nb + 2 * Hh * Hh + off, mask=m2, other=0.0)
                dd = tl.load(db + off, mask=m2, other=0.0)
                w = gy[:, None] * gx[None, :]
                # dL/dw = sum_c dnum_c val_c + dden; through the values, dL/du = w sum_c dnum_c dval_c/du (likewise v);
                # through the weights, dw/du = g(v) g'(u)
                dw = dd + n0 * _mm(a0, put, PREC) + n1 * _mm(a1, put, PREC) \
                    + n2 * _mm(a2, put, PREC)
                du = w * (n0 * _mm(a0, dput, PREC) + n1 * _mm(a1, dput, PREC)
                          + n2 * _mm(a2, dput, PREC)) + dw * (gy[:, None] * dgx[None, :])
                dv = w * (n0 * _mm(d0, put, PREC) + n1 * _mm(d1, put, PREC)
                          + n2 * _mm(d2, put, PREC)) + dw * (dgy[:, None] * gx[None, :])
                e0, e1, e2, e3, e4 = _geom_grads_axis(du, dv, u, v, hx, hy)
                gmx += e0; gmy += e1; ghx += e2; ghy += e3; gth += e4
                # dL/dcoef_c = psi(v)^T (w dnum_c) psi(u) / n
                r0 += _mm(w * n0, pu, PREC)
                r1 += _mm(w * n1, pu, PREC)
                r2 += _mm(w * n2, pu, PREC)
            pvt = tl.trans(pv)
            acc0 += _mm(pvt, r0, PREC)
            acc1 += _mm(pvt, r1, PREC)
            acc2 += _mm(pvt, r2, PREC)
        _store_tile(DCOEF, t, 0, n, offs_n, acc0)
        _store_tile(DCOEF, t, 1, n, offs_n, acc1)
        _store_tile(DCOEF, t, 2, n, offs_n, acc2)
        _store_dg(DG, t, gmx, gmy, ghx, ghy, gth)

    @triton.jit
    def _encode_dx_kernel(GC, G, NCOUNT, DX, K, Hh, R, n, inv2s2, BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        # the encoder's input gradient (only for diagnostics: training never differentiates w.r.t. the noisy input):
        # dx[c, p] += sum_{q,a} G_c phi_q(v) phi_a(u) / sqrt(S) at each frame pixel, at its (reflected) image position
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        rx, ry = _encode_radii(G, t, R)
        Wx = 2 * rx + 1
        P = Wx * (2 * ry + 1)
        sc = 1.0 / tl.sqrt(tl.load(NCOUNT + t))
        g0 = _load_tile(GC, t, 0, n, offs_n) * sc
        g1 = _load_tile(GC, t, 1, n, offs_n) * sc
        g2 = _load_tile(GC, t, 2, n, offs_n) * sc
        base = DX + b * 3 * Hh * Hh
        for l0 in range(0, P, BLOCK_P):
            l = l0 + tl.arange(0, BLOCK_P)
            u, v, ix, iy, hx, hy, c, s = _frame(t, l, Wx, rx, ry, G)
            m = (l < P) & (tl.abs(u) < 1.0) & (tl.abs(v) < 1.0)
            off = _reflect(iy, Hh) * Hh + _reflect(ix, Hh)
            wu, _ = _window(u, inv2s2); wv, _ = _window(v, inv2s2)
            pu = _basis_tile(u, freq, sn, valid_n) * wu[:, None]; pv = _basis_tile(v, freq, sn, valid_n) * wv[:, None]
            tl.atomic_add(base + off, tl.sum(_mm(pv, g0, PREC) * pu, axis=1), mask=m)
            tl.atomic_add(base + Hh * Hh + off, tl.sum(_mm(pv, g1, PREC) * pu, axis=1), mask=m)
            tl.atomic_add(base + 2 * Hh * Hh + off, tl.sum(_mm(pv, g2, PREC) * pu, axis=1), mask=m)

    @triton.jit
    def _encode_dx_axis_kernel(GC, G, NCOUNT, DX, K, Hh, R, n, inv2s2, BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr,
                               BLOCK_N: tl.constexpr, PREC: tl.constexpr):
        # the same for axis-aligned frames: a tile's gradient is phi(v) G_c phi(u)^T / sqrt(S), windows in the 1D bases
        t = tl.program_id(0)
        b = t // K
        offs_n = tl.arange(0, BLOCK_N)
        valid_n = offs_n < n
        freq = offs_n.to(tl.float32)
        sn = tl.where(offs_n == 0, 1.0, 1.4142135623730951)
        rx, ry = _encode_radii(G, t, R)
        sc = 1.0 / tl.sqrt(tl.load(NCOUNT + t))
        g0 = _load_tile(GC, t, 0, n, offs_n) * sc
        g1 = _load_tile(GC, t, 1, n, offs_n) * sc
        g2 = _load_tile(GC, t, 2, n, offs_n) * sc
        base = DX + b * 3 * Hh * Hh
        for y0 in range(0, 2 * ry + 1, BLOCK_Y):
            iy, v, ok_y = _axis(t, y0 - ry + tl.arange(0, BLOCK_Y), ry, G, 1)
            my = ok_y & (tl.abs(v) < 1.0)
            wy, _ = _window(v, inv2s2)
            pv = _basis_tile(v, freq, sn, valid_n) * tl.where(ok_y, wy, 0.0)[:, None]
            a0 = _mm(pv, g0, PREC); a1 = _mm(pv, g1, PREC); a2 = _mm(pv, g2, PREC)
            row = _reflect(iy, Hh) * Hh
            for x0 in range(0, 2 * rx + 1, BLOCK_X):
                ix, u, ok_x = _axis(t, x0 - rx + tl.arange(0, BLOCK_X), rx, G, 0)
                mx = ok_x & (tl.abs(u) < 1.0)
                wx, _ = _window(u, inv2s2)
                put = tl.trans(_basis_tile(u, freq, sn, valid_n) * tl.where(ok_x, wx, 0.0)[:, None])
                off = row[:, None] + _reflect(ix, Hh)[None, :]
                m2 = my[:, None] & mx[None, :]
                tl.atomic_add(base + off, _mm(a0, put, PREC), mask=m2)
                tl.atomic_add(base + Hh * Hh + off, _mm(a1, put, PREC), mask=m2)
                tl.atomic_add(base + 2 * Hh * Hh + off, _mm(a2, put, PREC), mask=m2)

    # launch configurations per kernel (L40S sweeps at scale range 3; small blocks and few warps keep the basis tiles in
    # registers): rotated frames, pixels per block; axis-aligned frames, tile rows and columns
    CONFIGS = {
        ('rot', 'encode'): dict(BLOCK_P=16, num_warps=2), ('rot', 'encode_bwd'): dict(BLOCK_P=16, num_warps=2),
        ('rot', 'render'): dict(BLOCK_P=32, num_warps=2), ('rot', 'render_bwd'): dict(BLOCK_P=16, num_warps=2),
        ('axis', 'encode'): dict(BLOCK_Y=16, BLOCK_X=16, num_warps=2), ('axis', 'encode_bwd'): dict(BLOCK_Y=16, BLOCK_X=16, num_warps=2),
        ('axis', 'render'): dict(BLOCK_Y=16, BLOCK_X=32, num_warps=4), ('axis', 'render_bwd'): dict(BLOCK_Y=16, BLOCK_X=16, num_warps=2),
        ('rot', 'encode_dx'): dict(BLOCK_P=32, num_warps=2), ('axis', 'encode_dx'): dict(BLOCK_Y=16, BLOCK_X=32, num_warps=4),
    }

    def _bn(n):
        return max(16, triton.next_power_of_2(n))

    def _blocks(axis, op):
        return CONFIGS['axis' if axis else 'rot', op]

    @torch.library.custom_op('gaussjit::encode', mutates_args=())
    def encode_op(x: torch.Tensor, geom: torch.Tensor, K: int, n: int, W: int, axis: bool, inv2s2: float) -> tuple[torch.Tensor, torch.Tensor]:
        # axis: the frames are axis-aligned (theta = 0), separable kernels. inv2s2: the encoder's window, 1 / (2 sigma_e^2)
        # (0: the box, uniform over the frame). Returns the coefficients and S (sum of squared weights; the pixel count for the box)
        x, geom = x.float().contiguous(), geom.float().contiguous()
        T, Hh = geom.shape[0], x.shape[-1]
        out = torch.empty(T, 3, n, n, device=x.device, dtype=torch.float32)
        N = torch.empty(T, device=x.device, dtype=torch.float32)
        kernel = _encode_fwd_axis_kernel if axis else _encode_fwd_kernel
        kernel[(T,)](x, geom, out, N, K, Hh, (W - 1) // 2, n, inv2s2, BLOCK_N=_bn(n), PREC=PRECISION, **_blocks(axis, 'encode'))
        return out, N

    @encode_op.register_fake
    def _(x, geom, K, n, W, axis, inv2s2):
        return x.new_empty((geom.shape[0], 3, n, n), dtype=torch.float32), x.new_empty((geom.shape[0],), dtype=torch.float32)

    @torch.library.custom_op('gaussjit::encode_bwd', mutates_args=())
    def encode_bwd_op(x: torch.Tensor, geom: torch.Tensor, gcoef: torch.Tensor, N: torch.Tensor, dS: torch.Tensor, K: int, n: int,
                      W: int, axis: bool, inv2s2: float) -> torch.Tensor:
        x, geom = x.float().contiguous(), geom.float().contiguous()
        T, Hh = geom.shape[0], x.shape[-1]
        dg = torch.empty(T, 5, device=x.device, dtype=torch.float32)
        kernel = _encode_bwd_axis_kernel if axis else _encode_bwd_kernel
        kernel[(T,)](x, geom, gcoef.float().contiguous(), N, dS.float().contiguous(), dg, K, Hh, (W - 1) // 2, n, inv2s2,
                     BLOCK_N=_bn(n), PREC=PRECISION, **_blocks(axis, 'encode_bwd'))
        return dg

    @encode_bwd_op.register_fake
    def _(x, geom, gcoef, N, dS, K, n, W, axis, inv2s2):
        return geom.new_empty((geom.shape[0], 5), dtype=torch.float32)

    @torch.library.custom_op('gaussjit::encode_dx', mutates_args=())
    def encode_dx_op(geom: torch.Tensor, gcoef: torch.Tensor, N: torch.Tensor, B: int, Hh: int, W: int, axis: bool, inv2s2: float) -> torch.Tensor:
        geom = geom.float().contiguous()
        T, n = geom.shape[0], gcoef.shape[-1]
        dx = torch.zeros(B, 3, Hh, Hh, device=geom.device, dtype=torch.float32)
        kernel = _encode_dx_axis_kernel if axis else _encode_dx_kernel
        kernel[(T,)](gcoef.float().contiguous(), geom, N, dx, T // B, Hh, (W - 1) // 2, n, inv2s2, BLOCK_N=_bn(n), PREC=PRECISION,
                     **_blocks(axis, 'encode_dx'))
        return dx

    @encode_dx_op.register_fake
    def _(geom, gcoef, N, B, Hh, W, axis, inv2s2):
        return geom.new_empty((B, 3, Hh, Hh), dtype=torch.float32)

    def _encode_setup(ctx, inputs, output):
        x, geom, K, n, W, axis, inv2s2 = inputs
        ctx.save_for_backward(x, geom, output[1], output[0] if inv2s2 > 0 else output[1][:0])
        ctx.K, ctx.n, ctx.W, ctx.axis, ctx.inv2s2 = K, n, W, axis, inv2s2

    def _encode_backward(ctx, gcoef, gN):
        x, geom, N, coef = ctx.saved_tensors
        if gcoef is None:
            return None, None, None, None, None, None, None
        # dL/dS for the normalizer (coef = A / sqrt(S)): -sum_c <G_c, coef_c> / (2 S); S is piecewise constant for the box
        dS = -0.5 * (gcoef.float() * coef).sum((1, 2, 3)) / N if ctx.inv2s2 > 0 else torch.zeros_like(N)
        dg = encode_bwd_op(x, geom, gcoef, N, dS, ctx.K, ctx.n, ctx.W, ctx.axis, ctx.inv2s2) if ctx.needs_input_grad[1] else None
        # the input's gradient only when asked for (diagnostics such as the input sensitivity; training's input is constant)
        dx = encode_dx_op(geom, gcoef, N, x.shape[0], x.shape[-1], ctx.W, ctx.axis, ctx.inv2s2).to(x.dtype) if ctx.needs_input_grad[0] else None
        return dx, dg, None, None, None, None, None

    encode_op.register_autograd(_encode_backward, setup_context=_encode_setup)

    @torch.library.custom_op('gaussjit::render', mutates_args=())
    def render_op(coef: torch.Tensor, geom: torch.Tensor, B: int, Hh: int, W: int, sigma: float, r_own: int,
                  axis: bool) -> tuple[torch.Tensor, torch.Tensor]:
        # r_own: the smallest box radius (render_radius); W: the largest box, 2 R + 1
        coef, geom = coef.float().contiguous(), geom.float().contiguous()
        T, n = geom.shape[0], coef.shape[-1]
        num = torch.zeros(B, 3, Hh, Hh, device=coef.device, dtype=torch.float32)
        den = torch.zeros(B, 1, Hh, Hh, device=coef.device, dtype=torch.float32)
        kernel = _render_fwd_axis_kernel if axis else _render_fwd_kernel
        kernel[(T,)](coef, geom, num, den, T // B, Hh, (W - 1) // 2, r_own, n, 1.0 / (2 * sigma * sigma), render_cut(sigma),
                     BLOCK_N=_bn(n), PREC=PRECISION, **_blocks(axis, 'render'))
        return num, den

    @render_op.register_fake
    def _(coef, geom, B, Hh, W, sigma, r_own, axis):
        return coef.new_empty((B, 3, Hh, Hh), dtype=torch.float32), coef.new_empty((B, 1, Hh, Hh), dtype=torch.float32)

    @torch.library.custom_op('gaussjit::render_bwd', mutates_args=())
    def render_bwd_op(coef: torch.Tensor, geom: torch.Tensor, dnum: torch.Tensor, dden: torch.Tensor, W: int, sigma: float,
                      r_own: int, axis: bool) -> tuple[torch.Tensor, torch.Tensor]:
        coef, geom = coef.float().contiguous(), geom.float().contiguous()
        T, n = geom.shape[0], coef.shape[-1]
        B, Hh = dnum.shape[0], dnum.shape[-1]
        dcoef = torch.empty_like(coef)
        dg = torch.empty(T, 5, device=coef.device, dtype=torch.float32)
        kernel = _render_bwd_axis_kernel if axis else _render_bwd_kernel
        kernel[(T,)](coef, geom, dnum.float().contiguous(), dden.float().contiguous(), dcoef, dg, T // B, Hh, (W - 1) // 2, r_own, n,
                     1.0 / (2 * sigma * sigma), render_cut(sigma), BLOCK_N=_bn(n), PREC=PRECISION, **_blocks(axis, 'render_bwd'))
        return dcoef, dg

    @render_bwd_op.register_fake
    def _(coef, geom, dnum, dden, W, sigma, r_own, axis):
        # contiguous, as the real op returns (empty_like would copy a permuted input's strides)
        return coef.new_empty(coef.shape, dtype=torch.float32), geom.new_empty((geom.shape[0], 5), dtype=torch.float32)

    def _render_setup(ctx, inputs, output):
        coef, geom, B, Hh, W, sigma, r_own, axis = inputs
        ctx.save_for_backward(coef, geom)
        ctx.B, ctx.Hh, ctx.W, ctx.sigma, ctx.r_own, ctx.axis = B, Hh, W, sigma, r_own, axis

    def _render_backward(ctx, dnum, dden):
        coef, geom = ctx.saved_tensors
        dnum = coef.new_zeros(ctx.B, 3, ctx.Hh, ctx.Hh) if dnum is None else dnum
        dden = coef.new_zeros(ctx.B, 1, ctx.Hh, ctx.Hh) if dden is None else dden
        dcoef, dg = render_bwd_op(coef, geom, dnum, dden, ctx.W, ctx.sigma, ctx.r_own, ctx.axis)
        return dcoef, dg, None, None, None, None, None, None

    render_op.register_autograd(_render_backward, setup_context=_render_setup)


def ref_encode(x, geom, K, n, W, inv2s2=0.0):
    """
    PyTorch reference for encode_op: x [B, 3, H, H], geom [T = B K, 5] -> (coef [T, 3, n, n], S [T]). inv2s2: the window
    (0: the box), w = exp(-(u^2 + v^2) inv2s2) inside the frame; coef = sum_p w x psi psi / sqrt(S), S = sum_p w^2.
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
    win = lambda z: torch.where(z.abs() < 1, torch.exp(-z * z * inv2s2), torch.zeros_like(z))
    w = win(u) * win(v) # [T, P]: 1 inside the frame for the box
    refl = lambda i: torch.where(i >= Hh, 2 * Hh - 1 - torch.where(i < 0, -i - 1, i), torch.where(i < 0, -i - 1, i)).clamp(0, Hh - 1)
    off = refl(iy) * Hh + refl(ix)
    b = torch.arange(T, device=x.device) // K
    vals = x.reshape(B, C, -1)[b[:, None], :, off].permute(0, 2, 1) * w[:, None] # [T, C, P]
    S = w.square().sum(-1).clamp_min(1e-6)
    return ref_project(vals, u, v, S.rsqrt(), n), S

def ref_render(coef, geom, B, Hh, W, sigma, r_own=0):
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
    rx, ry = render_radius(geom, sigma, r_own, R)
    in_box = ((l % W) - R).abs()[None] <= rx[:, None]
    in_box &= ((l // W) - R).abs()[None] <= ry[:, None]
    inside = ((ix >= 0) & (ix < Hh) & (iy >= 0) & (iy < Hh) & in_box).float()
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
    # fully fused encode / render vs their references (geometry gradients included); rotated and axis-aligned frames, with
    # half-sizes from 2 to 9 pixels so the per-token boxes vary
    B_, K_, Hh, Wd, r_own = 2, 4, 32, 21, 6
    x_img = torch.randn(B_, 3, Hh, Hh, device=dev)
    anchors = torch.tensor([[8., 8.], [24., 8.], [8., 24.], [24., 24.]], device=dev).repeat(B_, 1)
    for axis in (False, True):
        th = torch.zeros(B_ * K_, 1, device=dev) if axis else torch.rand(B_ * K_, 1, device=dev) * 1.4 - 0.7
        geom = torch.cat([anchors + torch.rand(B_ * K_, 2, device=dev) * 4 - 2, 2 + torch.rand(B_ * K_, 2, device=dev) * 7,
                          th], 1).requires_grad_(True)
        tag = 'axis' if axis else 'rot '
        for win, i2 in (('box', 0.0), ('gauss', 1 / (2 * 0.5 ** 2))):
            ck, Nk = encode_op(x_img, geom, K_, n, Wd, axis, i2)
            cr, Nr = ref_encode(x_img, geom, K_, n, Wd, i2)
            print(f'encode {tag} {win:<5}: S match', torch.allclose(Nk, Nr, rtol=1e-5), end=' | ')
            check(f'encode {tag} {win}', ck, cr, (geom,))
            xg = x_img.clone().requires_grad_(True)
            check(f'encode {tag} {win} (input grad)', encode_op(xg, geom, K_, n, Wd, axis, i2)[0], ref_encode(xg, geom, K_, n, Wd, i2)[0], (xg,))
        cf = torch.randn(B_ * K_, 3, n, n, device=dev, requires_grad=True)
        nk, dk = render_op(cf, geom, B_, Hh, Wd, 0.3, r_own, axis)
        nr, dr = ref_render(cf, geom, B_, Hh, Wd, 0.3, r_own)
        img_k, img_r = nk / dk.clamp_min(1e-30), nr / dr.clamp_min(1e-30) # the model's use: num / den
        check(f'render {tag}', img_k, img_r, (cf, geom))
    if dev.startswith('cuda') and not INTERPRET: # the fused ops at GaussJiT-B/16's sizes, scale range 3 (windows 75 / 83)
        Bt, Kt, H128 = 256, 64, 128
        g = torch.arange(8, device=dev).float() * 16 + 8
        anc = torch.stack(torch.meshgrid(g, g, indexing='xy'), -1).view(1, Kt, 2).expand(Bt, -1, -1).reshape(-1, 2)
        xb = torch.randn(Bt, 3, H128, H128, device=dev)
        for scale in (1.0, 3.0):
            for axis in (True, False):
                hh = torch.full((Bt * Kt, 2), 12.0 * scale, device=dev) * (0.9 + 0.2 * torch.rand(Bt * Kt, 2, device=dev))
                th = torch.zeros(Bt * Kt, 1, device=dev) if axis else (torch.rand(Bt * Kt, 1, device=dev) - 0.5) * 1.5
                gm = torch.cat([anc + torch.randn_like(anc), hh.clamp(max=36), th], 1).requires_grad_(True)
                cb = torch.randn(Bt * Kt, 3, n, n, device=dev, requires_grad=True)
                Wr = 75 if axis else 2 * math.ceil(36 * 2 ** 0.5) + 3
                ops = (('encode', lambda: encode_op(xb, gm, Kt, n, Wr, axis, 0.0)[0]), ('render', lambda: render_op(cb, gm, Bt, H128, 83, 0.3, 19, axis)[0]))
                for name, fn in ops:
                    for label, run in (('fwd', fn), ('fwd+bwd', lambda: torch.autograd.grad(fn().sum(), (gm,)))):
                        run(); torch.cuda.synchronize(); t0 = time.time()
                        for _ in range(5):
                            run()
                        torch.cuda.synchronize()
                        print(f'{name} {"axis" if axis else "rot "} scale {scale:.0f} {label:<8}: {(time.time() - t0) / 5 * 1000:7.1f} ms (batch {Bt})')
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
