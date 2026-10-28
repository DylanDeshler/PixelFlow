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

    PRECISION = 'ieee' if INTERPRET else 'tf32' # TF32 dots on GPU (the rest of the model trains in bf16); exact fp32 when testing

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
            f'grad {nm} rel err {rel(a, b):.2e}' for nm, a, b in zip(('f/coef', 'u', 'v'), gk, gr) if a is not None and b is not None))
    check('project ', project(f, u, v, scale, n), ref_project(f, u, v, scale, n), (f, u, v))
    check('evaluate', evaluate(coef, u, v, scale), ref_evaluate(coef, u, v, scale), (coef, u, v))
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
