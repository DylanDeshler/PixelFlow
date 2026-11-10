"""
Fused depth-mix kernel for Attention Residuals (model_jit.AttnResState), used with train_jit.py --compile_mode=blocks
--attn_res_kernel=True.

Every AttnRes mix is one operation, a multi-mix: S sources v_s [N, D] (N = batch x tokens) and weights alpha [S, N, K] (fp32)
give K outputs
    out[k, n, :] = sum_s alpha[s, n, k] v_s[n, :]
(a chunk of K sublayers mixed over the sources that existed at its start; K = 1 for folding in a chunk's new sources or a
plain mix). The softmax / logsumexp that produce alpha stay in PyTorch (small, [S, N, K]); this file does the bandwidth-heavy
weighted sum, forward and backward:
  forward   a program owns a tile of rows x feature dims and loops over the sources: each source tile is read once and
            accumulated into all K outputs in registers
  backward  a program loads its tile of the K output gradients once and loops over the sources: each source's gradient
            (sum_k alpha g_k) is written once, and alpha's gradient (a dot product over D) is accumulated across feature
            tiles with atomics
The sources stay separate tensors, passed as a table of data pointers: no stacked copy, and no recompilation per source
count (a Triton kernel specializes only on K's padded size and the dtype). Nothing [S, N, D]-sized is saved: the sources
are saved by reference (autograd keeps them anyway) with alpha.

multi_mix(alpha, sources) uses the kernel on CUDA when Triton is available, else multi_mix_reference (same math in PyTorch).
$ python test_attnres_kernel.py      # on the GPU: kernel vs reference (forward, backward, model level) and timings
"""

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

BLOCK_N, BLOCK_D = 16, 64 # rows x feature dims per program (powers of 2)


def multi_mix_reference(alpha, sources):
    # alpha [S, N, K] fp32, sources S x [N, D] -> [K, N, D] in the sources' dtype
    out = sum(alpha[i].float().unsqueeze(-1) * v.float().unsqueeze(1) for i, v in enumerate(sources)) # [N, K, D]
    return out.transpose(0, 1).to(sources[0].dtype).contiguous()


if HAVE_TRITON:
    _TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}

    @triton.jit
    def _mix_fwd(ptrs, alpha, out, S, N, K, D,
                 DTYPE: tl.constexpr, KP: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        rn = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
        rd = (pid_d * BD + tl.arange(0, BD)).to(tl.int64)
        rk = tl.arange(0, KP).to(tl.int64)
        mn, md, mk = rn < N, rd < D, rk < K
        mv = mn[:, None] & md[None, :]
        ma = mk[:, None] & mn[None, :]
        acc = tl.zeros((KP, BN, BD), dtype=tl.float32)
        for s in range(S):
            src = tl.load(ptrs + s).to(tl.pointer_type(DTYPE))
            v = tl.load(src + rn[:, None] * D + rd[None, :], mask=mv, other=0.0).to(tl.float32) # [BN, BD]
            a = tl.load(alpha + s * N * K + rn[None, :] * K + rk[:, None], mask=ma, other=0.0)  # [KP, BN]
            acc += a[:, :, None] * v[None, :, :]
        o = out + rk[:, None, None] * N * D + rn[None, :, None] * D + rd[None, None, :]
        tl.store(o, acc.to(DTYPE), mask=mk[:, None, None] & mv[None, :, :])

    @triton.jit
    def _mix_bwd(ptrs, gptrs, alpha, gout, galpha, S, N, K, D,
                 DTYPE: tl.constexpr, KP: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        rn = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
        rd = (pid_d * BD + tl.arange(0, BD)).to(tl.int64)
        rk = tl.arange(0, KP).to(tl.int64)
        mn, md, mk = rn < N, rd < D, rk < K
        mv = mn[:, None] & md[None, :]
        ma = mk[:, None] & mn[None, :]
        g = tl.load(gout + rk[:, None, None] * N * D + rn[None, :, None] * D + rd[None, None, :],
                    mask=mk[:, None, None] & mv[None, :, :], other=0.0).to(tl.float32) # [KP, BN, BD]
        for s in range(S):
            src = tl.load(ptrs + s).to(tl.pointer_type(DTYPE))
            dst = tl.load(gptrs + s).to(tl.pointer_type(DTYPE))
            v = tl.load(src + rn[:, None] * D + rd[None, :], mask=mv, other=0.0).to(tl.float32) # [BN, BD]
            a = tl.load(alpha + s * N * K + rn[None, :] * K + rk[:, None], mask=ma, other=0.0)  # [KP, BN]
            gv = tl.sum(a[:, :, None] * g, axis=0) # [BN, BD]: d out / d v_s
            tl.store(dst + rn[:, None] * D + rd[None, :], gv.to(DTYPE), mask=mv)
            ga = tl.sum(v[None, :, :] * g, axis=2) # [KP, BN]: this feature tile's part of d out / d alpha
            tl.atomic_add(galpha + s * N * K + rn[None, :] * K + rk[:, None], ga, mask=ma)

    def _ptr_table(tensors, device):
        # the sources' data pointers as a device int64 table, uploaded asynchronously from pinned memory: a plain
        # torch.tensor(..., device='cuda') copies from pageable memory, which blocks the host until the GPU drains, i.e. a
        # full synchronize on every mix (~80 per step), leaving the GPU idle while Python launches the next work. PyTorch's
        # pinned-memory allocator keeps the host buffer until the copy has run.
        host = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64).pin_memory()
        return host.to(device, non_blocking=True)

    def _launch_cfg(K):
        KP = max(1, triton.next_power_of_2(K))
        return KP, (8 if KP >= 16 else 4)

    def _mix_forward(alpha, sources):
        # alpha [S, N, K] fp32, sources S x [N, D] -> [K, N, D]
        S, N, K = alpha.shape
        D = sources[0].shape[-1]
        dt = sources[0].dtype
        out = torch.empty((K, N, D), device=alpha.device, dtype=dt)
        ptrs = _ptr_table(sources, alpha.device)
        KP, warps = _launch_cfg(K)
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(D, BLOCK_D))
        _mix_fwd[grid](ptrs, alpha, out, S, N, K, D, DTYPE=_TL_DTYPE[dt], KP=KP, BN=BLOCK_N, BD=BLOCK_D, num_warps=warps)
        return out

    class _MultiMix(torch.autograd.Function):
        @staticmethod
        def forward(ctx, alpha, *sources):
            out = _mix_forward(alpha, sources)
            ctx.save_for_backward(alpha, *sources)
            return out

        @staticmethod
        def backward(ctx, gout):
            alpha, *sources = ctx.saved_tensors
            return tuple(_mix_backward(alpha, sources, gout))

    def _mix_backward(alpha, sources, gout):
        # alpha [S, N, K] fp32, sources S x [N, D], gout [K, N, D] -> [d alpha, d source_0, ..., d source_S-1]
        S, N, K = alpha.shape
        D = sources[0].shape[-1]
        dt = sources[0].dtype
        gout = gout.contiguous()
        if gout.dtype != dt:
            gout = gout.to(dt)
        grads = [torch.empty_like(v) for v in sources]
        galpha = torch.zeros_like(alpha)
        ptrs = _ptr_table(sources, alpha.device)
        gptrs = _ptr_table(grads, alpha.device)
        KP, warps = _launch_cfg(K)
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(D, BLOCK_D))
        _mix_bwd[grid](ptrs, gptrs, alpha, gout, galpha, S, N, K, D, DTYPE=_TL_DTYPE[dt], KP=KP, BN=BLOCK_N, BD=BLOCK_D,
                       num_warps=warps)
        return [galpha, *grads]


def kernel_usable(sources):
    return (HAVE_TRITON and sources[0].is_cuda and sources[0].dtype in (torch.bfloat16, torch.float16, torch.float32)
            and all(v.dtype == sources[0].dtype and v.shape == sources[0].shape for v in sources))


def multi_mix(alpha, sources):
    """
    alpha [S, *rows, K] (any float dtype; computed in fp32), sources: S tensors [*rows, D] -> [K, *rows, D] in the
    sources' dtype. The fused kernel on CUDA (with Triton), else the PyTorch reference; same math.
    """
    rows = sources[0].shape[:-1]
    D = sources[0].shape[-1]
    S, K = alpha.shape[0], alpha.shape[-1]
    a = alpha.float().reshape(S, -1, K).contiguous()
    flat = [v.reshape(-1, D).contiguous() for v in sources]
    if kernel_usable(flat):
        out = _MultiMix.apply(a, *flat)
    else:
        out = multi_mix_reference(a, flat)
    return out.view(K, *rows, D)
