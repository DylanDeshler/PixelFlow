"""
Fused-backward Attention Residuals (train_jit.py --attn_res_fused=True): an opt-in, self-contained alternative to
model_jit.AttnResState's training path. Nothing here is used unless the flag is set; without it the model runs exactly as
before (model_jit.AttnResState + attnres_kernel.py), so falling back is just dropping the flag.

Same forward math as the chunked kernel path (attnres_kernel.multi_mix), different backward bookkeeping. In the default
path every mix's backward allocates a fresh gradient tensor for each source it read and autograd sums those into the
source's total, one allocation + read-modify-write per (source, mix call); at depth 18 that accumulation was ~20% of GPU
time. Here one gradient buffer G [S, N, D] (in the sources' dtype, as autograd accumulates them) lives per forward pass:
each mix's backward kernel adds its contribution for every source it read straight into G[s], and each source's producer
receives G[s] once, after every mix that read it is done. That order is autograd's own: sources are linked by a chain of
zero-size handle tensors (handle_s = Append(handle_{s-1}, v_s)); every mix takes the latest handle as an input, so
Append_s's backward runs only after all mixes downstream of handle_s, i.e. all readers of v_s.
  PreMix   a chunk's k partial mixes over the n0 sources existing at its start (forward: attnres_kernel's fused kernel;
           backward: one pass that accumulates sum_k alpha[s, k] g_k into G[s] for each source)
  FoldMix  one sublayer's mix: w0 A_j + sum_new w_i v_i (backward: A_j's gradient as a tensor, flowing into PreMix; the new
           sources' contributions into G)
  PlainMix an unchunked mix (attn_res_chunk 1)
Supported: full AttnRes in training (grad enabled), no in-context prepend, no Block AttnRes; anything else (eval,
diagnostics, in-context tokens) uses model_jit.AttnResState unchanged. On CUDA with Triton the kernels run; elsewhere the
same Functions use PyTorch ops (for tests).

$ python test_attnres_fused.py     # on the GPU: fused vs default path (forward, every gradient), and timings
"""

import torch

from attnres_kernel import HAVE_TRITON, multi_mix_reference

if HAVE_TRITON:
    import triton
    import triton.language as tl
    from attnres_kernel import _mix_forward, _launch_cfg, _ptr_table, _TL_DTYPE, BLOCK_N, BLOCK_D

    @triton.jit
    def _mix_bwd_accum(ptrs, gptrs, alpha, gout, galpha, S, N, K, D,
                       DTYPE: tl.constexpr, KP: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        # as attnres_kernel._mix_bwd, but each source's gradient is ADDED into its slice of the shared buffer G
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
            v = tl.load(src + rn[:, None] * D + rd[None, :], mask=mv, other=0.0).to(tl.float32)
            a = tl.load(alpha + s * N * K + rn[None, :] * K + rk[:, None], mask=ma, other=0.0)
            gv = tl.sum(a[:, :, None] * g, axis=0)
            off = dst + rn[:, None] * D + rd[None, :]
            old = tl.load(off, mask=mv, other=0.0).to(tl.float32)
            tl.store(off, (old + gv).to(DTYPE), mask=mv)
            ga = tl.sum(v[None, :, :] * g, axis=2)
            tl.atomic_add(galpha + s * N * K + rn[None, :] * K + rk[:, None], ga, mask=ma)


def _use_kernel(t):
    return HAVE_TRITON and t.is_cuda


def _forward_mix(alpha, sources):
    # alpha [S, N, K] fp32, sources S x [N, D] -> [K, N, D]
    if _use_kernel(sources[0]):
        return _mix_forward(alpha, sources)
    return multi_mix_reference(alpha, sources)


def _backward_accum(alpha, sources, gout, G, idx):
    # adds sum_k alpha[s, :, k] gout[k] into G[idx[s]] for every source s; returns d alpha [S, N, K] (fp32)
    S, N, K = alpha.shape
    if _use_kernel(sources[0]):
        dt = sources[0].dtype
        gout = gout.contiguous()
        if gout.dtype != dt:
            gout = gout.to(dt)
        galpha = torch.zeros_like(alpha)
        D = sources[0].shape[-1]
        KP, warps = _launch_cfg(K)
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(D, BLOCK_D))
        _mix_bwd_accum[grid](_ptr_table(sources, alpha.device), _ptr_table([G[i] for i in idx], alpha.device), alpha, gout, galpha,
                             S, N, K, D, DTYPE=_TL_DTYPE[dt], KP=KP, BN=BLOCK_N, BD=BLOCK_D, num_warps=warps)
        return galpha
    gf = gout.float()
    galpha = torch.stack([(v.float().unsqueeze(0) * gf).sum(-1).transpose(0, 1) for v in sources]) # [S, N, K]
    for s, i in enumerate(idx):
        G[i] += (alpha[s].unsqueeze(-1).transpose(0, 1) * gf).sum(0).to(G.dtype) # sum_k alpha[s, :, k] g_k
    return galpha


class _Append(torch.autograd.Function):
    # handle_s = Append(handle_{s-1}, v_s): links source s into the chain; backward hands v_s its accumulated gradient G[s]
    @staticmethod
    def forward(ctx, handle, v, state, s):
        ctx.state, ctx.s, ctx.shape = state, s, v.shape
        return v.new_empty(0)

    @staticmethod
    def backward(ctx, ghandle):
        state = ctx.state
        G = state.grad_buffer()
        gv = G[ctx.s].view(ctx.shape)
        if ctx.s == 0: # the last node of the backward (every mix depends on the first source): drop the state's references,
            # breaking the cycle state -> handle -> graph nodes -> state, so the buffer, the sources and the graph are freed by
            # reference counting now (the returned view keeps G[0] alive until autograd has used it)
            state.release()
        ctx.state = None
        return torch.empty(0, device=G.device, dtype=G.dtype), gv, None, None


class _PreMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, handle, state, n0):
        # alpha [n0, N, K] fp32 over the first n0 sources -> the chunk's partial mixes [K, N, D]
        ctx.state, ctx.n0 = state, n0
        ctx.save_for_backward(alpha)
        return _forward_mix(alpha, state.flat[:n0])

    @staticmethod
    def backward(ctx, gA):
        (alpha,) = ctx.saved_tensors
        galpha = _backward_accum(alpha, ctx.state.flat[:ctx.n0], gA, ctx.state.grad_buffer(), list(range(ctx.n0)))
        return galpha, torch.empty(0, device=gA.device, dtype=gA.dtype), None, None


class _FoldMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, A_j, handle, state, n0, S):
        # w [1 + n, N] (softmax over [the partial's logsumexp, the n new sources' logits]), A_j [N, D] -> [N, D]
        ctx.state, ctx.n0, ctx.S = state, n0, S
        ctx.save_for_backward(w, A_j)
        parts = [A_j, *state.flat[n0:S]]
        return _forward_mix(w.unsqueeze(-1).contiguous(), parts)[0]

    @staticmethod
    def backward(ctx, g):
        w, A_j = ctx.saved_tensors
        g = g.contiguous()
        gf = g.float()
        gA = (w[0].unsqueeze(-1) * gf).to(A_j.dtype) # d out / d A_j, flows into PreMix
        gw0 = (A_j.float() * gf).sum(-1)
        n0, S = ctx.n0, ctx.S
        if S > n0:
            gwn = _backward_accum(w[1:].unsqueeze(-1).contiguous(), ctx.state.flat[n0:S], g.unsqueeze(0), ctx.state.grad_buffer(),
                                  list(range(n0, S)))[..., 0]
            gw = torch.cat([gw0.unsqueeze(0), gwn])
        else:
            gw = gw0.unsqueeze(0)
        return gw, gA, torch.empty(0, device=g.device, dtype=g.dtype), None, None, None


class _PlainMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, handle, state, S):
        # alpha [S, N] fp32 over the first S sources -> [N, D]
        ctx.state, ctx.S = state, S
        ctx.save_for_backward(alpha)
        return _forward_mix(alpha.unsqueeze(-1).contiguous(), state.flat[:S])[0]

    @staticmethod
    def backward(ctx, g):
        (alpha,) = ctx.saved_tensors
        galpha = _backward_accum(alpha.unsqueeze(-1).contiguous(), ctx.state.flat[:ctx.S], g.unsqueeze(0), ctx.state.grad_buffer(),
                                 list(range(ctx.S)))[..., 0]
        return galpha, torch.empty(0, device=g.device, dtype=g.dtype), None, None


def fused_supported(net, training):
    return (training and torch.is_grad_enabled() and getattr(net, 'attn_res', 'none') == 'full'
            and getattr(net, 'in_context_len', 0) == 0)


def make_state(net, emb, c=None):
    # an AttnResState with the fused backward (see the module docstring); built lazily to avoid a circular import
    from model_jit import AttnResState

    class AttnResStateFused(AttnResState):
        def __init__(self, net, emb, c=None):
            self.flat, self.handle, self._G, self.rows = [], None, None, emb.shape[:-1]
            super().__init__(net, emb, c)

        def release(self):
            # after the backward: nothing of this forward pass stays referenced from the state
            self.flat, self.handle, self._G, self._pre = [], None, None, None
            self.sources, self.logits, self.partial = [], [], None

        def grad_buffer(self):
            # one slice per source, in the sources' dtype (bf16 under autocast, as autograd accumulates them), zeroed at the
            # first backward use; freed with the state after backward
            if self._G is None:
                v = self.flat[0]
                self._G = torch.zeros((len(self.flat), *v.shape), device=v.device, dtype=v.dtype)
            return self._G

        def _append(self, v):
            super()._append(v) # sources list + logits for every query (unchanged)
            D = v.shape[-1]
            flat = v.reshape(-1, D)
            flat = flat if flat.is_contiguous() else flat.contiguous()
            self.flat.append(flat.detach())
            root = self.handle if self.handle is not None else v.new_empty(0)
            self.handle = _Append.apply(root, v, self, len(self.flat) - 1)

        def prepend(self, prefix):
            raise NotImplementedError('attn_res_fused does not support in-context tokens (use the default path)')

        def add(self, out):
            assert self.group == 1, 'attn_res_fused: full AttnRes only'
            super().add(out)

        def _mix_chunked(self, l):
            pre, D = self._pre, self.sources[0].shape[-1]
            N = self.flat[0].shape[0]
            if pre is None or not (pre['start'] <= l < pre['start'] + len(pre['A'])):
                cols = list(range(l, min(l + self.chunk, self.logits[0].shape[-1])))
                z = torch.stack([c[..., cols] for c in self.logits]) # [S, B, T, k]
                n0 = len(self.sources)
                alpha = z.softmax(0).reshape(n0, N, len(cols)).contiguous()
                A = _PreMix.apply(alpha, self.handle, self, n0) # [k, N, D]
                pre = self._pre = dict(start=l, n0=n0, A=list(A.unbind(0)), lse=torch.logsumexp(z, 0))
            j = l - pre['start']
            S, n0 = len(self.sources), pre['n0']
            z_new = [self.logits[i][..., l].reshape(N) for i in range(n0, S)]
            w = torch.stack([pre['lse'][..., j].reshape(N), *z_new]).softmax(0) # [1 + n, N]
            return _FoldMix.apply(w.contiguous(), pre['A'][j], self.handle, self, n0, S).view(*self.rows, D)

        def mix(self, l):
            if self.weights is not None or self.partial is not None or len(self.sources) == 1:
                return super().mix(l)
            if self.chunk > 1:
                return self._mix_chunked(l)
            S, N, D = len(self.sources), self.flat[0].shape[0], self.sources[0].shape[-1]
            alpha = torch.stack([c[..., l].reshape(N) for c in self.logits]).softmax(0).contiguous()
            return _PlainMix.apply(alpha, self.handle, self, S).view(*self.rows, D)

        def mix_query(self, query, norm):
            raise NotImplementedError('attn_res_fused: mix_query unsupported (tail keys use the extra-query columns)')

    return AttnResStateFused(net, emb, c)
