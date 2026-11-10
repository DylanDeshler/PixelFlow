"""
GPU tests for the fused AttnRes depth-mix kernel (attnres_kernel.py). Exits non-zero if any check fails, so it can gate
--attn_res_kernel=True.
  1. unit: multi_mix (Triton) vs the PyTorch reference, forward and backward (alpha and every source's gradient), over
     dtypes, chunk widths K, source counts S and ragged sizes (N, D not multiples of the tile)
  2. model: a small full-AttnRes JiT (in-context tokens, own-key tail) in blocks mode, kernel vs compiled helpers vs eager:
     loss and gradients
  3. timing at the real size (batch 1024 x 64 tokens x 768, bf16): one chunk's mix, forward + backward, kernel vs the
     compiled helper (--compile_mode=blocks without the kernel) vs eager PyTorch

$ python test_attnres_kernel.py
$ python test_attnres_kernel.py --timing_only=True
"""

import os
import sys
import time

import torch

from attnres_kernel import multi_mix, multi_mix_reference, HAVE_TRITON

# -----------------------------------------------------------------------------
timing_only = False
timing_S = 33 # sources at the chunk's start (depth 18: chunks start with 9 / 17 / 25 / 33 sources)
timing_K = 8 # the chunk width (--attn_res_chunk)
timing_iters = 20
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

assert torch.cuda.is_available() and HAVE_TRITON, 'needs a CUDA GPU and Triton'
dev = 'cuda'
failures = []

def check(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        failures.append(name)

def rel_err(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12)).item()

if not timing_only:
    print('1. unit: kernel vs PyTorch reference')
    torch.manual_seed(0)
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        for S, K, N, D in ((1, 1, 37, 768), (2, 1, 1000, 768), (9, 8, 1000, 768), (37, 8, 4099, 768), (17, 3, 515, 100), (5, 16, 300, 768)):
            z = torch.randn(S, N, K, device=dev)
            src = [torch.randn(N, D, device=dev, dtype=dtype) for _ in range(S)]
            gout = torch.randn(K, N, D, device=dev, dtype=dtype)
            # kernel
            a1 = z.softmax(0).requires_grad_(True)
            v1 = [v.clone().requires_grad_(True) for v in src]
            o1 = multi_mix(a1, v1)
            o1.backward(gout)
            # reference in fp32 on the same values
            a2 = z.softmax(0).requires_grad_(True)
            v2 = [v.float().clone().requires_grad_(True) for v in src]
            o2 = multi_mix_reference(a2, v2)
            o2.backward(gout.float())
            tag = f'{str(dtype)[6:]:<8} S {S:>2} K {K:>2} N {N:>4} D {D:>3}'
            ef = rel_err(o1, o2)
            ea = rel_err(a1.grad, a2.grad)
            ev = max(rel_err(x.grad, y.grad) for x, y in zip(v1, v2))
            check(tag, ef < tol and ea < tol * 5 and ev < tol, f'forward {ef:.1e}, d alpha {ea:.1e}, d sources {ev:.1e} (tol {tol:g})')

    print('2. model: small full-AttnRes JiT, blocks mode, kernel vs compiled helpers vs eager (bf16 autocast)')
    from jit_common import resolve_arch, build_denoiser
    from masking import flow_loss
    import torch._dynamo as dyn
    name, arch = resolve_arch('JiT-B/16', in_context_len=8, in_context_start=2, depth=6, hidden_size=256, num_heads=4,
                              attn_res='full', adaln_single=True, mlp_ratio=2.0, tail_blocks=1, tail_dim=128, tail_heads=8,
                              tail_head_dim=16, tail_mlp_ratio=0, tail_keys='own')
    results = {}
    for mode in ('eager', 'blocks', 'blocks+kernel'):
        dyn.reset()
        torch.manual_seed(0)
        d = build_denoiser(name, 128, 1000, noise_scale=0.5, arch=arch).to(dev).train()
        for p in d.net.parameters():
            p.data.normal_(0, 0.03)
        d.net.attn_res_chunk = 4
        if mode != 'eager':
            d.net.use_block_compile()
        d.net.attn_res_kernel = mode.endswith('kernel')
        torch.manual_seed(1)
        x, y = torch.randn(32, 3, 128, 128, device=dev), torch.randint(0, 1000, (32,), device=dev)
        torch.manual_seed(2)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = flow_loss(d, x, y)
        loss.backward()
        results[mode] = (loss.item(), torch.cat([p.grad.float().flatten() for p in d.net.parameters() if p.grad is not None]))
    le, ge = results['eager']
    for mode in ('blocks', 'blocks+kernel'):
        l, g = results[mode]
        err = ((g - ge).abs().max() / ge.abs().max()).item()
        check(f'model {mode} vs eager', abs(l - le) / abs(le) < 2e-2 and err < 5e-2, f'loss {l:.5f} vs {le:.5f}, grad max rel diff {err:.1e} (bf16)')

print(f'3. timing: one chunk mix, S {timing_S} sources, K {timing_K}, batch 1024 x 64 tokens x 768, bf16, forward + backward')
from model_jit import _attnres_fn
N, D, S, K = 1024 * 64, 768, timing_S, timing_K
torch.manual_seed(0)
z = torch.randn(S, 1024, 64, K, device=dev)
src = [torch.randn(1024, 64, D, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(S)]
gout = torch.randn(K, 1024, 64, D, device=dev, dtype=torch.bfloat16)

def run_kernel():
    multi_mix(z.softmax(0), src).backward(gout)

chunk = _attnres_fn('chunk')
def run_compiled():
    A, _ = chunk(z, *src)
    A.backward(gout)

def run_eager():
    alpha = z.softmax(0)
    A = torch.stack([sum(alpha[i, ..., j].unsqueeze(-1).to(torch.bfloat16) * v for i, v in enumerate(src)) for j in range(K)])
    A.backward(gout)

for name_, fn in (('kernel', run_kernel), ('compiled helper', run_compiled), ('eager', run_eager)):
    try:
        t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        first = time.perf_counter() - t0
        for v in src:
            v.grad = None
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(timing_iters):
            fn()
            for v in src:
                v.grad = None
        en.record(); torch.cuda.synchronize()
        ms = st.elapsed_time(en) / timing_iters
        traffic = (S + K) * N * D * 2 * 3 / 1e9 # ~ reads + writes of the minimal access pattern, fwd + bwd, GB
        print(f"  {name_:<16} {ms:7.2f} ms  (first call incl. compile {first:6.1f} s; peak extra memory "
              f"{(torch.cuda.max_memory_allocated() - base) / 1e9:5.2f} GB; ~{traffic / ms * 1e3:5.0f} GB/s effective)", flush=True)
    except torch.OutOfMemoryError:
        print(f'  {name_:<16} out of memory')
        torch.cuda.empty_cache()

if failures:
    print(f'\n{len(failures)} check(s) FAILED: {failures}')
    sys.exit(1)
print('\nall checks passed' if not timing_only else '')
