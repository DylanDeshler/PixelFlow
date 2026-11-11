"""
GPU tests for the fused-backward AttnRes path (attnres_fused.py, --attn_res_fused=True). Exits non-zero if a check fails.
  1. unit: the accumulating backward kernel vs the PyTorch reference (sources' gradients added into a shared buffer that
     already holds values, and alpha's gradient), fp32 and bf16, ragged sizes
  2. model: a small full-AttnRes JiT (own-key tail) in blocks mode with the kernel, fused vs default path (bf16 autocast),
     two micro-steps: loss and every parameter's gradient
Speed: bench_attnres.py --attn_res_fused=True (see the end of the output).

$ python test_attnres_fused.py
"""

import sys

import torch

from attnres_kernel import HAVE_TRITON
import attnres_fused as af

assert torch.cuda.is_available() and HAVE_TRITON, 'needs a CUDA GPU and Triton'
dev = 'cuda'
failures = []

def check(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        failures.append(name)

def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12)).item()

print('1. unit: accumulating backward kernel vs reference')
torch.manual_seed(0)
for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 3e-2)):
    for S, K, N, D in ((1, 1, 37, 768), (9, 8, 1000, 768), (33, 8, 4099, 768), (17, 3, 515, 100), (5, 16, 300, 768)):
        alpha = torch.rand(S, N, K, device=dev)
        src = [torch.randn(N, D, device=dev, dtype=dtype) for _ in range(S)]
        gout = torch.randn(K, N, D, device=dev, dtype=dtype)
        idx = list(range(S))
        G0 = torch.randn(S + 2, N, D, device=dev, dtype=dtype) # pre-existing contents: must be added to, not overwritten
        G_k, G_r = G0.clone(), G0.clone().float()
        ga_k = af._backward_accum(alpha, src, gout, G_k, idx)
        # reference in fp32
        gf = gout.float()
        ga_r = torch.stack([(v.float().unsqueeze(0) * gf).sum(-1).transpose(0, 1) for v in src])
        for s in idx:
            G_r[s] += (alpha[s].unsqueeze(-1).transpose(0, 1) * gf).sum(0)
        tag = f'{str(dtype)[6:]:<8} S {S:>2} K {K:>2} N {N:>4} D {D:>3}'
        eg, ea = rel(G_k[:S], G_r[:S]), rel(ga_k, ga_r)
        untouched = torch.equal(G_k[S:], G0[S:])
        check(tag, eg < tol and ea < tol * 5 and untouched, f'G {eg:.1e}, d alpha {ea:.1e}, other slices untouched {untouched} (tol {tol:g})')

print('2. model: blocks mode + kernel, fused vs default path, bf16 autocast, 2 micro-steps')
from jit_common import resolve_arch, build_denoiser
from masking import flow_loss
import torch._dynamo as dyn
name, arch = resolve_arch('JiT-B/16', in_context_len=0, depth=6, hidden_size=256, num_heads=4, attn_res='full', adaln_single=True,
                          mlp_ratio=2.0, tail_blocks=1, tail_dim=128, tail_heads=8, tail_head_dim=16, tail_mlp_ratio=0, tail_keys='own')
res = {}
for fused in (False, True):
    dyn.reset(); torch.manual_seed(0)
    d = build_denoiser(name, 128, 1000, noise_scale=0.5, arch=arch).to(dev).train()
    for p in d.net.parameters():
        p.data.normal_(0, 0.03)
    d.net.attn_res_chunk = 4
    d.net.use_block_compile()
    d.net.attn_res_kernel = True
    d.net.attn_res_fused = fused
    torch.manual_seed(1)
    x, y = torch.randn(32, 3, 128, 128, device=dev), torch.randint(0, 1000, (32,), device=dev)
    tot = 0.0
    for m in range(2):
        torch.manual_seed(2 + m)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = flow_loss(d, x, y)
        loss.backward(); tot += loss.item()
    res[fused] = (tot, torch.cat([p.grad.float().flatten() for p in d.net.parameters() if p.grad is not None]))
(l0, g0), (l1, g1) = res[False], res[True]
err = ((g1 - g0).abs().max() / g0.abs().max()).item()
check('model fused vs default', abs(l1 - l0) / abs(l0) < 1e-3 and err < 5e-2, f'loss {l1:.5f} vs {l0:.5f}, grad max rel diff {err:.1e} (bf16)')

if failures:
    print(f'\n{len(failures)} check(s) FAILED: {failures}')
    sys.exit(1)
print('\nall checks passed. Speed on the real model:\n'
      "  python bench_attnres.py --depth=18 --mlp_ratio=2.0 --compile_mode=blocks --attn_res_kernel=True --batch_size=512 --rows='full + adaln, chunk 8' --attn_res_fused=True\n"
      '  (vs the same without --attn_res_fused: 1278 img/s, 20.8 GB)')
