"""
Where the time goes with Attention Residuals + adaLN-single, and how much chunked mixing (model_jit.AttnResState,
--attn_res_chunk) recovers. Full compiled training steps (flow_loss forward, backward, AdamW step) on random images at the
training batch, one config after another in one session, as bench_tokens.py:
  plain                  JiT-B/16 (the stack's base before stage 2)
  adaln_single           + adaLN-single only
  attn_res full          + full AttnRes only
  full + adaln           both (the stage-2 run)
  full + adaln, chunk k  the same model, mixing in chunks of k sublayers (identical outputs up to rounding)
  block + adaln          Block AttnRes (a different model: block sums as sources), for reference
  + tail N               with --tails: full + adaln (chunk 8) plus a Perceiver tail of N blocks (train_jit.py --tail_blocks)
Muon's cost is the same per matrix in every config, so AdamW stands in for the optimizer.

$ python bench_attnres.py
$ python bench_attnres.py --chunks=2,4,6,8 --iters=50
$ python bench_attnres.py --depth=18 --mlp_ratio=2.0 --chunks=8,16   # AttnRes's cost on a deeper trunk (compiles slowly: one model at a time)
$ python bench_attnres.py --chunks= --tails=1,2,4             # the Perceiver tail's cost on the stage-2 model
$ python bench_attnres.py --only_tails=True --tails=2 --tail_heads=8 --tail_checkpoint=True   # just that model; vs --act_memory_budget=0.7
"""

import os
import time
from contextlib import nullcontext

import warnings
import torch
warnings.filterwarnings('ignore', message=r'\s*Online softmax is disabled') # torch.compile's note that it skipped one softmax fusion: harmless, and printed for every compiled graph

from jit_common import resolve_arch, build_denoiser
from masking import flow_loss

# -----------------------------------------------------------------------------
img_size = 128
batch_size = 1024
chunks = '2,4,8'
depth = 12 # model shape for every row (e.g. --depth=18 --mlp_ratio=2.0: what AttnRes costs on a deeper, narrower trunk)
mlp_ratio = 4.0
tails = '' # Perceiver tail block counts to add as rows (e.g. '1,2'), with tail_query_patch / tail_dim below
tail_query_patch = 4
tail_dim = 256
tail_heads = 0 # 0 -> tail_dim / 64, as train_jit.py
tail_head_dim = 0
tail_mlp_ratio = 4.0 # 0 = attention-only tail
tail_keys = 'shared'
profile = False # profile 3 timed steps of each row with torch.profiler: top GPU ops by self time (compare with train_jit.py --profile_steps)
rows = '' # ';'-separated row names to run (e.g. 'plain;full + adaln, chunk 8;block + adaln'); '' = all
only_tails = False # run only the --tails rows (the stage-2 model + tail), not the AttnRes comparison
tail_checkpoint = False # as train_jit.py: the tail's queries in chunks, recomputed in the backward
tail_chunks = 8
act_memory_budget = 1.0 # as train_jit.py: < 1 lets torch.compile recompute activations in the backward (whole graph)
warmup = 10
iters = 30
compile = True
compile_mode = 'model' # as train_jit.py: 'model' (whole network) or 'blocks' (small graphs: fast compile for deep full-AttnRes models)
attn_res_kernel = False # full AttnRes: mixes through the fused Triton kernel (attnres_kernel.py; needs compile_mode='blocks' or no compile)
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if act_memory_budget < 1.0:
    import torch._functorch.config
    torch._functorch.config.activation_memory_budget = act_memory_budget
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' or dtype == 'float32') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def bench(name, overrides, chunk=1):
    model_name, arch = resolve_arch('JiT-B/16', in_context_len=0, depth=depth, mlp_ratio=mlp_ratio, **overrides)
    denoiser = build_denoiser(model_name, img_size, 1000, noise_scale=0.5, arch=arch).to(device).train()
    denoiser.net.attn_res_chunk = chunk
    denoiser.net.attn_res_kernel = attn_res_kernel
    if getattr(denoiser.net, 'tail', None) is not None:
        denoiser.net.tail.checkpoint, denoiser.net.tail.chunks = tail_checkpoint, tail_chunks
    opt = torch.optim.AdamW(denoiser.net.parameters(), lr=1e-4, fused=device_type == 'cuda')
    if compile:
        denoiser.net.use_block_compile() if compile_mode == 'blocks' and hasattr(denoiser.net, 'use_block_compile') else denoiser.net.compile()
    micro = batch_size
    while True:
        try:
            if device_type == 'cuda':
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(micro, 3, img_size, img_size, device=device)
            y = torch.randint(0, 1000, (micro,), device=device)
            accum = batch_size // micro
            def step():
                for _ in range(accum):
                    with ctx:
                        loss = flow_loss(denoiser, x, y) / accum
                    loss.backward()
                opt.step(); opt.zero_grad(set_to_none=True)
            print(f'  {name}: compiling + {warmup} warm-up steps ...', flush=True)
            tc = time.perf_counter()
            step()
            if device_type == 'cuda':
                torch.cuda.synchronize()
            compile_s = time.perf_counter() - tc
            print(f'  {name}: first step (compilation) {compile_s / 60:.1f} min', flush=True)
            for _ in range(warmup - 1):
                step()
            if device_type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                step()
            if device_type == 'cuda':
                torch.cuda.synchronize()
            break
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            assert micro > 1, 'out of memory at micro-batch 1'
            micro //= 2
            print(f'  {name}: out of memory, micro-batch {micro} x {batch_size // micro}', flush=True)
    ms = (time.perf_counter() - t0) / iters * 1e3
    if profile:
        from torch.profiler import profile as _profile, ProfilerActivity
        with _profile(activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device_type == 'cuda' else [])) as prof:
            for _ in range(3):
                step()
            if device_type == 'cuda':
                torch.cuda.synchronize()
        sort_key = 'self_cuda_time_total' if device_type == 'cuda' else 'self_cpu_time_total'
        print(f"  {name}: profile over 3 steps (top ops by GPU self time):\n" +
              prof.key_averages().table(sort_by=sort_key, row_limit=30, max_name_column_width=70), flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e9 if device_type == 'cuda' else float('nan')
    row = dict(name=name, ms=ms, ips=batch_size / ms * 1e3, peak=peak, micro=micro, compile_min=compile_s / 60)
    del denoiser, opt
    if device_type == 'cuda':
        torch._dynamo.reset()
    return row

configs = [('plain', {}, 1), ('adaln_single', dict(adaln_single=True), 1), ('attn_res full', dict(attn_res='full'), 1),
           ('full + adaln', dict(attn_res='full', adaln_single=True), 1)]
configs += [(f'full + adaln, chunk {k}', dict(attn_res='full', adaln_single=True), int(k)) for k in chunks.split(',') if k]
configs += [('block + adaln', dict(attn_res='block', adaln_single=True), 1)]
configs += [(f'full + adaln, c8 + tail {n}', dict(attn_res='full', adaln_single=True, tail_blocks=int(n), tail_query_patch=tail_query_patch,
                                                   tail_dim=tail_dim, tail_heads=tail_heads or None, tail_head_dim=tail_head_dim or None, tail_mlp_ratio=tail_mlp_ratio, tail_keys=tail_keys), 8) for n in tails.split(',') if n]
if rows:
    want = [r.strip() for r in rows.split(';') if r.strip()]
    unknown = [w for w in want if w not in [c[0] for c in configs]]
    assert not unknown, f"unknown rows {unknown}; available: {[c[0] for c in configs]}"
    configs = [c for c in configs if c[0] in want]
if only_tails:
    configs = [c for c in configs if 'tail' in c[0]]
    assert configs, 'only_tails: pass --tails (e.g. --tails=2)'
print(f"batch {batch_size} at {img_size}px, {dtype}, compile {compile}; {warmup} warm-up + {iters} timed steps; depth {depth}, mlp_ratio {mlp_ratio:g}")
fmt = lambda r, base: (f"{r['name']:<26} {r['ms']:>9.1f} {r['ips']:>8.0f} {r['ms'] / base['ms']:>8.2f}x {r['peak']:>8.1f} {r['micro']:>6} {r['compile_min']:>8.1f}"
                       if 'ms' in r else f"{r['name']:<26} failed: {r['error']}")
rows = []
for c in configs: # a config that fails (e.g. a compiler error) is reported and skipped; each result prints as it completes
    try:
        rows.append(bench(*c))
    except Exception as e:
        rows.append(dict(name=c[0], error=f'{type(e).__name__}: {str(e).splitlines()[0][:150] if str(e) else ""}'))
        if device_type == 'cuda':
            torch.cuda.empty_cache(); torch._dynamo.reset()
    print('  -> ' + fmt(rows[-1], rows[0] if 'ms' in rows[0] else rows[-1]), flush=True)
base = next(r for r in rows if 'ms' in r)
print(f"\n{'config':<26} {'ms/step':>9} {'img/s':>8} {'vs first':>9} {'peak GB':>8} {'micro':>6} {'compile':>8}")
for r in rows:
    print(fmt(r, base))
