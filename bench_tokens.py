"""
Training cost of two token sizes (train_jit.py --token_sizes) against the single-size models, in one session on one GPU:
JiT-B/16, JiT-B/8 and JiT-B/16 with 16/8 tokens at several fine shares. Each config runs full training steps (flow_loss
forward, backward, AdamW step; compiled as in train_jit.py) on random images at the training batch, and reports
  ms/step, img/s     timed over --iters steps after --warmup steps (compilation)
  peak GB            peak allocated memory (if a batch doesn't fit, micro-batches with accumulation, as train_jit.py would:
                     the micro column). Two token sizes backpropagate each path right after its forward, as train_jit.py
                     does, so the peak is the larger path's; one row repeats the training share with a single backward to
                     show what that saves
  max fine share     from the two-size rows' peaks (linear in the fine path's tokens), the largest fine share whose step
                     fits the GPU without accumulation
  GFLOP/img          analytic matmul + attention FLOPs of forward + backward (3x forward), per training image
  achieved TFLOP/s   GFLOP/img x img/s: how efficiently each config's shapes use the GPU (the two-size step runs one
                     smaller-batch forward per size)
Muon's per-step cost is the same for every config (identical block matrices), so AdamW stands in for the optimizer.

$ python bench_tokens.py                               # 128px, batch 1024
$ python bench_tokens.py --fine_fracs=0.25 --iters=50
$ python bench_tokens.py --act_memory_budget=0.5       # with Inductor's selective recompute (train_jit.py's flag)
"""

import os
import time
from contextlib import nullcontext

import torch

from jit_common import resolve_arch, build_denoiser
from masking import flow_loss

# -----------------------------------------------------------------------------
img_size = 128
batch_size = 1024
fine_fracs = '0.1,0.25,0.35,0.5'
warmup = 10
iters = 30
compile = True
act_memory_budget = 1.0 # as train_jit.py: < 1 recomputes activations in the backward to save memory
single_backward_frac = 0.25 # also time this fine share with one backward over both paths (0 = skip)
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._dynamo.config.automatic_dynamic_shapes = False # as train_jit.py with token_sizes: one static graph per shape
if act_memory_budget < 1.0:
    import torch._functorch.config
    torch._functorch.config.activation_memory_budget = act_memory_budget
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' or dtype == 'float32') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def gflop_fwd(n_tokens, arch):
    # per image, forward: per block 2 x (4 d^2 attention projections + 2 x mlp_ratio d^2 MLP) x N, + 2 x 2 N^2 d attention
    d, L, r = arch['hidden_size'], arch['depth'], arch['mlp_ratio']
    return L * (2 * (4 + 2 * r) * d * d * n_tokens + 4 * n_tokens * n_tokens * d) / 1e9

def bench(name, overrides, tokens_per_img, split=True):
    model_name, arch = resolve_arch('JiT-B/16', in_context_len=0, **overrides)
    denoiser = build_denoiser(model_name, img_size, 1000, noise_scale=0.5, arch=arch).to(device).train()
    opt = torch.optim.AdamW(denoiser.net.parameters(), lr=1e-4, fused=device_type == 'cuda')
    if compile:
        denoiser.net.compile()
    gfl = 3 * sum(f * gflop_fwd(n, arch) for f, n in tokens_per_img)
    micro = batch_size
    while True:
        try:
            if device_type == 'cuda':
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(micro, 3, img_size, img_size, device=device)
            y = torch.randint(0, 1000, (micro,), device=device)
            accum = batch_size // micro
            def step():
                two = 'token_sizes' in overrides
                for _ in range(accum):
                    with ctx: # two sizes: each path's backward right after its forward (as train_jit.py), unless split=False
                        loss = flow_loss(denoiser, x, y, backward=(lambda l: (l / accum).backward()) if two and split else None) / accum
                    if not (two and split):
                        loss.backward()
                opt.step(); opt.zero_grad(set_to_none=True)
            for _ in range(warmup):
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
            print(f'  {name}: out of memory, micro-batch {micro} x {batch_size // micro}')
    ms = (time.perf_counter() - t0) / iters * 1e3
    ips = batch_size / ms * 1e3
    peak = torch.cuda.max_memory_allocated() / 1e9 if device_type == 'cuda' else float('nan')
    tok = sum(f * n for f, n in tokens_per_img)
    row = dict(name=name, tokens=tok, ms=ms, ips=ips, peak=peak, micro=micro, gflop=gfl, tflops=gfl * ips / 1e3)
    print(f"{name:<22} {tok:>7.0f} {ms:>9.1f} {ips:>8.0f} {peak:>8.1f} {micro:>6} {gfl:>10.1f} {row['tflops']:>9.1f}")
    del denoiser, opt
    if device_type == 'cuda':
        torch._dynamo.reset() # fresh compile caches per config
    return row

n16, n8 = (img_size // 16) ** 2, (img_size // 8) ** 2
print(f"batch {batch_size} at {img_size}px, {dtype}, compile {compile}; {warmup} warm-up + {iters} timed steps")
print(f"{'config':<22} {'tok/img':>7} {'ms/step':>9} {'img/s':>8} {'peak GB':>8} {'micro':>6} {'GFLOP/img':>10} {'TFLOP/s':>9}")
rows = [bench('B/16', {}, [(1.0, n16)])]
for f in (float(v) for v in fine_fracs.split(',') if v):
    rows.append(bench(f'B/16+8 fine {f:g}', dict(token_sizes='16,8', fine_frac=f), [(1 - f, n16), (f, n8)]))
if single_backward_frac > 0:
    f = single_backward_frac
    rows.append(bench(f'  one backward {f:g}', dict(token_sizes='16,8', fine_frac=f), [(1 - f, n16), (f, n8)], split=False))
rows.append(bench('B/8', dict(patch_size=8), [(1.0, n8)]))
base = rows[0]
print("\nrelative to B/16 (wall-clock per image; FLOPs per image):")
for r in rows[1:]:
    print(f"  {r['name']:<22} time x{base['ips'] / r['ips']:.2f}, FLOPs x{r['gflop'] / base['gflop']:.2f}")
# largest fine share without accumulation: the split step's peak is the fine path's once f B n8 > (1 - f) B n16
two = [(float(r['name'].split()[-1]), r['peak']) for r in rows if r['name'].startswith('B/16+8') and r['micro'] == batch_size]
two = [(f, p) for f, p in two if f * n8 >= (1 - f) * n16]
if device_type == 'cuda' and len(two) >= 2:
    (f1, p1), (f2, p2) = two[0], two[-1]
    slope = (p2 - p1) / (f2 - f1)
    cap = torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.95
    f_max = f2 + (cap - p2) / slope if slope > 0 else float('inf')
    print(f"\nmemory: {slope:.1f} GB per unit fine share at batch {batch_size}; {cap:.0f} GB usable -> fine share up to ~{min(f_max, 1):.2f} "
          f"fits without gradient accumulation (act_memory_budget {act_memory_budget:g})")
