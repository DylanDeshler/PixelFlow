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
Muon's cost is the same per matrix in every config, so AdamW stands in for the optimizer.

$ python bench_attnres.py
$ python bench_attnres.py --chunks=2,4,6,8 --iters=50
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
chunks = '2,4,8'
warmup = 10
iters = 30
compile = True
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' or dtype == 'float32') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def bench(name, overrides, chunk=1):
    model_name, arch = resolve_arch('JiT-B/16', in_context_len=0, **overrides)
    denoiser = build_denoiser(model_name, img_size, 1000, noise_scale=0.5, arch=arch).to(device).train()
    denoiser.net.attn_res_chunk = chunk
    opt = torch.optim.AdamW(denoiser.net.parameters(), lr=1e-4, fused=device_type == 'cuda')
    if compile:
        denoiser.net.compile()
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
            print(f'  {name}: out of memory, micro-batch {micro} x {batch_size // micro}', flush=True)
    ms = (time.perf_counter() - t0) / iters * 1e3
    peak = torch.cuda.max_memory_allocated() / 1e9 if device_type == 'cuda' else float('nan')
    row = dict(name=name, ms=ms, ips=batch_size / ms * 1e3, peak=peak, micro=micro)
    del denoiser, opt
    if device_type == 'cuda':
        torch._dynamo.reset()
    return row

configs = [('plain', {}, 1), ('adaln_single', dict(adaln_single=True), 1), ('attn_res full', dict(attn_res='full'), 1),
           ('full + adaln', dict(attn_res='full', adaln_single=True), 1)]
configs += [(f'full + adaln, chunk {k}', dict(attn_res='full', adaln_single=True), int(k)) for k in chunks.split(',') if k]
configs += [('block + adaln', dict(attn_res='block', adaln_single=True), 1)]
print(f"batch {batch_size} at {img_size}px, {dtype}, compile {compile}; {warmup} warm-up + {iters} timed steps")
rows = [bench(*c) for c in configs]
base = rows[0]
print(f"\n{'config':<26} {'ms/step':>9} {'img/s':>8} {'vs plain':>9} {'peak GB':>8} {'micro':>6}")
for r in rows:
    print(f"{r['name']:<26} {r['ms']:>9.1f} {r['ips']:>8.0f} {r['ms'] / base['ms']:>8.2f}x {r['peak']:>8.1f} {r['micro']:>6}")
