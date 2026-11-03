"""
Where to apply the Dispersive Loss, from a finished checkpoint (no training). For every block, on held-out images noised as
in training (t from the run's logit-normal, labels dropped at its rate), one forward / backward pass of the diffusion loss:
  rms          the block output's RMS (feature scale: the loss's D is a per-dimension squared distance)
  disp[tau]    the InfoNCE-L2 dispersive loss at several temperatures, and its position between the two ends,
               pos = (L - log(1/B)) / (0 - log(1/B)): ~0 means already maximally dispersed for that tau (saturated: no
               gradient), ~1 means collapsed
  |g_disp|     per-sample norm of the dispersive loss's gradient w.r.t. the block output (at --tau)
  |g_diff|     the same for the diffusion loss (backpropagated from the output through the later blocks): the objective the run
               trained on, so with a learned per-noise-level weight (--loss_t_weight=learned, restored from the checkpoint)
               the weighted loss, whose gradient is ~10-20x the plain v-loss's in EDM2's unnormalized form
  share@w      the dispersive share of the combined gradient at loss weight w: w |g_disp| / (w |g_disp| + |g_diff|), what a
               fixed weight amounts to at that depth (and a starting point for --disp_balance)
The batch size sets the loss's floor log(1/B), so use the training micro-batch (1024 for the in1k runs) for numbers that
match training. Prints a table; writes {run}/diagnostics/dispersion_{iter}_{weights}.json.

$ python analyze_dispersion.py --ckpts=runs/in1k_128/jit-b16_icl0/muon_losst-learned --data_dir=/data/in1k_128
"""

import os
import json
import math
from contextlib import nullcontext

import numpy as np
import torch

from data_jit import MemmapImages, to_model_input
from jit_common import load_denoiser
from repa import dispersive_loss
from loss_weighting import LossWeighting

# -----------------------------------------------------------------------------
ckpts = '' # comma list of run directories (their ckpt.pt) or checkpoint files
data_dir = ''
ema = 0 # 0 = raw weights (what training steps from), 1 / 2 = that EMA
batch_size = 1024 # one batch; the loss's floor is log(1/B)
taus = '0.25,0.5,1.0,2.0'
tau = 0.5 # for the gradient norms
weight = 0.5 # for the share column (the paper's lambda)
kind = 'infonce_l2' # repa.dispersive_loss variant (infonce_l2n: on unit-RMS features, scale-invariant)
seed = 0
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

assert ckpts and data_dir, 'pass --ckpts and --data_dir'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' or dtype == 'float32') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
tau_list = [float(v) for v in taus.split(',') if v]

val = MemmapImages(os.path.join(data_dir, 'val'))
idx = np.linspace(0, len(val) - 1, min(batch_size, len(val))).round().astype(np.int64)
xs_np, ys_np = val.get(idx)

for spec in ckpts.split(','):
    path = os.path.join(spec, 'ckpt.pt') if os.path.isdir(spec) else spec
    denoiser, meta = load_denoiser(path, ema, device)
    net = denoiser.net.train() # as in training (dropout etc. off in JiT-B anyway)
    blocks = list(net.blocks)
    n_strip = lambda i: net.in_context_len if (net.in_context_len > 0 and i >= net.in_context_start) else 0
    outs = {}
    def make_hook(i):
        def hook(module, inputs, output):
            output.retain_grad()
            outs[i] = output
        return hook
    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]
    x = to_model_input(xs_np, flip=False, device=device)
    y = torch.from_numpy(ys_np).to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    B = len(x)
    t = torch.sigmoid(torch.randn(B, generator=g, device=device) * denoiser.P_std + denoiser.P_mean).view(-1, 1, 1, 1)
    drop = torch.rand(B, generator=g, device=device) < denoiser.label_drop_prob
    y = torch.where(drop, torch.full_like(y, denoiser.num_classes), y)
    z = t * x + (1 - t) * torch.randn(x.shape, generator=g, device=device) * denoiser.noise_scale
    with ctx:
        x_pred = net(z, t.flatten(), y)
    r = (x - x_pred.float()) / (1 - t).clamp_min(denoiser.t_eps) # v - v_pred
    ck = torch.load(path, map_location='cpu', weights_only=False)
    run_cfg, lw_state = ck.get('config', {}), ck.get('loss_weighting')
    del ck
    if lw_state is not None and run_cfg.get('loss_t_weight', 'none') != 'none' and run_cfg.get('loss_freq_weight', 'none') == 'none':
        # the training objective: the learned per-noise-level weight as trained (frequency weighting isn't rebuilt here)
        weighter = LossWeighting(denoiser.noise_scale, t_weight=run_cfg['loss_t_weight'],
                                 normalize_learned=run_cfg.get('loss_weight_normalize', True)).to(device)
        weighter.load_state_dict(lw_state)
        loss = weighter(r, t.flatten())
        objective = f"learned per-noise-level weight ({'normalized' if run_cfg.get('loss_weight_normalize', True) else 'EDM2 unnormalized'})"
    else:
        loss = r.pow(2).mean() # the plain training loss (v-loss)
        objective = 'plain v-loss' + (' (frequency weighting not rebuilt)' if run_cfg.get('loss_freq_weight', 'none') != 'none' else '')
    loss.backward()
    for h in handles:
        h.remove()
    floor = math.log(1.0 / B)
    rows = []
    for i in range(len(blocks)):
        h_full = outs[i]
        h = h_full[:, n_strip(i):].detach().float()
        g_diff = h_full.grad[:, n_strip(i):].float().flatten(1).norm(dim=1).mean().item()
        vals = {tv: dispersive_loss(h, tv, kind).item() for tv in tau_list}
        hg = h.clone().requires_grad_(True)
        gd, = torch.autograd.grad(dispersive_loss(hg, tau, kind), hg)
        g_disp = gd.flatten(1).norm(dim=1).mean().item()
        rows.append({'block': i + 1, 'rms': h.pow(2).mean().sqrt().item(), 'disp': vals,
                     'pos': {tv: (v - floor) / (0 - floor) for tv, v in vals.items()},
                     'g_disp': g_disp, 'g_diff': g_diff, 'share': weight * g_disp / (weight * g_disp + g_diff + 1e-30)})
    run = os.path.basename(os.path.dirname(os.path.abspath(path)))
    it = meta.get('iter_num')
    print(f"\n=== {run} (iter {it}, {'ema' + str(ema) if ema else 'raw'} weights, batch {B}: loss floor log(1/B) = {floor:.2f}; "
          f"dispersive form {kind}; diffusion gradient from the {objective})")
    print(f"{'block':>5} {'rms':>7} " + ''.join(f"{'pos@' + str(tv):>9}" for tv in tau_list) + f"  {'|g_disp|':>9} {'|g_diff|':>9} {'share@' + str(weight):>10}")
    for r in rows:
        print(f"{r['block']:>5} {r['rms']:>7.3f} " + ''.join(f"{r['pos'][tv]:>9.3f}" for tv in tau_list)
              + f"  {r['g_disp']:>9.2e} {r['g_diff']:>9.2e} {r['share']:>10.3f}")
    print("pos: 0 = saturated (maximally dispersed for that tau, no gradient), 1 = collapsed. share: the dispersive part of the"
          f" combined gradient at the block, at weight {weight} and tau {tau}")
    out = os.path.join(os.path.dirname(os.path.abspath(path)), 'diagnostics')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"dispersion_{it}_{'ema' + str(ema) if ema else 'raw'}.json"), 'w') as f:
        json.dump({'run': run, 'iter': it, 'batch': B, 'rows': rows, 'config': config}, f)
    del denoiser, net, outs
    if device_type == 'cuda':
        torch.cuda.empty_cache()
