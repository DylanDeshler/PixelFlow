"""
FID / IS evaluation of a JiT checkpoint, with a CFG sweep. Works on train_jit.py checkpoints and on
JiT's released checkpoints (checkpoint-*.pth from their README), so the pipeline can be checked
against the paper before trusting it on your own runs.

$ python eval_jit.py --ckpt=runs/in100_128/jit-b16/base/ckpt.pt --data_dir=/data/in100_128 --cfgs=1.0,1.5,2.0,2.5,3.0
$ python eval_jit.py --ckpt=/path/to/jit-b-16/checkpoint-last.pth --data_dir=/data/in1k_256 --cfgs=3.0 --interval_min=0.1 --num_images=50000

CFG sweep: --cfgs takes a comma list or an inclusive start:stop:step range. It reports the best-FID scale and a
trade-off pick (highest IS among scales within --fid_tolerance of the best FID), the FID/IS Pareto frontier, and
plots FID, IS, precision/recall vs CFG and the FID-IS trade-off curve. Sweep cheaply, then re-score the picks:
$ python eval_jit.py --ckpt=... --data_dir=/data/in100_128 --cfgs=1.0:4.0:0.25 --num_images=10000 --final_num_images=50000

--save_samples keeps every generated image, one output per cfg, in a standard format:
    png   folder of PNGs named {index}_class{label}.png, as JiT's evaluation writes (works with their torch-fidelity CLI)
    npz   uint8 'arr_0' [N, H, W, 3] (OpenAI's ADM evaluator format) plus int 'labels' [N]
and a {name}_classes.json mapping label -> ImageNet-1k index and class name.
"""

import os
import json
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image

from data_jit import MemmapImages
from jit_common import build_denoiser, generate_and_score, save_labeled_grid
from fid import InceptionMetrics, load_ref_stats, precision_recall

from plot_style import SURFACE, INK, INK_2, GRID, BLUE, ORANGE # validated categorical pair on the light surface

def plot_cfg_sweep(results, cfgs, path, title):
    """
    Small multiples (one y-scale per panel): FID vs CFG, IS vs CFG, precision / recall vs CFG (if available), and the
    FID-IS trade-off curve. The best-FID and trade-off scales are marked and labeled.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fid = [results['fid'][str(c)] for c in cfgs]
    isc = [results['is'][str(c)] for c in cfgs]
    has_pr = bool(results['precision'])
    best, trade = results['best_fid_cfg'], results['tradeoff_cfg']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    axes = axes.flatten()

    def style(ax, xlabel, ylabel, panel_title):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.set_title(panel_title, color=INK, fontsize=10, loc='left')

    def mark(ax, x, y, text, kind):
        # circle = best FID, diamond = trade-off pick; labels offset in opposite directions so adjacent picks don't collide
        marker, offset, ha = ('o', (8, -15), 'left') if kind == 'best' else ('D', (-8, 10), 'right')
        ax.plot([x], [y], marker, color=BLUE, markersize=9 if kind == 'best' else 8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        if text:
            ax.annotate(text, (x, y), xytext=offset, textcoords='offset points', color=INK, fontsize=8, ha=ha)

    # 1. FID vs CFG
    ax = axes[0]
    ax.plot(cfgs, fid, '-o', color=BLUE, linewidth=2, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    mark(ax, best, results['fid'][str(best)], f"best FID {results['fid'][str(best)]:.2f} @ cfg {best:g}", 'best')
    if trade != best:
        mark(ax, trade, results['fid'][str(trade)], f"trade-off @ cfg {trade:g}", 'trade')
    ax.margins(y=0.15)
    style(ax, 'CFG scale', 'FID (lower is better)', 'FID vs CFG  (circle: best FID, diamond: trade-off pick)')
    # 2. IS vs CFG
    ax = axes[1]
    ax.plot(cfgs, isc, '-o', color=BLUE, linewidth=2, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    if trade != best:
        mark(ax, best, results['is'][str(best)], f"best FID @ cfg {best:g}: IS {results['is'][str(best)]:.1f}", 'best')
    mark(ax, trade, results['is'][str(trade)], f"trade-off @ cfg {trade:g}: IS {results['is'][str(trade)]:.1f}", 'trade')
    ax.margins(y=0.15)
    style(ax, 'CFG scale', 'Inception Score (higher is better)', 'IS vs CFG')
    # 3. precision / recall vs CFG
    ax = axes[2]
    if has_pr:
        prec = [results['precision'][str(c)] for c in cfgs]
        rec = [results['recall'][str(c)] for c in cfgs]
        ax.plot(cfgs, prec, '-o', color=BLUE, linewidth=2, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label='precision (fidelity)')
        ax.plot(cfgs, rec, '-o', color=ORANGE, linewidth=2, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label='recall (coverage)')
        for series, name in ((prec, 'precision'), (rec, 'recall')):
            ax.annotate(name, (cfgs[-1], series[-1]), xytext=(6, 0), textcoords='offset points', color=INK_2, fontsize=8, va='center')
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc='best')
        ax.set_ylim(0, 1)
        style(ax, 'CFG scale', 'fraction', 'precision / recall vs CFG')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'precision / recall need train_inception.npy (fid.py)', ha='center', va='center', color=INK_2, fontsize=9)
    # 4. FID vs IS trade-off curve, in CFG order
    ax = axes[3]
    ax.plot(isc, fid, '-o', color=BLUE, linewidth=2, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for c, x, y in zip(cfgs, isc, fid):
        if c in (cfgs[0], cfgs[-1]) and c not in (best, trade):
            ax.annotate(f'cfg {c:g}', (x, y), xytext=(6, 4), textcoords='offset points', color=INK_2, fontsize=8)
    mark(ax, results['is'][str(best)], results['fid'][str(best)], f'best FID: cfg {best:g}', 'best')
    if trade != best:
        mark(ax, results['is'][str(trade)], results['fid'][str(trade)], f'trade-off: cfg {trade:g}', 'trade')
    ax.margins(x=0.08, y=0.15)
    style(ax, 'Inception Score (higher is better)', 'FID (lower is better)', 'FID-IS trade-off (lower right is better)')

    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path

# -----------------------------------------------------------------------------
ckpt = ''
data_dir = '' # prepared dataset dir, used for the FID reference stats and class count
out_dir = '' # defaults to the checkpoint's dir
num_images = 50000
cfgs = '1.0' # guidance scales: comma list (1.0,2.0,3.0) or inclusive range start:stop:step (1.0:4.0:0.25)
fid_tolerance = 0.05 # trade-off pick: highest IS among scales with FID <= (1 + tol) * best FID
final_num_images = 0 # > 0: re-score the best-FID and trade-off scales with this many images (e.g. 50000)
plot = True # save the CFG sweep plot (needs >= 2 scales)
interval_min = 0.0
interval_max = 1.0
sampling_method = 'heun'
num_sampling_steps = 50
time_schedule = 'uniform' # 'uniform' (JiT), 'shift' (FLUX/SD3-style, uses the checkpoint's P_mean), 'train_quantile'
gen_batch_size = 256
ema = 1 # which EMA to sample with (JiT uses the first, decay 0.9999)
seed = 0
save_samples = '' # '', 'png' or 'npz'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

assert ckpt and data_dir, 'pass --ckpt and --data_dir'
if not compile:
    torch._dynamo.config.disable = True
out_dir = out_dir or os.path.dirname(os.path.abspath(ckpt))
torch.backends.cuda.matmul.allow_tf32 = True # same matmul precision as training (and silences torch.compile's TF32 warning)
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' or dtype == 'float32') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
if 'denoiser_args' in checkpoint:
    denoiser_args = dict(checkpoint['denoiser_args'])
else:
    # released JiT checkpoint: rebuild from its argparse namespace
    a = checkpoint['args']
    denoiser_args = dict(model=a.model, img_size=a.img_size, num_classes=a.class_num, P_mean=a.P_mean, P_std=a.P_std,
                         t_eps=a.t_eps, noise_scale=a.noise_scale)
denoiser = build_denoiser(**denoiser_args).to(device)
denoiser.load_state_dict(checkpoint[f'model_ema{ema}'])
denoiser.time_schedule = time_schedule
print(f"loaded EMA{ema} weights of {denoiser_args['model']} from {ckpt}")
checkpoint = None

data = MemmapImages(data_dir)
assert data.num_classes == denoiser.num_classes and data.img_size == denoiser.img_size, 'checkpoint and dataset disagree'
ref_stats_path = os.path.join(data_dir, 'fid_stats_jit.npz')
if not os.path.exists(ref_stats_path):
    ref_stats_path = os.path.join(data_dir, 'fid_stats.npz')
ref_stats = load_ref_stats(ref_stats_path)
class_names = data.class_names()
metrics = InceptionMetrics(device)
print(f"FID reference: {ref_stats_path}")

# precision / recall against an equally sized random subset of the training features, if fid.py stored them
feats_path = os.path.join(data_dir, 'train_inception.npy')
train_feats = np.load(feats_path, mmap_mode='r') if os.path.exists(feats_path) else None

def real_features(n):
    if train_feats is None:
        return None
    idx = np.sort(np.random.default_rng(0).choice(len(train_feats), min(n, len(train_feats)), replace=False))
    return torch.from_numpy(train_feats[idx].astype(np.float32)).to(device)

def parse_cfgs(spec):
    if ':' in spec:
        start, stop, step = (float(v) for v in spec.split(':'))
        return [float(round(v, 6)) for v in np.arange(start, stop + step / 2, step)]
    return [float(c) for c in spec.split(',')]

real_feats = real_features(num_images)

results = {'ckpt': ckpt, 'num_images': num_images, 'steps': num_sampling_steps, 'method': sampling_method, 'time_schedule': time_schedule,
           'interval': [interval_min, interval_max], 'fid': {}, 'is': {}, 'precision': {}, 'recall': {}}
class SampleWriter:
    """
    Receives every generated uint8 batch [B, 3, H, W] and writes it out as PNGs or into an npz.
    """
    def __init__(self, path, kind, n, size):
        self.path, self.kind, self.count = path, kind, 0
        if kind == 'png':
            os.makedirs(path, exist_ok=True)
        else:
            # fill a temporary .npy memmap, then stream it into the npz at the end (50k at 256px is ~10GB)
            self.tmp = path + '.tmp.npy'
            self.arr = np.lib.format.open_memmap(self.tmp, mode='w+', dtype=np.uint8, shape=(n, size, size, 3))
            self.labels = np.zeros(n, dtype=np.int64)
        base = path[:-4] if path.endswith('.npz') else path
        with open(base + '_classes.json', 'w') as f:
            json.dump({str(i): {'imagenet_index': idx, 'name': name} for i, (idx, name) in
                       enumerate(zip(data.meta.get('imagenet_index') or list(range(0, 1000, 1000 // data.num_classes)), class_names))}, f, indent=1)

    def __call__(self, x_uint8, labels):
        x = x_uint8.permute(0, 2, 3, 1).cpu().numpy()
        labels = labels.tolist()
        if self.kind == 'png':
            for i, (img, label) in enumerate(zip(x, labels)):
                Image.fromarray(img).save(os.path.join(self.path, f'{self.count + i:05d}_class{label:04d}.png'))
        else:
            self.arr[self.count:self.count + len(x)] = x
            self.labels[self.count:self.count + len(x)] = labels
        self.count += len(x)

    def close(self):
        if self.kind == 'npz':
            self.arr.flush()
            np.savez(self.path, arr_0=self.arr, labels=self.labels)
            del self.arr
            os.remove(self.tmp)
        print(f"saved {self.count:,} samples to {self.path}")

def evaluate_cfg(cfg, n, real, save=True):
    writer = None
    if save and save_samples:
        assert save_samples in ('png', 'npz'), "save_samples must be '', 'png' or 'npz'"
        name = f"samples_{n / 1000:g}k_{sampling_method}{num_sampling_steps}{'' if time_schedule == 'uniform' else '_' + time_schedule}_cfg{cfg}_interval{interval_min}-{interval_max}"
        writer = SampleWriter(os.path.join(out_dir, name + ('.npz' if save_samples == 'npz' else '')), save_samples, n, data.img_size)
    out = generate_and_score(
        denoiser, metrics, ref_stats, n, data.num_classes, device, ctx,
        cfg=cfg, interval=(interval_min, interval_max), steps=num_sampling_steps,
        method=sampling_method, batch_size=gen_batch_size, seed=seed, on_batch=writer,
    )
    if writer is not None:
        writer.close()
    r = {'fid': out['fid'], 'is': out['is']}
    if real is not None:
        r['precision'], r['recall'] = precision_recall(real, out['features'].to(device))
    pr_str = f", precision {r['precision']:.3f}, recall {r['recall']:.3f}" if 'precision' in r else ''
    print(f"cfg {cfg}: FID-{n / 1000:g}k {r['fid']:.3f}, IS {r['is']:.1f}{pr_str}")
    return r

cfg_list = parse_cfgs(cfgs)
for cfg in cfg_list:
    r = evaluate_cfg(cfg, num_images, real_feats)
    for k, v in r.items():
        results[k][str(cfg)] = v

# selections
fids = {float(c): v for c, v in results['fid'].items()}
iss = {float(c): v for c, v in results['is'].items()}
best_fid_cfg = min(fids, key=fids.get)
within = [c for c in fids if fids[c] <= (1 + fid_tolerance) * fids[best_fid_cfg]]
tradeoff_cfg = max(within, key=iss.get)
pareto = sorted(c for c in fids if not any(fids[o] <= fids[c] and iss[o] >= iss[c] and (fids[o], iss[o]) != (fids[c], iss[c]) for o in fids))
results.update({'best_fid_cfg': best_fid_cfg, 'tradeoff_cfg': tradeoff_cfg, 'fid_tolerance': fid_tolerance, 'pareto_cfgs': pareto})

print(f"\n  cfg     FID       IS" + ("    precision  recall" if results['precision'] else ""))
for c in cfg_list:
    marks = ' '.join(m for m, ok in (('<- best FID', c == best_fid_cfg), (f'<- trade-off (FID within {fid_tolerance:.0%})', c == tradeoff_cfg),
                                       ('pareto', c in pareto)) if ok)
    pr = f"   {results['precision'][str(c)]:.3f}    {results['recall'][str(c)]:.3f}" if results['precision'] else ''
    print(f"  {c:<5g}  {fids[c]:8.3f}  {iss[c]:7.1f}{pr}   {marks}")
print(f"best FID: cfg {best_fid_cfg:g} (FID {fids[best_fid_cfg]:.3f}, IS {iss[best_fid_cfg]:.1f}); "
      f"trade-off: cfg {tradeoff_cfg:g} (FID {fids[tradeoff_cfg]:.3f}, IS {iss[tradeoff_cfg]:.1f})")

# optional final re-scoring of the picks with more samples
if final_num_images > 0:
    final_real = real_features(final_num_images)
    results['final'] = {'num_images': final_num_images}
    for c in sorted({best_fid_cfg, tradeoff_cfg}):
        results['final'][str(c)] = evaluate_cfg(c, final_num_images, final_real, save=False)

best_cfg = best_fid_cfg
# a sample grid at the best cfg
denoiser.cfg_scale = float(best_cfg)
labels = torch.linspace(0, data.num_classes - 1, 64).long().to(device)
torch.manual_seed(seed)
with torch.no_grad(), ctx:
    x = denoiser.generate(labels)
tag = f"{num_images / 1000:g}k_{sampling_method}{num_sampling_steps}" + ('' if time_schedule == 'uniform' else f'_{time_schedule}')
save_labeled_grid(x, [class_names[i] for i in labels.tolist()], os.path.join(out_dir, f'eval_samples_{tag}.png'))
with open(os.path.join(out_dir, f'eval_{tag}.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"results written to {os.path.join(out_dir, f'eval_{tag}.json')}")
if plot and len(cfg_list) >= 2:
    path = plot_cfg_sweep(results, cfg_list, os.path.join(out_dir, f'cfg_sweep_{tag}.png'),
                          f"{os.path.basename(os.path.dirname(os.path.abspath(ckpt)))}: CFG sweep, FID-{num_images / 1000:g}k, "
                          f"{sampling_method} {num_sampling_steps} steps, interval [{interval_min}, {interval_max}]")
    print(f"plot saved to {path}")
