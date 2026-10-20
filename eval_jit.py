"""
FID / IS evaluation of a JiT checkpoint, with a CFG sweep. Works on train_jit.py checkpoints and on
JiT's released checkpoints (checkpoint-*.pth from their README), so the pipeline can be checked
against the paper before trusting it on your own runs.

$ python eval_jit.py --ckpt=runs/in100_128/jit-b16/base/ckpt.pt --data_dir=/data/in100_128 --cfgs=1.0,1.5,2.0,2.5,3.0
$ python eval_jit.py --ckpt=/path/to/jit-b-16/checkpoint-last.pth --data_dir=/data/in1k_256 --cfgs=3.0 --interval_min=0.1 --num_images=50000

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

# -----------------------------------------------------------------------------
ckpt = ''
data_dir = '' # prepared dataset dir, used for the FID reference stats and class count
out_dir = '' # defaults to the checkpoint's dir
num_images = 50000
cfgs = '1.0' # comma separated guidance scales to sweep
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
real_feats = None
feats_path = os.path.join(data_dir, 'train_inception.npy')
if os.path.exists(feats_path):
    train_feats = np.load(feats_path, mmap_mode='r')
    idx = np.sort(np.random.default_rng(0).choice(len(train_feats), min(num_images, len(train_feats)), replace=False))
    real_feats = torch.from_numpy(train_feats[idx].astype(np.float32)).to(device)

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

for cfg in [float(c) for c in cfgs.split(',')]:
    writer = None
    if save_samples:
        assert save_samples in ('png', 'npz'), "save_samples must be '', 'png' or 'npz'"
        name = f"samples_{num_images / 1000:g}k_{sampling_method}{num_sampling_steps}{'' if time_schedule == 'uniform' else '_' + time_schedule}_cfg{cfg}_interval{interval_min}-{interval_max}"
        writer = SampleWriter(os.path.join(out_dir, name + ('.npz' if save_samples == 'npz' else '')), save_samples, num_images, data.img_size)
    out = generate_and_score(
        denoiser, metrics, ref_stats, num_images, data.num_classes, device, ctx,
        cfg=cfg, interval=(interval_min, interval_max), steps=num_sampling_steps,
        method=sampling_method, batch_size=gen_batch_size, seed=seed, on_batch=writer,
    )
    if writer is not None:
        writer.close()
    results['fid'][str(cfg)], results['is'][str(cfg)] = out['fid'], out['is']
    pr_str = ''
    if real_feats is not None:
        precision, recall = precision_recall(real_feats, out['features'].to(device))
        results['precision'][str(cfg)], results['recall'][str(cfg)] = precision, recall
        pr_str = f", precision {precision:.3f}, recall {recall:.3f}"
    print(f"cfg {cfg}: FID-{num_images / 1000:g}k {out['fid']:.3f}, IS {out['is']:.1f}{pr_str}")

best_cfg = min(results['fid'], key=results['fid'].get)
print(f"best: cfg {best_cfg}, FID {results['fid'][best_cfg]:.3f}")

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
