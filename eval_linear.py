"""
Frozen-backbone evaluation on ImageNet-1k: weighted 20-NN + linear probe.

Linear probe follows the DINOv2 protocol: one pass of the frozen backbone feeds a bank of
linear heads that sweep the learning rate x {last 1 / last 4 blocks' cls} x {+ avgpooled patch tokens}.
12500 iters at batch 1024, SGD momentum 0.9, no weight decay, cosine lr, RandomResizedCrop+flip,
and the best head on val is reported. The swept configs include the ones DINO v1 used
(ViT-S: cls of last 4 blocks, ViT-B: cls + avgpool of the last block).

$ python eval_linear.py --ckpt=dino_vits16_100ep/ckpt.pt --data_dir=/data/imagenet
$ python eval_linear.py --ckpt=dino_vits16          # official DINO weights, should give ~74.5 k-NN / ~77.0 linear
$ python eval_linear.py --ckpt=dinov2_vits14        # official DINOv2 weights via torch.hub
$ python eval_linear.py --ckpt=mae_vitb16          # official MAE weights (any name in vit.PRETRAINED works)

JiT checkpoints (train_jit.py ckpt.pt) work too, as the accurate counterpart of train_jit's quick mid-training probes.
Features are the mean-pooled patch tokens after each of jit_blocks, on the input noised to jit_t, with the null label
(as in diagnostics.probe_features); the head bank sweeps lr x {each block, all blocks concatenated}, and k-NN runs per
block. Data is the prepared memmap dataset the model trained on (data_dir/ and data_dir/val/, found from the run's
config.json by default) at the model's resolution; the linear heads train with random flips only, since the memmaps
are pre-cropped (JiT's own augmentation).
$ python eval_linear.py --ckpt=runs/in100_128/jit-b16_icl0/muon/ckpt.pt --jit_t=0.75
"""

import os
import json
import math
import time
import importlib
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data import eval_transform, linear_train_transform, imagenet, make_loader, InfiniteLoader
from evaluation import extract_features, knn_classifier
from vit import PRETRAINED, load_pretrained
from data_jit import MemmapImages, EpochSampler, to_model_input

# -----------------------------------------------------------------------------
# I/O
ckpt = '' # path to a train_dino.py ckpt.pt, a name in vit.PRETRAINED, or 'dinov2_vit{s,b,l,g}14[_reg]'
out_dir = '' # defaults to {ckpt dir}/eval, or eval_{ckpt} for official weights
eval_interval = 2500
log_interval = 100
wandb_log = False
wandb_project = 'cross_patch'
wandb_run_name = '' # defaults to out_dir
# data
data_dir = '' # '' -> /data/imagenet, or for JiT checkpoints the dataset the run trained on
num_workers = 16
batch_size = 1024
eval_batch_size = 512
# k-NN
knn = True
knn_k = 20
knn_temperature = 0.07
# linear probe
linear = True
max_iters = 12500
n_last_blocks = 4
learning_rates = '1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1' # per 256 images, scaled linearly
# JiT checkpoints
jit_t = 1.0 # noise level of the input (t=1 is clean); the mid-training probe heatmap shows which t is best
jit_blocks = '' # comma separated, 1-indexed; '' -> a third, two thirds and the last block (train_jit's probe blocks)
jit_ema = 1 # 1 or 2: which EMA weights, 0: raw weights
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

assert ckpt, 'pass --ckpt=path/to/ckpt.pt (or an official model name)'
is_file = os.path.isfile(ckpt)
out_dir = out_dir or (os.path.join(os.path.dirname(ckpt), 'eval') if is_file else f'eval_{ckpt}')
wandb_run_name = wandb_run_name or out_dir
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

class HubDINOv2(nn.Module):
    """
    Adapts torch.hub DINOv2 models to the encoder interface in vit.py.
    Register tokens are dropped by DINOv2's get_intermediate_layers, so only the cls token is a prefix.
    """
    def __init__(self, name):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', name)
        self.embed_dim = self.model.embed_dim
        self.num_prefix_tokens = 1

    def forward(self, x):
        return self.model(x)

    def get_intermediate_layers(self, x, n=1):
        outputs = self.model.get_intermediate_layers(x, n, return_class_token=True, norm=True)
        return [torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1) for patch_tokens, cls_token in outputs]

class JiTBackbone(nn.Module):
    """
    Adapts a JiT denoiser to the encoder interface: get_intermediate_layers returns the patch tokens after each probe
    block (in-context class tokens dropped) and forward their mean-pooled concatenation. Inputs are images in [-1, 1].
    """
    def __init__(self, net, blocks, t, noise_scale):
        super().__init__()
        self.net = net
        self.blocks = blocks
        self.t = t
        self.noise_scale = noise_scale
        self.embed_dim = net.hidden_size
        self.num_prefix_tokens = 0
        self.captured = {}
        for b in blocks:
            n_strip = net.in_context_len if (net.in_context_len > 0 and b - 1 >= net.in_context_start) else 0
            net.blocks[b - 1].register_forward_hook(lambda m, i, o, b=b, n=n_strip: self.captured.__setitem__(b, o[:, n:]))

    def get_intermediate_layers(self, x, n=None):
        t = torch.full((x.shape[0],), self.t, device=x.device)
        z = self.t * x + (1 - self.t) * self.noise_scale * torch.randn_like(x)
        y = torch.full((x.shape[0],), self.net.num_classes, device=x.device, dtype=torch.long) # null label
        self.net(z, t, y)
        tokens = [self.captured[b] for b in self.blocks]
        self.captured.clear()
        return tokens

    def forward(self, x):
        return torch.cat([tokens.mean(1) for tokens in self.get_intermediate_layers(x)], dim=-1)

class MemmapLoader:
    """
    Batches of a prepared JiT memmap dataset as ([-1, 1] images, labels) on device, in order or reshuffled per epoch.
    """
    def __init__(self, data, batch_size, shuffle=False, flip=False):
        self.data, self.batch_size, self.shuffle, self.flip = data, batch_size, shuffle, flip
        self.sampler = EpochSampler(len(data), batch_size) if shuffle else None

    def __len__(self):
        return len(self.data) // self.batch_size if self.shuffle else math.ceil(len(self.data) / self.batch_size)

    def __iter__(self):
        for i in range(len(self)):
            idx = self.sampler.next() if self.shuffle else np.arange(i * self.batch_size, min((i + 1) * self.batch_size, len(self.data)))
            x, y = self.data.get(idx)
            yield to_model_input(x, self.flip, device), torch.from_numpy(y).to(device)

def load_jit(ckpt, checkpoint):
    from jit_common import build_denoiser
    denoiser = build_denoiser(**checkpoint['denoiser_args'])
    denoiser.load_state_dict(checkpoint[f'model_ema{jit_ema}' if jit_ema else 'model'])
    net = denoiser.net
    n = len(net.blocks)
    blocks = [int(b) for b in jit_blocks.split(',')] if jit_blocks else sorted({max(1, n // 3), max(1, 2 * n // 3), n})
    assert all(1 <= b <= n for b in blocks), f'jit_blocks {blocks} outside 1..{n}'
    print(f"loaded {'EMA' + str(jit_ema) if jit_ema else 'raw'} weights of {checkpoint['denoiser_args']['model']} from {ckpt} "
          f"at iter {checkpoint['iter_num']} (epoch {checkpoint.get('epoch', 0):.1f}): blocks {blocks}, t={jit_t}")
    backbone = JiTBackbone(net, blocks, jit_t, denoiser.noise_scale)
    if compile and 'cuda' in device:
        net.compile() # after the capture hooks are registered
    backbone.iter_num, backbone.epoch = checkpoint['iter_num'], checkpoint.get('epoch')
    return backbone

def jit_data_dir(ckpt, checkpoint):
    # the dataset the run trained on: resolved path in the run's config.json, else data_root/dataset from the checkpoint
    path = os.path.join(os.path.dirname(ckpt), 'config.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)['resolved']['data_dir']
    c = checkpoint['config']
    return c['data_dir'] or os.path.join(c['data_root'], c['dataset'])

def load_backbone(ckpt):
    # returns (backbone, the dataset a JiT checkpoint trained on or None)
    if ckpt in PRETRAINED:
        return load_pretrained(ckpt), None
    elif ckpt.startswith('dinov2_'):
        return HubDINOv2(ckpt), None
    else:
        checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
        if 'denoiser_args' in checkpoint:
            return load_jit(ckpt, checkpoint), jit_data_dir(ckpt, checkpoint)
        module_name, class_name = checkpoint['net'].rsplit('.', 1)
        net = getattr(importlib.import_module(module_name), class_name)
        backbone = net(**checkpoint['model_args'])
        print(f"loaded {checkpoint['net']} from {ckpt} at iter {checkpoint['iter_num']}")
        # evaluate the teacher, as DINO does
        state_dict = checkpoint['teacher']
        # fix the keys of the state dictionary :(
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        state_dict = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    backbone.load_state_dict(state_dict, strict=True)
    return backbone, None

backbone, trained_on = load_backbone(ckpt)
backbone = backbone.to(device).eval()
is_jit = isinstance(backbone, JiTBackbone)
data_dir = data_dir or trained_on or '/data/imagenet'
backbone.requires_grad_(False)
embed_dim = backbone.embed_dim
num_prefix_tokens = backbone.num_prefix_tokens
print(f"backbone: {sum(p.numel() for p in backbone.parameters()) / 1e6:.1f}M params, embed_dim {embed_dim}")

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

results = {'ckpt': ckpt, 'data_dir': data_dir}
suffix = ''
if is_jit:
    results.update(iter=backbone.iter_num, epoch=backbone.epoch, jit_t=jit_t, jit_blocks=backbone.blocks, jit_ema=jit_ema)
    suffix = f'_t{jit_t:g}' + (f'_ema{jit_ema}' if jit_ema != 1 else '') # so evals at several noise levels sit side by side
    n_last_blocks = len(backbone.blocks) # get_intermediate_layers returns exactly the probe blocks
    train_data, val_data = MemmapImages(data_dir), MemmapImages(os.path.join(data_dir, 'val'))
    assert train_data.img_size == backbone.net.input_size, f'{data_dir} is {train_data.img_size}px, the model {backbone.net.input_size}px'
    num_classes = train_data.num_classes
    print(f"data: {data_dir} ({len(train_data):,} train / {len(val_data):,} val images, {num_classes} classes, {train_data.img_size}px)")
else:
    num_classes = 1000

# -----------------------------------------------------------------------------
# k-NN on the full train set

if is_jit:
    val_loader = MemmapLoader(val_data, eval_batch_size)
else:
    val_loader = make_loader(imagenet(data_dir, 'val', eval_transform), eval_batch_size, num_workers, shuffle=False, drop_last=False)

if knn:
    if is_jit:
        knn_train_loader = MemmapLoader(train_data, eval_batch_size)
    else:
        knn_train_loader = make_loader(imagenet(data_dir, 'train', eval_transform), eval_batch_size, num_workers, shuffle=False, drop_last=False)
    train_features, train_labels = extract_features(backbone, knn_train_loader, device, ctx, desc='k-NN train')
    val_features, val_labels = extract_features(backbone, val_loader, device, ctx, desc='k-NN val')
    if is_jit:
        # one k-NN per probe block: split the concatenated block features and re-normalize each
        knn_runs = {f'block{b}': (F.normalize(train_features.chunk(len(backbone.blocks), dim=1)[i], dim=-1),
                                  F.normalize(val_features.chunk(len(backbone.blocks), dim=1)[i], dim=-1))
                    for i, b in enumerate(backbone.blocks)}
    else:
        knn_runs = {'': (train_features, val_features)}
    for name, (tr, va) in knn_runs.items():
        top1, top5 = knn_classifier(tr, train_labels, va, val_labels, k=knn_k, T=knn_temperature, num_classes=num_classes)
        print(f"{knn_k}-NN{' ' + name if name else ''}: top1 {top1:.2f}%, top5 {top5:.2f}%")
        if name:
            results[f'knn_top1_{name}'], results[f'knn_top5_{name}'] = top1, top5
        if 'knn_top1' not in results or top1 > results['knn_top1']:
            results['knn_top1'], results['knn_top5'] = top1, top5
            if name:
                results['knn_best_block'] = name
    if wandb_log:
        wandb.log({'knn_top1': top1, 'knn_top5': top5})
    del train_features, train_labels, val_features, val_labels, knn_train_loader, knn_runs

# -----------------------------------------------------------------------------
# linear probe sweep

class LinearClassifier(nn.Module):
    def __init__(self, use_n_blocks, use_avgpool, num_classes=1000, pooled_blocks=None):
        """
        pooled_blocks: indices into the token list whose mean-pooled tokens form the features (JiT, which has no cls
        token); otherwise the cls tokens of the last use_n_blocks, optionally with the avgpooled last-block patch tokens.
        """
        super().__init__()
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.pooled_blocks = pooled_blocks
        in_dim = embed_dim * (len(pooled_blocks) if pooled_blocks is not None else use_n_blocks + int(use_avgpool))
        self.linear = nn.Linear(in_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        if self.pooled_blocks is not None:
            return self.linear(torch.cat([x_tokens_list[i].mean(1) for i in self.pooled_blocks], dim=-1))
        intermediate = x_tokens_list[-self.use_n_blocks:]
        x = torch.cat([tokens[:, 0] for tokens in intermediate], dim=-1)
        if self.use_avgpool:
            x = torch.cat([x, x_tokens_list[-1][:, num_prefix_tokens:].mean(1)], dim=-1)
        return self.linear(x)

if linear:
    lrs = [float(lr) for lr in learning_rates.split(',')]
    heads = nn.ModuleDict()
    param_groups = []
    if is_jit:
        # each probe block on its own, and all of them concatenated
        configs = {f'block{b}': [i] for i, b in enumerate(backbone.blocks)}
        if len(backbone.blocks) > 1:
            configs['blocks' + '+'.join(map(str, backbone.blocks))] = list(range(len(backbone.blocks)))
        for feat_name, pooled in configs.items():
            for lr in lrs:
                name = f'{feat_name}_lr{lr:.0e}'
                heads[name] = LinearClassifier(None, False, num_classes, pooled_blocks=pooled)
                param_groups.append({'params': heads[name].parameters(), 'base_lr': lr * batch_size / 256})
    else:
        for n in sorted({1, n_last_blocks}):
            for avgpool in (False, True):
                for lr in lrs:
                    name = f'blocks{n}_avgpool{int(avgpool)}_lr{lr:.0e}'
                    heads[name] = LinearClassifier(n, avgpool, num_classes)
                    param_groups.append({'params': heads[name].parameters(), 'base_lr': lr * batch_size / 256})
    heads.to(device)
    optimizer = torch.optim.SGD(param_groups, lr=0.0, momentum=0.9, weight_decay=0.0)
    print(f"training {len(heads)} linear heads for {max_iters} iters")

    if is_jit:
        assert len(train_data) >= batch_size, f'batch_size {batch_size} is larger than the {len(train_data)}-image dataset'
        train_loader = InfiniteLoader(MemmapLoader(train_data, batch_size, shuffle=True, flip=True))
    else:
        train_loader = InfiniteLoader(make_loader(imagenet(data_dir, 'train', linear_train_transform), batch_size, num_workers, shuffle=True, drop_last=True))

    def get_batch():
        x, y = train_loader.next()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    features_fn = backbone.get_intermediate_layers
    if compile and 'cuda' in device and not is_jit: # the JiT network is compiled in load_jit
        print("compiling the backbone... (takes a ~minute)")
        features_fn = torch.compile(features_fn)

    @torch.no_grad()
    def get_features(x):
        with ctx:
            tokens = features_fn(x, n_last_blocks)
        # heads train in fp32 on top of the bf16 backbone
        return [t.float() for t in tokens]

    @torch.no_grad()
    def evaluate():
        heads.eval()
        correct = {name: torch.zeros((), device=device) for name in heads}
        total = 0
        for x, y in tqdm(val_loader, desc='linear val'):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            tokens = get_features(x)
            for name, head in heads.items():
                correct[name] += (head(tokens).argmax(-1) == y).sum()
            total += y.shape[0]
        heads.train()
        return {name: 100.0 * c.item() / total for name, c in correct.items()}

    def get_lr_coeff(it):
        # cosine decay to 0 (no warmup), as in DINOv2
        return 0.5 * (1.0 + math.cos(math.pi * it / max_iters))

    X, Y = get_batch()
    assert len(get_features(X[:2])) == n_last_blocks, f'backbone returned fewer than n_last_blocks={n_last_blocks} layers'
    t0 = time.time()
    for iter_num in range(max_iters + 1):
        if iter_num > 0 and (iter_num % eval_interval == 0 or iter_num == max_iters):
            accs = evaluate()
            best_name = max(accs, key=accs.get)
            print(f"iter {iter_num}: best linear top1 {accs[best_name]:.2f}% ({best_name})")
            if wandb_log:
                wandb.log({'iter': iter_num, 'linear_top1': accs[best_name]}, step=iter_num)
        if iter_num == max_iters:
            break

        coeff = get_lr_coeff(iter_num)
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * coeff

        tokens = get_features(X)
        X_next, Y_next = get_batch()
        # heads are independent, so summing their losses trains each one with its own lr
        loss = sum(F.cross_entropy(head(tokens), Y) for head in heads.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        X, Y = X_next, Y_next

        if iter_num % log_interval == 0:
            t1 = time.time()
            dt = (t1 - t0) / (log_interval if iter_num > 0 else 1)
            t0 = t1
            lossf = loss.item() / len(heads)
            print(f"iter {iter_num}: mean head loss {lossf:.4f}, {dt*1000:.0f}ms/iter, {batch_size / dt:.0f} img/s")
            if wandb_log:
                wandb.log({'iter': iter_num, 'train/mean_head_loss': lossf, 'lr_coeff': coeff}, step=iter_num)

    if is_jit: # best head per feature set, so the block comparison is visible at a glance
        for feat_name in configs:
            best = max((n for n in accs if n.rsplit('_lr', 1)[0] == feat_name), key=accs.get)
            results[f'linear_top1_{feat_name}'] = accs[best]
            print(f"  {feat_name:24s} {accs[best]:.2f} ({best.rsplit('_', 1)[1]})")
    print("\nall heads (val top1):")
    for name, acc in sorted(accs.items(), key=lambda kv: -kv[1]):
        print(f"  {name:32s} {acc:.2f}")
    results['linear'] = accs
    results['linear_top1'] = accs[best_name]
    results['linear_best_head'] = best_name
    torch.save(heads.state_dict(), os.path.join(out_dir, f'linear_heads{suffix}.pt'))

with open(os.path.join(out_dir, f'results{suffix}.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n{ckpt}: " + ', '.join(f"{k} {results[k]:.2f}" for k in ('knn_top1', 'linear_top1') if k in results))
print(f"results written to {os.path.join(out_dir, f'results{suffix}.json')}")
