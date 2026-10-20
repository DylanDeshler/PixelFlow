"""
DINO self-supervised pretraining on ImageNet-1k, single GPU.

Defaults reproduce the official DINO ViT-S/16 100 epoch recipe (main_dino.py defaults,
effective batch 512), which the DINO repo reports at 69.3% k-NN / 74.0% linear.
Swap `net` below for your own encoder (see the interface in vit.py) to compare against it.

$ python train_dino.py --data_dir=/data/imagenet
$ python train_dino.py --out_dir=my_idea --batch_size=128 --gradient_accumulation_steps=4
$ python train_dino.py --out_dir=my_idea --init_from=resume   # after preemption

Final numbers come from eval_linear.py on the saved checkpoint.
"""

import os
import time
import math
import uuid
import copy
from contextlib import nullcontext
from torchinfo import summary

import torch
import torch.nn.functional as F

from vit import VisionTransformer as net
from vit import DINOHead, MultiCropWrapper, DINOLoss
from data import DataAugmentationDINO, eval_transform, imagenet, make_loader, InfiniteLoader
from evaluation import extract_features, knn_classifier

# -----------------------------------------------------------------------------
# I/O
out_dir = 'dino_vits16_100ep'
eval_interval = 25000 # k-NN eval every ~10 epochs at the default batch size
log_interval = 100
save_interval = 2500 # overwrite ckpt.pt roughly once an epoch, so a preempted run loses little
eval_only = False # if True, script exits right after the first k-NN eval
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = True
wandb_project = 'cross_patch'
wandb_run_name = '' # defaults to out_dir
# data
data_dir = '/data/imagenet'
num_workers = 16 # multi-crop decoding is CPU heavy, give it as many cores as you have
gradient_accumulation_steps = 2
batch_size = 256 # micro batch; effective batch = batch_size * gradient_accumulation_steps
global_crops_scale_min = 0.4
local_crops_scale_max = 0.4
local_crops_number = 8
# k-NN monitoring (teacher backbone, every train_stride-th train image as the memory bank, full val set)
knn_train_stride = 10
knn_k = 20
knn_temperature = 0.07
eval_batch_size = 512
# model
patch_size = 16
embed_dim = 384
depth = 12
num_heads = 6
mlp_ratio = 4.0
drop_path_rate = 0.1
# dino head + loss
out_dim = 65536
norm_last_layer = True # True is more stable, False can give a bit better performance (DINO README)
use_bn_in_head = False
warmup_teacher_temp = 0.04
teacher_temp = 0.04
warmup_teacher_temp_epochs = 0
momentum_teacher = 0.996 # cosine scheduled to 1.0
freeze_last_layer_epochs = 1
# adamw optimizer
epochs = 100
learning_rate = 5e-4 # per 256 images, scaled linearly with the effective batch size
min_lr = 1e-6
warmup_epochs = 10
weight_decay = 0.04 # cosine scheduled to weight_decay_end
weight_decay_end = 0.4
beta1 = 0.9
beta2 = 0.999
grad_clip = 3.0 # clip gradients at this value, or disable if == 0.0
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

wandb_run_name = wandb_run_name or out_dir
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data
train_dataset = imagenet(data_dir, 'train', DataAugmentationDINO(
    global_crops_scale=(global_crops_scale_min, 1.0),
    local_crops_scale=(0.05, local_crops_scale_max),
    local_crops_number=local_crops_number,
))
train_loader = InfiniteLoader(make_loader(train_dataset, batch_size, num_workers, shuffle=True, drop_last=True))

knn_train_dataset = imagenet(data_dir, 'train', eval_transform)
knn_train_dataset = torch.utils.data.Subset(knn_train_dataset, range(0, len(knn_train_dataset), knn_train_stride))
knn_train_loader = make_loader(knn_train_dataset, eval_batch_size, num_workers, shuffle=False, drop_last=False)
val_loader = make_loader(imagenet(data_dir, 'val', eval_transform), eval_batch_size, num_workers, shuffle=False, drop_last=False)

effective_batch_size = batch_size * gradient_accumulation_steps
iters_per_epoch = len(train_loader.loader) // gradient_accumulation_steps
max_iters = epochs * iters_per_epoch
warmup_iters = warmup_epochs * iters_per_epoch
warmup_teacher_temp_iters = warmup_teacher_temp_epochs * iters_per_epoch
freeze_last_layer_iters = freeze_last_layer_epochs * iters_per_epoch
peak_lr = learning_rate * effective_batch_size / 256
ncrops = 2 + local_crops_number
print(f"{len(train_dataset):,} train images, effective batch {effective_batch_size}, {iters_per_epoch:,} iters/epoch, {max_iters:,} total iters, peak lr {peak_lr:.2e}")

def get_batch():
    crops, _ = train_loader.next()
    return [c.to(device, non_blocking=True) for c in crops]

# model
model_args = dict(
    patch_size=patch_size,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    drop_path_rate=drop_path_rate,
)
head_args = dict(
    in_dim=embed_dim,
    out_dim=out_dim,
    use_bn=use_bn_in_head,
    norm_last_layer=norm_last_layer,
)

raw_student = MultiCropWrapper(net(**model_args), DINOHead(**head_args)).to(device)
# teacher has no stochastic depth and is only ever updated as an EMA of the student
raw_teacher = MultiCropWrapper(net(**{**model_args, 'drop_path_rate': 0.0}), DINOHead(**head_args)).to(device)
raw_teacher.load_state_dict(raw_student.state_dict())
raw_teacher.requires_grad_(False)
dino_loss = DINOLoss(out_dim, ncrops).to(device)
summary(raw_student)

# no weight decay on biases and norm params (same rule as DINO)
regularized, not_regularized = [], []
for name, p in raw_student.named_parameters():
    if not p.requires_grad:
        continue
    if name.endswith('.bias') or p.ndim == 1:
        not_regularized.append(p)
    else:
        regularized.append(p)
optimizer = torch.optim.AdamW(
    [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.0}],
    lr=peak_lr, betas=(beta1, beta2), fused=device_type == 'cuda',
)

# init these up here, can override if init_from='resume'
iter_num = 0
best_knn_top1 = 0.0
wandb_run_id = None

if init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    raw_student.load_state_dict(checkpoint['student'])
    raw_teacher.load_state_dict(checkpoint['teacher'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    dino_loss.load_state_dict(checkpoint['dino_loss'])
    iter_num = checkpoint['iter_num']
    best_knn_top1 = checkpoint['best_knn_top1']
    wandb_run_id = checkpoint.get('wandb_run_id')
    train_loader.epoch = iter_num // iters_per_epoch
    checkpoint = None # free up memory

student_params = list(raw_student.parameters())
teacher_params = list(raw_teacher.parameters())
last_layer_params = [raw_student.head.last_layer_v, raw_student.head.last_layer_g]

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

# compile the model
student, teacher = raw_student, raw_teacher
if compile and 'cuda' in device:
    print("compiling the model... (takes a ~minute)")
    student = torch.compile(raw_student)
    teacher = torch.compile(raw_teacher)

def cosine(it, total, start, end):
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * min(it, total) / total))

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if it < warmup_iters:
        return peak_lr * (it + 1) / (warmup_iters + 1)
    return cosine(it - warmup_iters, max_iters - warmup_iters, peak_lr, min_lr)

def get_teacher_temp(it):
    if it < warmup_teacher_temp_iters:
        return warmup_teacher_temp + (teacher_temp - warmup_teacher_temp) * it / warmup_teacher_temp_iters
    return teacher_temp

@torch.no_grad()
def evaluate_knn():
    # eval on the uncompiled teacher backbone to avoid recompiles for the eval batch shapes
    train_features, train_labels = extract_features(raw_teacher.backbone, knn_train_loader, device, ctx, desc='k-NN bank')
    val_features, val_labels = extract_features(raw_teacher.backbone, val_loader, device, ctx, desc='k-NN val')
    return knn_classifier(train_features, train_labels, val_features, val_labels, k=knn_k, T=knn_temperature)

@torch.no_grad()
def teacher_stats(teacher_output, temp):
    # collapse monitors: per-sample entropy -> 0 means one-hot collapse, batch-mean entropy -> 0 means every image maps to the same output
    p = F.softmax((teacher_output.float() - dino_loss.center) / temp, dim=-1)
    entropy = -(p * torch.log(p + 1e-12)).sum(-1).mean()
    p_mean = p.mean(0)
    mean_entropy = -(p_mean * torch.log(p_mean + 1e-12)).sum()
    return entropy.item(), mean_entropy.item()

def save_checkpoint():
    checkpoint = {
        'student': raw_student.state_dict(),
        'teacher': raw_teacher.state_dict(),
        'optimizer': optimizer.state_dict(),
        'dino_loss': dino_loss.state_dict(),
        'net': f'{net.__module__}.{net.__name__}',
        'model_args': model_args,
        'head_args': head_args,
        'iter_num': iter_num,
        'best_knn_top1': best_knn_top1,
        'config': config,
        'wandb_run_id': wandb_run_id,
    }
    # write then rename, so a preemption mid-save can't corrupt the only checkpoint
    tmp_path = os.path.join(out_dir, 'ckpt.pt.tmp')
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, os.path.join(out_dir, 'ckpt.pt'))
    print(f"saved checkpoint to {out_dir} at iter {iter_num}")

# logging
if wandb_log:
    import wandb
    if wandb_run_id is None:
        wandb_run_id = uuid.uuid4().hex[:8] # stored in the checkpoint so a resumed run continues the same wandb run
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, id=wandb_run_id, resume='allow')

# training loop
X = get_batch() # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
while True:

    # determine and set the learning rate, weight decay, and teacher temperature for this iteration
    lr = get_lr(iter_num)
    wd = cosine(iter_num, max_iters, weight_decay, weight_decay_end)
    temp = get_teacher_temp(iter_num)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[0]['weight_decay'] = wd
    optimizer.param_groups[1]['lr'] = lr

    # k-NN eval of the teacher
    if (iter_num % eval_interval == 0 or iter_num == max_iters) and (eval_only or not (init_from == 'resume' and local_iter_num == 0)):
        top1, top5 = evaluate_knn()
        best_knn_top1 = max(best_knn_top1, top1)
        print(f"iter {iter_num} (epoch {iter_num / iters_per_epoch:.1f}): k-NN top1 {top1:.2f}%, top5 {top5:.2f}%")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "epoch": iter_num / iters_per_epoch,
                "val/knn_top1": top1,
                "val/knn_top5": top5,
            }, step=iter_num)
        t0 = time.time() # don't count eval time in the next iteration's timing

    if eval_only:
        break

    if iter_num > 0 and local_iter_num > 0 and (iter_num % save_interval == 0 or iter_num == max_iters):
        save_checkpoint()

    # termination conditions
    if iter_num >= max_iters:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    batch_centers = []
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            with torch.no_grad():
                teacher_output = teacher(X[:2]) # only the 2 global views pass through the teacher
            student_output = student(X)
        loss = dino_loss(student_output, teacher_output, temp)
        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        batch_centers.append(teacher_output.float().mean(0, keepdim=True))
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X = get_batch()
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # keep the head's last layer fixed during the first epoch(s), DINO found this helps avoid early collapse
    if iter_num < freeze_last_layer_iters:
        for p in last_layer_params:
            p.grad = None
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_params, grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # EMA update of the teacher and the loss center
    momentum = cosine(iter_num, max_iters, momentum_teacher, 1.0)
    with torch.no_grad():
        torch._foreach_lerp_(teacher_params, student_params, 1.0 - momentum)
    dino_loss.update_center(torch.cat(batch_centers).mean(0, keepdim=True))

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if not math.isfinite(lossf):
            raise RuntimeError(f"loss is {lossf} at iter {iter_num}, stopping")
        entropy, mean_entropy = teacher_stats(teacher_output, temp)
        imgs_per_sec = effective_batch_size / dt
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, {imgs_per_sec:.0f} img/s, teacher entropy {entropy:.3f} / batch {mean_entropy:.3f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "epoch": iter_num / iters_per_epoch,
                "train/loss": lossf,
                "train/teacher_entropy": entropy,
                "train/teacher_batch_entropy": mean_entropy,
                "lr": lr,
                "wd": wd,
                "teacher_momentum": momentum,
                "teacher_temp": temp,
                "imgs_per_sec": imgs_per_sec,
            }, step=iter_num)
    iter_num += 1
    local_iter_num += 1

print(f"done. best k-NN top1 {best_knn_top1:.2f}%. run linear eval with:\n$ python eval_linear.py --ckpt={os.path.join(out_dir, 'ckpt.pt')} --data_dir={data_dir}")
