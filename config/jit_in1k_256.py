# The target: JiT-B/16 on ImageNet 256 with the paper's settings (effective batch 1024, lr 2e-4).
# Paper: FID 4.37 at 200 epochs / 3.66 at 600 epochs (50k samples, with CFG, see eval_jit.py).
dataset = 'in1k_256'
batch_size = 128
gen_batch_size = batch_size * 2
gradient_accumulation_steps = 2
epochs = 400
eval_every_epochs = 5
fid_num_images = 10000
in_context_len = 0
fid_num_images = 10000
model = 'JiT-L/16'
