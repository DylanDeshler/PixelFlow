# The target: JiT-B/16 on ImageNet 256 with the paper's settings (effective batch 1024, lr 2e-4).
# Paper: FID 4.37 at 200 epochs / 3.66 at 600 epochs (50k samples, with CFG, see eval_jit.py).
dataset = 'in1k_128'
batch_size = 1024
gradient_accumulation_steps = 1
epochs = 400
eval_every_epochs = 5
fid_num_images = 10000
in_context_len = 0
