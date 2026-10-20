# The target: JiT-B/16 on ImageNet 256 with the paper's settings (effective batch 1024, lr 2e-4).
# Paper: FID 4.37 at 200 epochs / 3.66 at 600 epochs (50k samples, with CFG, see eval_jit.py).
data_dir = '/data/jit/in1k_256'
out_dir = 'jit_b16_in1k_256'
batch_size = 256
gradient_accumulation_steps = 4
epochs = 200
eval_every_epochs = 40
fid_num_images = 10000
