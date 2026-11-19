# ImageNet 256 with JiT-B/16 (the paper's main setting: 256 tokens, noise scale 1.0 by the img_size / 256 rule), effective
# batch 1024 as the paper and the 128px ladder. Paper: FID-50k 4.37 at 200 epochs / 3.66 at 600 (with CFG; eval_jit.py).
# (config/jit_in1k_256.py is the older JiT-L/16 setup at effective batch 256; kept as it was for its runs.)
# 256 tokens are 4x the 128px ladder's 64, so batch 1024 runs as 256 x 4 accumulation; lower batch_size / raise
# gradient_accumulation_steps together if it doesn't fit (the effective batch, and so the lr, stay the same).
dataset = 'in1k_256'
model = 'JiT-B/16'
batch_size = 256
gradient_accumulation_steps = 4
gen_batch_size = 256
epochs = 400 # constant lr after warm-up: pass --epochs for the run length (extending later resumes exactly)
eval_every_epochs = 5
fid_num_images = 10000
in_context_len = 0
