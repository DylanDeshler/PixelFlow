# ImageNet-100 (every 10th class) at 128px with JiT-B/16 (8x8 tokens, still 768-d patches): the fast tier for sweeps.
# Starting point, not tuned: diffusion convergence tracks iterations more than epochs, so the smaller batch
# keeps ~200k iters over 400 epochs. Check the baseline FID curve before fixing the budget for comparisons.
dataset = 'in100_128'
batch_size = 256
gen_batch_size = batch_size * 2
gradient_accumulation_steps = 1
epochs = 400
eval_every_epochs = 10
fid_num_images = 10000
in_context_len = 0
# noise_scale defaults to img_size/256 = 0.5 (JiT's Table 9 rule). The paper only tests patch size scaled with
# resolution (always 256 tokens), never 128px with 16px patches, so sweep --noise_scale=0.5/0.75/1.0 on the baseline first.
