# ImageNet-100 at 256px with JiT-B/16: same 16x16 token regime as the target, 10x fewer images.
dataset = 'in100_256'
batch_size = 256
gradient_accumulation_steps = 1
epochs = 400
eval_every_epochs = 40
fid_num_images = 10000
