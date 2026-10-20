# Full ImageNet at 128px with JiT-B/16: checks that in100 findings hold with 1000 classes, at ~1/3 the cost of 256px.
data_dir = '/data/jit/in1k_128'
out_dir = 'jit_b16_in1k_128'
batch_size = 1024
gradient_accumulation_steps = 1
epochs = 200
eval_every_epochs = 20
fid_num_images = 10000
# noise_scale defaults to img_size/256 = 0.5 (JiT's Table 9 rule). The paper only tests patch size scaled with
# resolution (always 256 tokens), never 128px with 16px patches, so sweep --noise_scale=0.5/0.75/1.0 on the baseline first.
