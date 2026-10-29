# Synthetic Gaussian-blob images (prepare_toy_gaussians.py): 50k train / 5k val at 128px, 10 classes by blob shape, for fast
# GPU experiments on adaptive tokenization. Pair with a model config, e.g.
#   python train_jit.py config/toy_gauss_128.py config/model_gaussjit_b16.py --optimizer_type=muon
dataset = 'toy_gauss_128'
batch_size = 256
gradient_accumulation_steps = 1
epochs = 200
eval_every_epochs = 10
fid_num_images = 5000
in_context_len = 0
