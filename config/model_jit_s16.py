# JiT-S/16: a ~4x cheaper JiT-B/16 (half width, same depth and patch size) for fast ablations.
# Compose with a dataset tier: python train_jit.py config/jit_in100_128.py config/model_jit_s16.py  (-> runs/in100_128/jit-b16_w384_h6/base)
model = 'JiT-B/16'
hidden_size = 384
num_heads = 6
