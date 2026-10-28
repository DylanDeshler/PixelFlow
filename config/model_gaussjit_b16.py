# Gaussian-placed tokens (model_gauss.py): JiT-B's backbone on 64 tokens (at 128px) that are oriented, scaled patches whose
# centres, sizes and rotations a small conv placer predicts from the noisy input in the same forward pass, trained only by
# the denoising loss (a deformable patch embedding). Compare with JiT-B/16 (64 tokens) at matched steps, e.g.
#   python train_jit.py config/jit_in1k_128.py config/model_gaussjit_b16.py --optimizer_type=muon --attn_res=full --adaln_single=True
# (-> runs/in1k_128/gaussjit-b16/muon_attnres-full_adalnsingle/...). See README "Gaussian-placed tokens (GaussJiT)".
model = 'GaussJiT-B/16'
