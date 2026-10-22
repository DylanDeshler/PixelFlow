# JiT-B/16 with one extra block (13 instead of 12): the parameter-matched control for the DC-AE conv additions.
# On the best in100_128 setup (full AttnRes, adaLN-single, skip + stop-grad + own final layer), one block adds +7.1M
# params and about 1/12 more compute; dcae_stem + dcae_head + dcae_trunk_head add +6.0M (default width / kernels).
# Compose with a dataset tier: python train_jit.py config/jit_in100_128.py config/model_jit_b16_d13.py ...  (-> runs/in100_128/jit-b16_d13_icl0/...)
# Probes default to blocks 4, 8 and 13 (a third, two thirds and the last), vs 4, 8 and 12 at depth 12.
model = 'JiT-B/16'
depth = 13
