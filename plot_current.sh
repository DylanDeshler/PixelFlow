#!/usr/bin/env bash
# The in1k_128 comparisons so far as FID vs images seen / training compute / parameters (plot_scaling.py), one chart per
# question. Runs that don't exist or have no evals yet are skipped, so rerun it as runs finish.
#   bash plot_current.sh            -> runs/plots/{tail,tokens,stack}.png (+ .json; a table per chart on stdout)
set -u
R=runs/in1k_128
LR=muon_losst-learned_no-lossnorm_warmup1
BEST=$R/jit-b16_icl0/${LR}_clip0_attnres-full_adalnsingle
mkdir -p runs/plots

# 1. decoder tail vs an extra trunk block, on the best setup (no in-context tokens)
python plot_scaling.py --out=runs/plots/tail --runs="best=$BEST;\
tail 2 (4px)=$R/jit-b16_icl0/muon_tail2_tailheads8_*;\
tail 1 (4px)=$R/jit-b16_icl0/muon_tail1_tailheads8_*;\
tail 1 (8px)=$R/jit-b16_icl0/muon_tail1_tailq8*;\
depth 13=$R/jit-b16_d13_icl0/${LR}_clip0_attnres-full_adalnsingle"

# 2. extra tokens: in-context 32 and two token sizes, vs the best
python plot_scaling.py --out=runs/plots/tokens --runs="best=$BEST;\
in-context 32=$R/jit-b16/${LR}_clip0_attnres-full_adalnsingle;\
two sizes, fine 0.11=$R/jit-b16_icl0/muon_tok-16,8_fine0.11_losst-learned_no-lossnorm_warmup1_clip0_attnres-full_adalnsingle"

# 3. the stack: each adopted stage, plus Block AttnRes (the speed variant)
python plot_scaling.py --out=runs/plots/stack --runs="warm-up 1 + learned weight=$R/jit-b16_icl0/$LR;\
+ no clipping=$R/jit-b16_icl0/${LR}_clip0;\
+ AttnRes + adaLN-single=$BEST;\
Block AttnRes=$R/jit-b16_icl0/${LR}_clip0_attnres-block_adalnsingle"
