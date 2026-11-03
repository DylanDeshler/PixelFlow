#!/usr/bin/env bash
# in1k_128 (2026-11-03): gradient clipping on the best setup so far, the learned per-noise-level weight (EDM2's unnormalized form)
# with a 1-epoch lr warm-up (runs/in1k_128/jit-b16_icl0/muon_losst-learned_no-lossnorm_warmup1, FID-10k 109.7 / 91.9 / 82.5 /
# 76.1 at 5 / 10 / 15 / 20 epochs). The learned weight scales gradients ~5-20x, so grad_clip=1.0 fired on every step in epochs
# 1-5 and on 51% after; the baselines' gradients sit far below 1 after the first epoch (clipped 0%). So every learned-vs-baseline
# comparison so far also compares clipped against unclipped. One run, one change: --grad_clip=0.
#   about the same  the learned weight's gain is real, clipping was incidental (Muon and AdamW updates don't depend on the
#                   gradient's scale)
#   worse           clipping on every step, a per-step gradient normalization, was part of the gain: next, the baseline with
#                   a clip threshold that actually fires
#   better          clipping cost something: drop it from the best setup
# Watch the logged grad norm (pre-clip) for spikes. Shares the queues' GPU lock; resumes via .done markers and checkpoints.
#
#   nohup bash queues/in1k_128_clip.sh > runs/queue_logs/queue_clip.log 2>&1 &
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
BEST="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA --warmup_epochs=1 --loss_t_weight=learned --loss_weight_normalize=False"
mkdir -p runs/queue_logs
GPU_LOCK=runs/queue_logs/gpu.lock # shared with the other in1k_128 queues: their steps take turns on the GPU
step() { # step name command... (as in in1k_128_weighting_batch.sh)
    local name=$1; shift
    local log="runs/queue_logs/$name.log" done="runs/queue_logs/$name.done"
    if [ -e "$done" ]; then echo "skip $name (done)"; return; fi
    local extra=() mode=">"
    if [ -e "$log" ] && [ "$2" = "train_jit.py" ]; then
        local dir; dir=$(grep -m1 '^run:  ' "$log" | sed 's/^run:  //')
        if [ -n "$dir" ] && [ -e "$dir/ckpt.pt" ]; then extra=(--init_from=resume); mode=">>"; fi
    fi
    echo "=== $(date '+%F %T') start $name ${extra[*]:-}: $*"
    if [ "$mode" = ">>" ]; then flock "$GPU_LOCK" "$@" "${extra[@]}" >> "$log" 2>&1; else flock "$GPU_LOCK" "$@" > "$log" 2>&1; fi
    local rc=$?
    [ $rc -eq 0 ] && touch "$done"
    echo "=== $(date '+%F %T') end $name (exit $rc)"
}

step noclip_warmup1_learned python train_jit.py $BEST --epochs=20 --grad_clip=0
echo "=== $(date '+%F %T') clip queue done"
