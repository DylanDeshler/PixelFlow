#!/usr/bin/env bash
# in1k_128 follow-up (2026-11-02): build on the best result so far, the learned per-noise-level weight. Each run differs from
# an existing run in exactly one thing, so its effect is readable without a grid:
#   1. + 1-epoch lr warm-up       vs runs/in1k_128/jit-b16_icl0/muon_losst-learned (JiT's 5-epoch warm-up)
#   2. + 768-wide bottleneck      vs runs/in1k_128/jit-b16_bn768_icl0/muon (no learned weight)
# Both use EDM2's unnormalized form of the learned weight, as the existing learned run (--loss_weight_normalize=False). The
# plain 1-epoch warm-up run was dropped: its effect is already visible (FID 112.1 vs 129.1 at 5 epochs; 94.4 vs 100.8 at 10
# in the calibration reference). 20-epoch runs, batch 1024. The queues share a GPU lock (runs/queue_logs/gpu.lock), so this
# can be started any time; it resumes like the others (.done markers, interrupted training continues from its checkpoint).
#
#   nohup bash queues/in1k_128_warmup.sh > runs/queue_logs/queue_warmup.log 2>&1 &
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
BASE="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA --epochs=20"
mkdir -p runs/queue_logs
# one GPU: every step of every queue holds this lock while it runs, so the queues can be started in any order and take
# turns step by step (an earlier version only checked which queues were running at start, and three runs collided)
GPU_LOCK=runs/queue_logs/gpu.lock
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

step warmup1_learned python train_jit.py $BASE --warmup_epochs=1 --loss_t_weight=learned --loss_weight_normalize=False
step bn768_learned   python train_jit.py $BASE --bottleneck_dim=768 --loss_t_weight=learned --loss_weight_normalize=False
echo "=== $(date '+%F %T') warm-up queue done"
