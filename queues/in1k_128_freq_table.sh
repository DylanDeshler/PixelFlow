#!/usr/bin/env bash
# in1k_128 follow-up (2026-10-31): frequency weighting by what a trained model recovers (the nonlinear counterpart of the
# Wiener gain, which made FID worse). Builds the gain table from the best checkpoint (the 80-epoch recipe run, EMA weights:
# recovery is a property of the data and noise, the model only estimates it, so the strongest model estimates it best), then
# a 20-epoch run matched to the baseline runs/in1k_128/jit-b16_icl0/muon (batch 1024). The queues share a GPU lock
# (runs/queue_logs/gpu.lock), so it can be started any time. Resumes like the main queue (.done markers, interrupted training
# continues from its checkpoint).
#
#   nohup bash queues/in1k_128_freq_table.sh > runs/queue_logs/queue_freq_table.log 2>&1 &
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
TABLE=runs/freq_tables/in1k_128_recipe80.json
BASE="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA --epochs=20"
mkdir -p runs/queue_logs runs/freq_tables
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

step freq_table python build_freq_table.py --ckpt=runs/in1k_128/jit-b16_icl0/muon_attnres-full_adalnsingle_skip_skipsg_skipfl \
    --data_dir=$DATA --out=$TABLE
step table_weight python train_jit.py $BASE --loss_freq_weight=table --loss_freq_table=$TABLE
echo "=== $(date '+%F %T') freq-table queue done"
