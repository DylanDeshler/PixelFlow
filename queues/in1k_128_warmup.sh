#!/usr/bin/env bash
# in1k_128 follow-up (2026-11-02): learning-rate warm-up. The calibration reference run (1-epoch warm-up, tuned schedule)
# reached FID-10k 94.4 at 10 epochs vs the baseline's 100.8 with JiT's 5-epoch warm-up. Does it hold at 20 epochs, and does
# it stack with the learned per-noise-level weight (now gradient-normalized, loss_weighting.py)? 20-epoch runs matched to the
# baseline runs/in1k_128/jit-b16_icl0/muon (batch 1024). Waits for the other in1k_128 queues, so it can be started now;
# resumes like them (.done markers, interrupted training continues from its checkpoint).
#
#   nohup bash queues/in1k_128_warmup.sh > runs/queue_logs/queue_warmup.log 2>&1 &
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
BASE="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA --epochs=20"
mkdir -p runs/queue_logs
while pgrep -f "in1k_128_weighting_batch.sh|in1k_128_freq_table.sh" > /dev/null; do sleep 300; done
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
    if [ "$mode" = ">>" ]; then "$@" "${extra[@]}" >> "$log" 2>&1; else "$@" > "$log" 2>&1; fi
    local rc=$?
    [ $rc -eq 0 ] && touch "$done"
    echo "=== $(date '+%F %T') end $name (exit $rc)"
}

step warmup1         python train_jit.py $BASE --warmup_epochs=1
step warmup1_learned python train_jit.py $BASE --warmup_epochs=1 --loss_t_weight=learned
# the 768-wide bottleneck with the learned weight (JiT's 5-epoch warm-up, as the other bn768 runs), completing the bottleneck x
# weighting comparison: bn768 lost to bn128 by ~5 FID through copying; does the learned weight's down-weighting of near-clean
# levels (where bn768's loss advantage is) recover any of it?
step bn768_learned   python train_jit.py $BASE --bottleneck_dim=768 --loss_t_weight=learned
echo "=== $(date '+%F %T') warm-up queue done"
