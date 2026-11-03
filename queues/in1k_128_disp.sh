#!/usr/bin/env bash
# in1k_128 (2026-11-02): Dispersive Loss (Wang & He 2025) on the best setup so far: the learned per-noise-level weight (EDM2's
# unnormalized form) with a 1-epoch lr warm-up, runs/in1k_128/jit-b16_icl0/muon_losst-learned_no-lossnorm_warmup1
# (FID-10k 109.7 / 91.9 / 82.5 / 76.1 at 5 / 10 / 15 / 20 epochs). Depth by successive halving, then the weighting:
#   0. analyze_dispersion.py on the best run (minutes): per block, how dispersed the features already are and what share
#      of the gradient lambda = 0.5 amounts to -- read it before the screen finishes, it may justify pruning arms
#   1-4. screen: blocks 3 (the paper's depth // 4), 6, 9 (JiT's built-in encoder / decoder may push the best depth later),
#      all (averaged; best in the paper's ablation), lambda 0.5, tau 0.5, 10 epochs each
#   5. the screen's best continues to 20 epochs (constant lr after warm-up, so resuming 10 -> 20 equals a 20-epoch run)
#   6. EnCodec's gradient balancer (balancer.py, from Jazz) instead of the fixed weight, at the best single block, 20 epochs:
#      the dispersive loss gets a fixed share DISP_SHARE of the gradient into the blocks below (default 0.2; step 0's
#      share column says what lambda 0.5 amounts to)
# Every run differs from an existing one in one thing. Watch the logged disp: at batch 1024 it lives in [-6.93, 0]; pinned at
# -6.93 means no gradient (raise --disp_temp). Shares the queues' GPU lock; resumes via .done markers and checkpoints.
#
#   nohup bash queues/in1k_128_disp.sh > runs/queue_logs/queue_disp.log 2>&1 &
#   DISP_SHARE=0.1 nohup bash queues/in1k_128_disp.sh ...      # another balancer share
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
SHARE=${DISP_SHARE:-0.2}
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
last_fid() { grep -oE 'FID-[0-9.]+k [0-9]+(\.[0-9]+)?' "runs/queue_logs/$1.log" 2>/dev/null | tail -1 | awk '{print $2}'; }

step disp_analysis python analyze_dispersion.py --ckpts=runs/in1k_128/jit-b16_icl0/muon_losst-learned_no-lossnorm_warmup1 --data_dir=$DATA
for b in 3 6 9 all; do
    step "disp_b${b}_10ep" python train_jit.py $BEST --epochs=10 --disp_weight=0.5 --disp_blocks=$b
done

# the screen's best (lowest FID-10k at 10 epochs); the balancer uses the best single block
best=""; best_fid=""; best_single=""; best_single_fid=""
for b in 3 6 9 all; do
    f=$(last_fid "disp_b${b}_10ep"); [ -z "$f" ] && continue
    echo "screen: block(s) $b -> FID-10k $f at 10 epochs"
    if [ -z "$best_fid" ] || awk "BEGIN {exit !($f < $best_fid)}"; then best=$b; best_fid=$f; fi
    if [ "$b" != all ] && { [ -z "$best_single_fid" ] || awk "BEGIN {exit !($f < $best_single_fid)}"; }; then best_single=$b; best_single_fid=$f; fi
done
if [ -n "$best" ]; then
    echo "best: block(s) $best (FID-10k $best_fid); best single block: $best_single"
    step "disp_b${best}_20ep" python train_jit.py $BEST --epochs=20 --disp_weight=0.5 --disp_blocks=$best --init_from=resume
    step "disp_balance${SHARE}_b${best_single}" python train_jit.py $BEST --epochs=20 --disp_balance=$SHARE --disp_blocks=$best_single
fi
echo "=== $(date '+%F %T') dispersive queue done"
