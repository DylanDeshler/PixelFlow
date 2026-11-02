#!/usr/bin/env bash
# in1k_128 (2026-11-02): two token sizes on one trunk (--token_sizes=16,8, model_jit.TokenAllocation), on the best setup so far,
# the learned per-noise-level weight (runs/in1k_128/jit-b16_icl0/muon_losst-learned, FID-10k 80.3 at 20 epochs). Each
# training sample's token size is drawn in proportion to the fine tokens' measured loss gain at its noise level (running
# per-t statistics of the two paths' losses, different images, no third forward); --fine_frac is the token budget.
# For reference at 20 epochs: B/16 81.0 (2065 img/s), B/8 65.6 (529 img/s, 3.9x the time). The question: how much of B/8's
# gain do 16/8 tokens keep at fine share 0.25 (~1.75x B/16's FLOPs per image), per image and per GPU-hour.
#   0. bench_tokens.py (minutes): ms/step, img/s, peak memory, FLOPs for B/16, B/16+8 at fine 0.1/0.25/0.35/0.5 (and 0.25 with one
#      backward over both paths), B/8, same session, and the largest fine share that fits batch 1024 without accumulation;
#      again with --act_memory_budget=0.5 (selective recompute): what more memory headroom costs in time. Each path
#      backpropagates right after its forward, so at fine 0.25 the step's peak is the fine path's 256 x 256 tokens, B/16's
#      1024 x 64: it should fit wherever B/16 does (B/8 needed batch 256 x 4 accumulation)
#   1. 16/8 tokens, fine share 0.25, 20 epochs, batch 1024 without accumulation; differs from the learned run only in
#      --token_sizes. Every eval adds alloc_eval.py: matched per-size held-out loss per t, FID-1k under fixed allocations
#      (coarse, the learned ranking at 0.25 / 0.5, fine, and fine in one t quarter at a time; same noise and labels), fixed
#      sample rows, plots in {run}/diagnostics/allocation_*.png
#   2. inference budgets (no training): the same checkpoint sampled with fine tokens on 0 / 0.1 / 0.5 / 1.0 of the calls
#      (FID-10k each; 0.25 is step 1's own eval): FID vs sampling compute from one model
#   3. only if step 1 beats the learned run at 20 epochs: the learned run continued to the same GPU-hours (constant lr after
#      warm-up, so resuming equals a longer run; the epoch count comes from step 1's measured time), the compute-matched
#      baseline. If step 1 doesn't beat it per image, it can't per hour, and this is skipped
# Shares the queues' GPU lock; resumes via .done markers and checkpoints.
#
#   nohup bash queues/in1k_128_tokens.sh > runs/queue_logs/queue_tokens.log 2>&1 &
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
BEST="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA --loss_t_weight=learned --loss_weight_normalize=False"
LEARNED=runs/in1k_128/jit-b16_icl0/muon_losst-learned
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
hours() { # training hours of a run directory's last logged step
    python -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1] + '/train_log.jsonl')]; print([x['train_hours'] for x in r if 'train_hours' in x][-1])" "$1"
}

step tokens_bench python bench_tokens.py
step tokens_bench_mem0.5 python bench_tokens.py --act_memory_budget=0.5
step tokens_f0.25 python train_jit.py $BEST --epochs=20 --token_sizes=16,8 --fine_frac=0.25

RUN=$(grep -m1 '^run:  ' runs/queue_logs/tokens_f0.25.log 2>/dev/null | sed 's/^run:  //')
if [ -n "$RUN" ] && [ -e "$RUN/ckpt.pt" ]; then
    for f in 0 0.1 0.5 1.0; do
        step "tokens_f0.25_infer$f" python eval_jit.py --ckpt=$RUN/ckpt.pt --data_dir=$DATA --final_num_images=10000 \
            --fine_frac=$f --out_dir=$RUN/eval_fine$f
    done
    f_tok=$(last_fid tokens_f0.25); f_ref=80.33
    echo "16/8 tokens: FID-10k $f_tok at 20 epochs (learned run: $f_ref)"
    if [ -n "$f_tok" ] && awk "BEGIN {exit !($f_tok < $f_ref)}"; then
        ep=$(python -c "import sys; print(max(21, round(20 * float(sys.argv[1]) / float(sys.argv[2]))))" "$(hours $RUN)" "$(hours $LEARNED)")
        echo "compute-matched baseline: the learned run to $ep epochs (its 20-epoch ckpt.pt kept as ckpt_20ep.pt)"
        [ -e "$LEARNED/ckpt_20ep.pt" ] || cp "$LEARNED/ckpt.pt" "$LEARNED/ckpt_20ep.pt"
        step "learned_${ep}ep" python train_jit.py $BEST --epochs=$ep --init_from=resume
    fi
fi
echo "=== $(date '+%F %T') tokens queue done"
