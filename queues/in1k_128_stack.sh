#!/usr/bin/env bash
# in1k_128 (2026-11-03): stack improvements one at a time. Every stage runs on the best setup so far; if its FID-10k at 20
# epochs beats the current best, its change is adopted and the next stage builds on it. The decisions are replayed from the
# step logs on every (re)start, so the queue resumes after a shutdown with the same base at every stage.
# Start: warm-up 1 + learned per-noise-level weight (EDM2 unnormalized), runs/in1k_128/jit-b16_icl0/muon_losst-learned_no-lossnorm_warmup1,
# FID-10k 109.7 / 91.9 / 82.5 / 76.1 at 5 / 10 / 15 / 20 epochs.
#   1. clip    --grad_clip=0. The learned weight clips every step in epochs 1-5 (51% after); the baselines 0% after epoch 1.
#              Shares its step with in1k_128_clip.sh (an earlier queue), so a run already done or in progress there is reused
#   2. arch    --attn_res=full --adaln_single=True together (the in100 wins; the one agreed exception to one change per stage).
#              The in1k recipe with them trained at 1488 vs 2069 img/s: compare its img/s in the log too
#   3. tokens  --token_sizes=16,8 --fine_frac=0.25: bench_tokens.py (minutes, with and without selective recompute), the run,
#              sampling budgets on its checkpoint (FID-10k with fine tokens on 0 / 0.1 / 0.5 / 1.0 of the calls), and, if it
#              wins per image, the previous best continued to the same GPU-hours (per-hour comparison; its 20-epoch
#              checkpoint kept as ckpt_20ep.pt). Adoption is per image; the log prints the per-hour numbers next to it
#   4. disp    Dispersive Loss: analyze_dispersion.py on the current best (minutes), depth screen at blocks 3 / 6 / 9 / all
#              (lambda 0.5, tau 0.5, 10 epochs each), the screen's best resumed to 20 epochs, and EnCodec's gradient
#              balancer (share DISP_SHARE, default 0.2) at the best single block for 20 epochs; the better of the two is
#              adopted if it beats the best
# Ends by writing the final best setup to runs/queue_logs/stack_best.txt. Shares the queues' GPU lock.
#
#   nohup bash queues/in1k_128_stack.sh > runs/queue_logs/queue_stack.log 2>&1 &
#   DISP_SHARE=0.1 nohup bash queues/in1k_128_stack.sh ...      # another balancer share
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
SHARE=${DISP_SHARE:-0.2}
BASE="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA"
mkdir -p runs/queue_logs
GPU_LOCK=runs/queue_logs/gpu.lock # shared with the other in1k_128 queues: their steps take turns on the GPU
step() { # step name command...: skipped if done (also re-checked once the GPU lock is held); interrupted training resumes
    local name=$1; shift
    local log="runs/queue_logs/$name.log" done="runs/queue_logs/$name.done"
    if [ -e "$done" ]; then echo "skip $name (done)"; return; fi
    local extra=() mode=">"
    if [ -e "$log" ] && [ "$2" = "train_jit.py" ]; then
        local dir; dir=$(grep -m1 '^run:  ' "$log" | sed 's/^run:  //')
        if [ -n "$dir" ] && [ -e "$dir/ckpt.pt" ]; then extra=(--init_from=resume); mode=">>"; fi
    fi
    echo "=== $(date '+%F %T') start $name ${extra[*]+${extra[*]}}: $*"
    local guarded=(flock "$GPU_LOCK" bash -c '[ -e "$0" ] && exit 0; exec "$@"' "$done" "$@" ${extra[@]+"${extra[@]}"})
    if [ "$mode" = ">>" ]; then "${guarded[@]}" >> "$log" 2>&1; else "${guarded[@]}" > "$log" 2>&1; fi
    local rc=$?
    [ $rc -eq 0 ] && touch "$done"
    echo "=== $(date '+%F %T') end $name (exit $rc)"
}
last_fid() { grep -oE 'FID-[0-9.]+k [0-9]+(\.[0-9]+)?' "runs/queue_logs/$1.log" 2>/dev/null | tail -1 | awk '{print $2}'; }
run_of() { grep -m1 '^run:  ' "runs/queue_logs/$1.log" 2>/dev/null | sed 's/^run:  //'; }
hours() { python -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1] + '/train_log.jsonl')]; print([x['train_hours'] for x in r if 'train_hours' in x][-1])" "$1"; }
lt() { awk "BEGIN {exit !($1 < $2)}"; }

BEST_ARGS="--warmup_epochs=1 --loss_t_weight=learned --loss_weight_normalize=False"
BEST_RUN=runs/in1k_128/jit-b16_icl0/muon_losst-learned_no-lossnorm_warmup1
BEST_FID=$(last_fid warmup1_learned); BEST_FID=${BEST_FID:-76.1}
ARCH_ARGS="" # the architecture flags adopted so far (for bench_tokens.py)
adopt() { # adopt step-name "args": the step's change becomes part of the best setup if its 20-epoch FID beats the best
    local f; f=$(last_fid "$1")
    if [ -z "$f" ]; then echo ">>> $1: no FID (failed or unfinished); best stays $BEST_FID"; return 1; fi
    if lt "$f" "$BEST_FID"; then
        echo ">>> $1: FID-10k $f < $BEST_FID: adopted ($2)"
        BEST_ARGS="$BEST_ARGS $2"; BEST_RUN=$(run_of "$1"); BEST_FID=$f
    else
        echo ">>> $1: FID-10k $f >= $BEST_FID: not adopted"; return 1
    fi
}
echo "=== $(date '+%F %T') stack: start from FID-10k $BEST_FID ($BEST_ARGS)"

# 1. clip
step noclip_warmup1_learned python train_jit.py $BASE $BEST_ARGS --epochs=20 --grad_clip=0
adopt noclip_warmup1_learned "--grad_clip=0"

# 2. arch
step stack_arch python train_jit.py $BASE $BEST_ARGS --epochs=20 --attn_res=full --adaln_single=True
adopt stack_arch "--attn_res=full --adaln_single=True" && ARCH_ARGS="--attn_res=full --adaln_single=True"

# 3. tokens
step stack_tokens_bench python bench_tokens.py $ARCH_ARGS
step stack_tokens_bench_mem0.5 python bench_tokens.py $ARCH_ARGS --act_memory_budget=0.5
step stack_tokens python train_jit.py $BASE $BEST_ARGS --epochs=20 --token_sizes=16,8 --fine_frac=0.25
TOK_RUN=$(run_of stack_tokens)
if [ -n "$TOK_RUN" ] && [ -e "$TOK_RUN/ckpt.pt" ]; then
    for f in 0 0.1 0.5 1.0; do
        step "stack_tokens_infer$f" python eval_jit.py --ckpt=$TOK_RUN/ckpt.pt --data_dir=$DATA --final_num_images=10000 --plot=False \
            --fine_frac=$f --out_dir=$TOK_RUN/eval_fine$f
    done
fi
PREV_RUN=$BEST_RUN PREV_ARGS=$BEST_ARGS
if adopt stack_tokens "--token_sizes=16,8 --fine_frac=0.25"; then
    # per GPU-hour: the previous best continued to the 16/8 run's training time (constant lr after warm-up: resuming = a longer run)
    ep=$(python -c "import sys; print(max(21, round(20 * float(sys.argv[1]) / float(sys.argv[2]))))" "$(hours "$TOK_RUN")" "$(hours "$PREV_RUN")")
    [ -e "$PREV_RUN/ckpt_20ep.pt" ] || cp "$PREV_RUN/ckpt.pt" "$PREV_RUN/ckpt_20ep.pt"
    step "stack_tokens_matched_${ep}ep" python train_jit.py $BASE $PREV_ARGS --epochs=$ep --init_from=resume
    echo ">>> per GPU-hour: 16/8 tokens FID-10k $BEST_FID at 20 epochs vs the previous best continued to $ep epochs (same hours): $(last_fid "stack_tokens_matched_${ep}ep")"
fi

# 4. disp
step stack_disp_analysis python analyze_dispersion.py --ckpts=$BEST_RUN --data_dir=$DATA
for b in 3 6 9 all; do
    step "stack_disp_b${b}_10ep" python train_jit.py $BASE $BEST_ARGS --epochs=10 --disp_weight=0.5 --disp_blocks=$b
done
best=""; best_fid=""; best_single=""; best_single_fid=""
for b in 3 6 9 all; do
    f=$(last_fid "stack_disp_b${b}_10ep"); [ -z "$f" ] && continue
    echo "screen: block(s) $b -> FID-10k $f at 10 epochs"
    if [ -z "$best_fid" ] || lt "$f" "$best_fid"; then best=$b; best_fid=$f; fi
    if [ "$b" != all ] && { [ -z "$best_single_fid" ] || lt "$f" "$best_single_fid"; }; then best_single=$b; best_single_fid=$f; fi
done
if [ -n "$best" ]; then
    echo "screen best: block(s) $best (FID-10k $best_fid); best single block: $best_single"
    step "stack_disp_b${best}_20ep" python train_jit.py $BASE $BEST_ARGS --epochs=20 --disp_weight=0.5 --disp_blocks=$best --init_from=resume
    step "stack_disp_balance${SHARE}_b${best_single}" python train_jit.py $BASE $BEST_ARGS --epochs=20 --disp_balance=$SHARE --disp_blocks=$best_single
    fw=$(last_fid "stack_disp_b${best}_20ep"); fb=$(last_fid "stack_disp_balance${SHARE}_b${best_single}")
    if [ -n "$fb" ] && { [ -z "$fw" ] || lt "$fb" "$fw"; }; then
        adopt "stack_disp_balance${SHARE}_b${best_single}" "--disp_balance=$SHARE --disp_blocks=$best_single"
    else
        adopt "stack_disp_b${best}_20ep" "--disp_weight=0.5 --disp_blocks=$best"
    fi
fi

printf 'FID-10k %s\nrun %s\nargs %s %s\n' "$BEST_FID" "$BEST_RUN" "$BASE" "$BEST_ARGS" > runs/queue_logs/stack_best.txt
echo "=== $(date '+%F %T') stack done: best FID-10k $BEST_FID, $BEST_RUN"
echo "    python train_jit.py $BASE $BEST_ARGS"
