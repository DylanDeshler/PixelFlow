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
#   3. disp    Dispersive Loss on mean-pooled, unit-RMS features (--disp_kind=infonce_l2n --disp_pool=mean). Earlier attempts:
#              stack_disp_* (the paper's form pinned at its floor by growing the features, ~13 FID worse) and stack_disp2_*
#              (normalized but unpooled: saturated in ~600 steps by carrying input noise). analyze_dispersion.py on the
#              current best (minutes), then best shot first, 10 epochs
#              each: DISP_ARMS = block:weight pairs, default block 8 at weight 0.5 then 0.1, stopping at the first that beats
#              the current best's own 10-epoch FID. That arm resumes to 20 epochs, then EnCodec's gradient balancer (share
#              DISP_SHARE, default 0.2) at the same block for 20 epochs; the better of the two is adopted if it beats the best. A run stopped by train_jit.py's
#              dispersive guard (the loss at its best possible value: the noise shortcut) counts as a finished failure
#   4. block   --attn_res=block (Block AttnRes: 1.14x plain step time vs full's 1.25x), if full AttnRes was adopted; adopted if
#              FID-10k is within ATTNRES_BLOCK_TOL (default 0.5) of the best: a speed change
#   (hyperparameter / noise tuning stages slot in here when added; patch allocation stays last: it slows every step)
#   5. budget  extra tokens at equal compute (x1.34 FLOPs): JiT's 32 in-context tokens from block 4 vs two token sizes at fine
#              share 0.11; the better is adopted if it beats the best
#   6. tokens  --token_sizes=16,8 --fine_frac=0.25: bench_tokens.py (minutes, with and without selective recompute), the run,
#              sampling budgets on its checkpoint (FID-10k with fine tokens on 0 / 0.1 / 0.5 / 1.0 of the calls), and, if it
#              wins per image, the previous best continued to the same GPU-hours (per-hour comparison; its 20-epoch
#              checkpoint kept as ckpt_20ep.pt). Adoption is per image; the log prints the per-hour numbers next to it
# Ends by writing the final best setup to runs/queue_logs/stack_best.txt. Shares the queues' GPU lock.
#
#   nohup bash queues/in1k_128_stack.sh > runs/queue_logs/queue_stack.log 2>&1 &
#   DISP_SHARE=0.1 nohup bash queues/in1k_128_stack.sh ...      # another balancer share
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
SHARE=${DISP_SHARE:-0.2}
DISP_KIND=infonce_l2n # scale-invariant (repa.dispersive_loss): plain infonce_l2 pinned at its floor by growing the features (2026-11-03)
DISP_POOL=mean # one vector per image: the unpooled normalized form saturated in ~600 steps through the input-noise shortcut (2026-11-03)
DISP_ARGS="--disp_kind=$DISP_KIND --disp_pool=$DISP_POOL"
ATTNRES_BLOCK_TOL=${ATTNRES_BLOCK_TOL:-0.5} # stage 5: FID-10k Block AttnRes may lose and still be adopted (about run-to-run noise)
DISP_ARMS=${DISP_ARMS:-"8:0.5 8:0.1 8:0.1:32:l2s all:0.1:32:l2s 8:0.05 8:0.01"} # block:weight[:group[:form]], in order (form: l2n, the
    # default, or l2s). Block 8: where probes peak (kNN 3.42 vs 2.24 at block 4, 1.06 at block 12). Weight 0.5 (the paper's) failed
    # (FID 90.87 @10, linear probe 5.68 -> 1.24), then 0.1. Then the paper's setting made scale-safe: batch-standardized InfoNCE-L2
    # (scale- and translation-invariant) within groups of 32 (the paper's per-GPU batch), weight 0.1 (FID 81.57 @10: re-routed,
    # the dispersed mix leaned on block 6's attention output, 0.50, a dead end the other mixes ignore); then the same on all
    # blocks (averaged: every mix must disperse, so the loss can't be routed around; the paper's best placement); then weaker
    # weights 0.05, 0.01 of the per-sample-normalized form on the whole batch
BASE="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA"
IMPL="--attn_res_chunk=8" # implementation only (same results, not in run names): full AttnRes mixed in chunks of 8, 1.38x -> 1.25x plain
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
    if [ $rc -eq 0 ] || [ $rc -eq 3 ]; then touch "$done"; fi # 3: train_jit.py's dispersive guard stopped it (a finished negative result)
    echo "=== $(date '+%F %T') end $name (exit $rc)"
}
last_fid() { grep -oE 'FID-[0-9.]+k [0-9]+(\.[0-9]+)?' "runs/queue_logs/$1.log" 2>/dev/null | tail -1 | awk '{print $2}'; }
run_of() { grep -m1 '^run:  ' "runs/queue_logs/$1.log" 2>/dev/null | sed 's/^run:  //'; }
fid_at_epoch() { # a run's in-loop FID at the eval nearest an epoch (from its diagnostics)
    python -c "
import json, glob, sys, os
run, ep = sys.argv[1], float(sys.argv[2])
c = json.load(open(os.path.join(run, 'config.json'))); ipe = c.get('resolved', {}).get('iters_per_epoch') or c.get('derived', {}).get('iters_per_epoch')
d = [(abs(int(f.rsplit('_', 1)[1][:-5]) / ipe - ep), f) for f in glob.glob(os.path.join(run, 'diagnostics', 'diag_*.json'))]
f = min(d)[1] if d else None
print(json.load(open(f))['scalars'].get('val/fid', '') if f and min(d)[0] < 0.5 else '')
" "$1" "$2" 2>/dev/null
}
ips() { # a run's median img/s (from its train log)
    [ -n "$1" ] && python -c "import json,sys; v=sorted(json.loads(l)['imgs_per_sec'] for l in open(sys.argv[1] + '/train_log.jsonl') if 'imgs_per_sec' in l and json.loads(l).get('epoch', 0) > 0.5); print(round(v[len(v) // 2]) if v else '?')" "$1" 2>/dev/null || echo '?'
}
hours() { python -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1] + '/train_log.jsonl')]; print([x['train_hours'] for x in r if 'train_hours' in x][-1])" "$1"; }
lt() { awk "BEGIN {exit !($1 < $2)}"; }

BEST_ARGS="--warmup_epochs=1 --loss_t_weight=learned --loss_weight_normalize=False"
BEST_RUN=runs/in1k_128/jit-b16_icl0/muon_losst-learned_no-lossnorm_warmup1
BEST_FID=$(last_fid warmup1_learned); BEST_FID=${BEST_FID:-76.1}
ARCH_ARGS="" # the architecture flags adopted so far (for bench_tokens.py)
BENCH_TAG="" # _block once Block AttnRes is adopted: the token benchmark then measures that architecture
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
step noclip_warmup1_learned python train_jit.py $BASE $IMPL $BEST_ARGS --epochs=20 --grad_clip=0
adopt noclip_warmup1_learned "--grad_clip=0"

# 2. arch
step stack_arch python train_jit.py $BASE $IMPL $BEST_ARGS --epochs=20 --attn_res=full --adaln_single=True
adopt stack_arch "--attn_res=full --adaln_single=True" && ARCH_ARGS="--attn_res=full --adaln_single=True"

# 3. disp (pooled, scale-invariant; best shot first: DISP_ARMS in order, the first arm that beats the current best's own
#    10-epoch FID continues to 20 epochs)
step stack_disp3_analysis python analyze_dispersion.py --ckpts=$BEST_RUN --data_dir=$DATA --kind=$DISP_KIND --pool=$DISP_POOL
REF10=$(fid_at_epoch "$BEST_RUN" 10)
echo "dispersive stage: reference FID-10k at 10 epochs ${REF10:-unknown} ($BEST_RUN)"
win=""; wlam=""
for arm in $DISP_ARMS; do
    IFS=: read -r b lam grp form <<< "$arm"
    gtag=${grp:+_g$grp}; gargs=${grp:+--disp_group=$grp}
    if [ -n "$form" ] && [ "$form" != l2n ]; then gtag="${gtag}_$form"; gargs="$gargs --disp_kind=infonce_$form"; fi
    step "stack_disp3_b${b}_w${lam}${gtag}_10ep" python train_jit.py $BASE $IMPL $BEST_ARGS --epochs=10 $DISP_ARGS $gargs --disp_weight=$lam --disp_blocks=$b
    f=$(last_fid "stack_disp3_b${b}_w${lam}${gtag}_10ep")
    echo "screen: block $b, weight $lam${grp:+, groups of $grp}${form:+, form $form} -> FID-10k ${f:-none (stopped or failed)} at 10 epochs (reference ${REF10:-unknown})"
    if [ -n "$f" ] && [ -n "$REF10" ] && lt "$f" "$REF10"; then win=$b; wlam=$lam; wtag=$gtag; wargs=$gargs; break; fi
done
if [ -n "$win" ]; then
    step "stack_disp3_b${win}_w${wlam}${wtag}_20ep" python train_jit.py $BASE $IMPL $BEST_ARGS --epochs=20 $DISP_ARGS $wargs --disp_weight=$wlam --disp_blocks=$win --init_from=resume
    [ "$win" != all ] && step "stack_disp3_balance${SHARE}_b${win}${wtag}" python train_jit.py $BASE $IMPL $BEST_ARGS --epochs=20 $DISP_ARGS $wargs --disp_balance=$SHARE --disp_blocks=$win # (the balancer needs a single block)
    fw=$(last_fid "stack_disp3_b${win}_w${wlam}${wtag}_20ep"); fb=$(last_fid "stack_disp3_balance${SHARE}_b${win}${wtag}")
    if [ -n "$fb" ] && { [ -z "$fw" ] || lt "$fb" "$fw"; }; then
        adopt "stack_disp3_balance${SHARE}_b${win}${wtag}" "$DISP_ARGS $wargs --disp_balance=$SHARE --disp_blocks=$win"
    else
        adopt "stack_disp3_b${win}_w${wlam}${wtag}_20ep" "$DISP_ARGS $wargs --disp_weight=$wlam --disp_blocks=$win"
    fi
else
    echo ">>> dispersive: no arm beat the reference at 10 epochs; not adopted"
fi

# 4. Block AttnRes, for speed (only if full AttnRes was adopted): sources are block sums, 1.14x plain step time vs full's 1.25x
#    (bench_attnres.py). Adopted if its FID is within ATTNRES_BLOCK_TOL of the best (a speed change: equal quality is enough)
if [[ "$BEST_ARGS" == *"--attn_res=full"* ]]; then
    step stack_attnres_block python train_jit.py $BASE $IMPL $BEST_ARGS --epochs=20 --attn_res=block
    fbk=$(last_fid stack_attnres_block)
    echo "block AttnRes: FID-10k ${fbk:-none} vs best $BEST_FID; img/s $(ips "$(run_of stack_attnres_block)") vs $(ips "$BEST_RUN")"
    if [ -n "$fbk" ] && awk "BEGIN {exit !($fbk <= $BEST_FID + $ATTNRES_BLOCK_TOL)}"; then
        echo ">>> stack_attnres_block: FID-10k $fbk within $ATTNRES_BLOCK_TOL of $BEST_FID: adopted for speed (--attn_res=block)"
        BEST_ARGS="$BEST_ARGS --attn_res=block"; BEST_RUN=$(run_of stack_attnres_block); BEST_FID=$fbk
        ARCH_ARGS="--attn_res=block --adaln_single=True"; BENCH_TAG=_block
    else
        echo ">>> stack_attnres_block: not adopted (more than $ATTNRES_BLOCK_TOL worse, or failed)"
    fi
fi

# (hyperparameter and noise tuning go here when added: weight decay, Muon momentum / beta2, lr, noise schedule; patch
#  allocation stays last because it slows every step, unless the tuning is better done on the final architecture)

# 5. extra-token budget at equal compute: registers or finer detail? JiT's 32 in-context class tokens from block 4 (85.3 tokens
#    per block on average, x1.34 FLOPs at 128px; the paper's count, though at 256px it was only +8% tokens) against two token
#    sizes at the FLOP-matched fine share 0.11. The better is adopted if it beats the best (the per-image comparison between
#    the two is the result; both logs print img/s)
step stack_icl32 python train_jit.py $BASE $IMPL $BEST_ARGS --epochs=20 --in_context_len=32
step stack_fine0.11 python train_jit.py $BASE $IMPL $BEST_ARGS --epochs=20 --token_sizes=16,8 --fine_frac=0.11
fi=$(last_fid stack_icl32); ff=$(last_fid stack_fine0.11)
echo "extra tokens at x1.34 FLOPs: in-context 32 -> FID-10k ${fi:-none} ($(ips "$(run_of stack_icl32)") img/s), fine 0.11 -> ${ff:-none} ($(ips "$(run_of stack_fine0.11)") img/s); best $BEST_FID"
ICL=""
if [ -n "$fi" ] && { [ -z "$ff" ] || lt "$fi" "$ff"; }; then
    adopt stack_icl32 "--in_context_len=32" && ICL=1
elif [ -n "$ff" ]; then
    adopt stack_fine0.11 "--token_sizes=16,8 --fine_frac=0.11"
fi

# 6. tokens (last: two forwards per step make it the slowest stage); skipped if in-context tokens were adopted (the two-size
#    model doesn't support them)
if [ -n "$ICL" ]; then echo ">>> tokens: skipped (in-context tokens adopted; not supported with two token sizes)"; else
step stack_tokens_bench$BENCH_TAG python bench_tokens.py $ARCH_ARGS $IMPL
step stack_tokens_bench${BENCH_TAG}_mem0.5 python bench_tokens.py $ARCH_ARGS $IMPL --act_memory_budget=0.5
step stack_tokens python train_jit.py $BASE $IMPL $BEST_ARGS --epochs=20 --token_sizes=16,8 --fine_frac=0.25
TOK_RUN=$(run_of stack_tokens)
if [ -n "$TOK_RUN" ] && [ -e "$TOK_RUN/ckpt.pt" ]; then
    for f in 0 0.1 0.5 1.0; do
        step "stack_tokens_infer$f" python eval_jit.py $IMPL --ckpt=$TOK_RUN/ckpt.pt --data_dir=$DATA --final_num_images=10000 --plot=False \
            --fine_frac=$f --out_dir=$TOK_RUN/eval_fine$f
    done
fi
PREV_RUN=$BEST_RUN PREV_ARGS=$BEST_ARGS
if adopt stack_tokens "--token_sizes=16,8 --fine_frac=0.25"; then
    # per GPU-hour: the previous best continued to the 16/8 run's training time (constant lr after warm-up: resuming = a longer run)
    ep=$(python -c "import sys; print(max(21, round(20 * float(sys.argv[1]) / float(sys.argv[2]))))" "$(hours "$TOK_RUN")" "$(hours "$PREV_RUN")")
    [ -e "$PREV_RUN/ckpt_20ep.pt" ] || cp "$PREV_RUN/ckpt.pt" "$PREV_RUN/ckpt_20ep.pt"
    step "stack_tokens_matched_${ep}ep" python train_jit.py $BASE $IMPL $PREV_ARGS --epochs=$ep --init_from=resume
    echo ">>> per GPU-hour: 16/8 tokens FID-10k $BEST_FID at 20 epochs vs the previous best continued to $ep epochs (same hours): $(last_fid "stack_tokens_matched_${ep}ep")"
fi

fi

printf 'FID-10k %s\nrun %s\nargs %s %s\n' "$BEST_FID" "$BEST_RUN" "$BASE" "$BEST_ARGS" > runs/queue_logs/stack_best.txt
echo "=== $(date '+%F %T') stack done: best FID-10k $BEST_FID, $BEST_RUN"
echo "    python train_jit.py $BASE $IMPL $BEST_ARGS"
