#!/usr/bin/env bash
# Plan step 2 (2026-11-26): standard JiT conditioning vs the KV prefix, before the main study commits to one.
#   A  incontext  JiT's 32 in-context class tokens (prepended at block 4)            --in_context_len=32
#   B  kvtable    the KV prefix as a learned 32-token x 1,000-class table (+ null row)  --kv_prefix_len=32 --kv_prefix_class_tokens=True
# Both: plain JiT + Muon (match_rms_adamw), batch 1024, 500-update warm-up then constant lr, clean-image prediction, identical
# data / augmentation; post-hoc EMA tracked (sigma_rel 0.05, 0.10). Every evaluation: guided FID with a fixed budget of 6
# guidance scales (eval_jit --cfg_search=6, CFG at all t), post-hoc EMA sigma_rel 0.10 rebuilt at the checkpoint's step.
# Equal search allowance per arm, at W192 (12 x 192, 3 heads of 64) on full data:
#   1. screen  Muon lr (blr 1e-4 / 2e-4 / 4e-4) x pixel noise scale (0.5 / 1.0): 6 runs of 2,000 updates (2.048M images),
#              guided FID-2K
#   2. aux     the AdamW groups' lr (embeddings incl. the prefix table, readout, norms, biases) x 0.5 and x 2 at the arm's best
#              screen setting: 2 runs, FID-2K
#   3. promote the arm's best two settings resumed to 8,000 updates (8.192M images): searched at 2K, the picks re-scored at FID-8K
#              with fresh samples
#   4. FLOPs   the cheaper arm's best resumed to the other arm's measured training compute, re-scored the same way
#   5. confirm each arm's best setting at S (12 x 384, 6 heads) on full data and at W192 on N/8, 8,000 updates, FID-8K
# Decisions (the lowest FID-2K / FID-8K) are printed with ">>>" and written to runs/conditioning/*.txt (delete one to redo it).
# Read-out: guided FID at matched images (step 3) and at matched FLOPs (step 4), the S and N/8 checks (step 5), and the
# sampling times printed by each evaluation.
#
#   tmux new -s cond
#   bash queues/conditioning.sh 2>&1 | tee -a runs/queue_logs/queue_conditioning.log
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
mkdir -p runs/queue_logs runs/conditioning
GPU_LOCK=runs/queue_logs/gpu.lock
BASE="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA --warmup_updates=500 --phema_stds=0.05,0.10 \
--fid_num_images=2000 --eval_every_epochs=1000"
W192="--hidden_size=192 --num_heads=3"
S="--hidden_size=384 --num_heads=6"
arm_flags() { case $1 in incontext) echo "--in_context_len=32" ;; kvtable) echo "--kv_prefix_len=32 --kv_prefix_class_tokens=True" ;; esac; }
SEARCH="--phema_std=0.10 --cfg_search=6"

step() { # step name command...: skipped if done; output to its log and the screen; interrupted training resumes
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
    local tee_args=(); [ "$mode" = ">>" ] && tee_args=(-a)
    "${guarded[@]}" 2>&1 | tee ${tee_args[@]+"${tee_args[@]}"} "$log"
    local rc=${PIPESTATUS[0]}
    [ $rc -eq 0 ] && touch "$done"
    echo "=== $(date '+%F %T') end $name (exit $rc)"
}
run_of() { grep -m1 '^run:  ' "runs/queue_logs/$1.log" 2>/dev/null | sed 's/^run:  //'; }
done_ok() { [ -e "runs/queue_logs/$1.done" ]; }
flags_of() { # step name -> its training flags (cond_<arm>_blr<x>_noise<y>[_aux<z>])
    local n=$1 aux=1.0
    case $n in *_aux*) aux=${n##*_aux}; n=${n%_aux*} ;; esac
    echo "--blr=$(echo "$n" | sed 's/.*_blr\([^_]*\)_noise.*/\1/') --noise_scale=${n##*_noise} --aux_lr_mult=$aux"
}
best_fid() { # an evaluation directory's best FID (the re-scored one when there is one)
    python -c "
import json, glob, sys
fs = glob.glob(sys.argv[1] + '/eval_*.json')
if not fs: sys.exit()
r = json.load(open(fs[0]))
f = r.get('final', {})
v = [f[k]['fid'] for k in f if isinstance(f[k], dict) and 'fid' in f[k]] or list(r['fid'].values())
print(min(v))" "$1"
}
evaluate() { # step-name run-dir n-search n-final [eval-dir-name]: guided search of the run's last checkpoint
    local d=$2 out="$2/${5:-eval_search}"
    [ -n "$d" ] && step "$1" python eval_jit.py --ckpt="$d/ckpt.pt" --data_dir=$DATA $SEARCH --sweep_num_images=$3 \
        --final_num_images=$4 --out_dir="$out"
}
pick() { # file what step-name...: the step whose evaluation has the lowest FID; prints and stores "step fid"
    local file=runs/conditioning/$1.txt what=$2; shift 2
    if [ -e "$file" ]; then cat "$file"; return; fi
    local best="" bf="" s f
    for s in "$@"; do
        done_ok "${s}_eval" || { echo ">>> $what: $s not evaluated yet; not decided" >&2; return 1; }
        f=$(best_fid "$(run_of "$s")/eval_search")
        echo "    $s: FID $f" >&2
        if [ -z "$bf" ] || python -c "import sys; sys.exit(0 if float('$f') < float('$bf') else 1)"; then best=$s; bf=$f; fi
    done
    echo "$best $bf" | tee "$file"
    echo ">>> $what: $best (FID $bf)" >&2
}

for arm in incontext kvtable; do
    A=$(arm_flags $arm)
    # 1. screen
    names=()
    for blr in 1e-4 2e-4 4e-4; do for ns in 0.5 1.0; do
        n="cond_${arm}_blr${blr}_noise${ns}"; names+=("$n")
        step "$n" python train_jit.py $BASE $W192 $A --blr=$blr --noise_scale=$ns --max_updates=2000
        done_ok "$n" && evaluate "${n}_eval" "$(run_of "$n")" 2000 2000
    done; done
    best1=$(pick "${arm}_screen" "$arm screen (W192, 2,000 updates, FID-2K)" "${names[@]}") || continue
    b=${best1%% *}; blr=$(echo "$b" | sed 's/.*_blr\([^_]*\)_noise.*/\1/'); ns=${b##*_noise}; ns=${ns%%_aux*}
    # 2. aux-group lr at the best screen setting
    for m in 0.5 2; do
        n="cond_${arm}_blr${blr}_noise${ns}_aux${m}"; names+=("$n")
        step "$n" python train_jit.py $BASE $W192 $A --blr=$blr --noise_scale=$ns --aux_lr_mult=$m --max_updates=2000
        done_ok "$n" && evaluate "${n}_eval" "$(run_of "$n")" 2000 2000
    done
    pick "${arm}_tuned" "$arm best of screen + aux (FID-2K)" "$b" "cond_${arm}_blr${blr}_noise${ns}_aux0.5" "cond_${arm}_blr${blr}_noise${ns}_aux2" >/dev/null || continue
    # 3. promote the best two (by FID-2K over all 8 runs) to 8,000 updates
    top2=$(for s in "${names[@]}"; do echo "$(best_fid "$(run_of "$s")/eval_search") $s"; done | sort -g | head -2 | awk '{print $2}')
    for s in $top2; do
        d=$(run_of "$s")
        flags=$(flags_of "$s")
        step "${s}_8k" python train_jit.py $BASE $W192 $A $flags --max_updates=8000 --out_dir="$d" --init_from=resume
        done_ok "${s}_8k" && evaluate "${s}_8k_eval" "$d" 2000 8000 eval_8k
    done
done

# 4. matched FLOPs: each arm's best promoted run (lowest FID-8K); the cheaper arm resumed to the other's training compute
best8() { # arm -> "step fid" of its best promoted run
    for s in $(ls runs/queue_logs/cond_"$1"_*_8k_eval.done 2>/dev/null | sed 's|runs/queue_logs/||; s|_8k_eval.done||'); do
        echo "$(best_fid "$(run_of "$s")/eval_8k") $s"
    done | sort -g | head -1 | awk '{print $2, $1}'
}
read -r bi fi_ <<< "$(best8 incontext)"; read -r bk fk <<< "$(best8 kvtable)"
if [ -n "${bi:-}" ] && [ -n "${bk:-}" ]; then
    echo ">>> matched images (8,000 updates, FID-8K): incontext $bi $fi_ | kvtable $bk $fk"
    pf() { python -c "import json,sys; r=json.load(open(sys.argv[1]+'/config.json'))['resolved']; print(r['fwd_gflops'] * 3 * r['effective_batch_size'] * 8000 / 1e6)" "$1"; }
    pi=$(pf "$(run_of "$bi")"); pk=$(pf "$(run_of "$bk")")
    if python -c "import sys; sys.exit(0 if float('$pk') < float('$pi') else 1)"; then cheap=$bk; target=$pi; else cheap=$bi; target=$pk; fi
    d=$(run_of "$cheap")
    flags=$(flags_of "$cheap")
    carm=$(echo "$cheap" | sed 's/^cond_\([a-z]*\)_.*/\1/')
    echo ">>> training compute at 8,000 updates: incontext $pi PF, kvtable $pk PF; $cheap extended to $target PF"
    step "${cheap}_flops" python train_jit.py $BASE $W192 $(arm_flags $carm) $flags --budget_pflops=$target --out_dir="$d" --init_from=resume
    done_ok "${cheap}_flops" && evaluate "${cheap}_flops_eval" "$d" 2000 8000 eval_flops
    # 5. confirm at S (full data) and at W192 on N/8, each arm's best setting, 8,000 updates
    for pair in "incontext $bi" "kvtable $bk"; do
        set -- $pair; arm=$1; s=$2
        flags=$(flags_of "$s")
        step "cond_${arm}_S" python train_jit.py $BASE $S $(arm_flags $arm) $flags --max_updates=8000
        done_ok "cond_${arm}_S" && evaluate "cond_${arm}_S_eval" "$(run_of "cond_${arm}_S")" 2000 8000 eval_8k
        step "cond_${arm}_W192_frac0.125" python train_jit.py $BASE $W192 $(arm_flags $arm) $flags --train_frac=0.125 --max_updates=8000
        done_ok "cond_${arm}_W192_frac0.125" && evaluate "cond_${arm}_W192_frac0.125_eval" "$(run_of "cond_${arm}_W192_frac0.125")" 2000 8000 eval_8k
    done
    for s in cond_incontext_S cond_kvtable_S cond_incontext_W192_frac0.125 cond_kvtable_W192_frac0.125; do
        echo ">>> $s: FID-8K $(best_fid "$(run_of "$s")/eval_8k")"
    done
    echo ">>> matched FLOPs: $cheap at $target PF: FID-8K $(best_fid "$d/eval_flops")"
fi
echo "=== $(date '+%F %T') conditioning queue done"
