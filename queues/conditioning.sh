#!/usr/bin/env bash
# Plan step 2 (2026-11-26): standard JiT conditioning vs the KV prefix, before the main study commits to one.
#   A  incontext  JiT's 32 in-context class tokens (prepended at block 4)            --in_context_len=32
#   B  kvtable    the KV prefix as a learned 32-token x 1,000-class table (+ null row)  --kv_prefix_len=32 --kv_prefix_class_tokens=True
# Both: plain JiT + Muon (match_rms_adamw), batch 1024, 500-update warm-up then constant lr, clean-image prediction, identical
# data / augmentation. Run length is not fixed: every run trains until its own curve has converged (jit_common.converged:
# unguided FID-10k improved by < 2% over the last third of training at 2 consecutive evals, never before 20 epochs; 400 epochs
# is only a cap), because the question is the trade-off between configurations, and the epochs one needs depend on the
# configuration (size, data, conditioning). Evals every 5 epochs of full-data training (unguided FID-10k, a kept checkpoint and
# post-hoc EMA snapshots each, sigma_rel 0.02 / 0.05 / 0.10 in bfloat16), so each run is also a whole FID-vs-compute curve.
# Selection: the final checkpoint's guided FID, post-hoc EMA sigma_rel 0.10 rebuilt at that step, a fixed budget of 6 guidance
# scales (eval_jit --cfg_search=6, CFG at all t) searched with 2K samples and the pick re-scored at FID-10K on fresh samples.
# Equal search allowance per arm, at W192 (12 x 192, 3 heads of 64) on full data:
#   1. screen  Muon lr (blr 1e-4 / 2e-4 / 4e-4) x pixel noise scale (0.5 / 1.0): 6 runs
#   2. aux     the AdamW groups' lr (embeddings incl. the prefix table, readout, norms, biases) x 0.5 and x 2 at the arm's best
#              screen setting: 2 runs
#   3. compute each arm's best at convergence, and at matched training compute: the longer arm's kept checkpoint nearest the
#              shorter arm's final compute, searched the same way (no extra training)
#   4. confirm each arm's best setting at S (12 x 384, 6 heads) on full data and at W192 on N/8, both until converged (on N/8
#              evals and the minimum length are counted in full-data epochs, so memorisation shows as a flat or rising curve)
#   5. report  each arm's best W192 run at its chosen scale with fresh FID-50K
# Decisions (the lowest FID-10K) are printed with ">>>" and written to runs/conditioning/*.txt (delete one to redo it).
#
#   tmux new -s cond
#   bash queues/conditioning.sh 2>&1 | tee -a runs/queue_logs/queue_conditioning.log
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
mkdir -p runs/queue_logs runs/conditioning
GPU_LOCK=runs/queue_logs/gpu.lock
CAP=400 # epochs: only a cap, runs stop at convergence
BASE="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA --warmup_updates=500 --phema_stds=0.02,0.05,0.10 \
--phema_dtype=bfloat16 --fid_num_images=10000 --converge_tol=0.02 --converge_patience=2 --diag=False --trend_plots=False --progress_gif=False"
W192="--hidden_size=192 --num_heads=3"
S="--hidden_size=384 --num_heads=6"
arm_flags() { case $1 in incontext) echo "--in_context_len=32" ;; kvtable) echo "--kv_prefix_len=32 --kv_prefix_class_tokens=True" ;; esac; }
SEARCH="--phema_std=0.10 --cfg_search=6 --sweep_num_images=2000 --final_num_images=10000"
FULL="--epochs=$CAP --eval_every_epochs=5 --converge_min_epochs=20" # full data
N8="--train_frac=0.125 --epochs=$((CAP * 8)) --eval_every_epochs=40 --converge_min_epochs=160" # N/8, in full-data epochs

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
eval_json() { python -c "import glob, sys; fs = glob.glob(sys.argv[1] + '/eval_*.json'); print(fs[0] if fs else '')" "$1"; }
best_fid() { # an evaluation directory's re-scored FID (the search's best when there was no re-scoring)
    local j; j=$(eval_json "$1"); [ -n "$j" ] || { echo "-"; return; }
    python -c "
import json, sys
r = json.load(open(sys.argv[1]))
f = r.get('final', {})
v = [f[k]['fid'] for k in f if isinstance(f[k], dict) and 'fid' in f[k]] or list(r['fid'].values())
print(min(v))" "$j"
}
best_cfg() { python -c "import json, sys; print(json.load(open(sys.argv[1]))['best_fid_cfg'])" "$(eval_json "$1")"; }
evaluate() { # step-name run-dir eval-dir-name [checkpoint file]: guided search of a checkpoint (default the run's last)
    [ -n "$2" ] && step "$1" python eval_jit.py --ckpt="${4:-$2/ckpt.pt}" --data_dir=$DATA $SEARCH --out_dir="$2/$3"
}
pick() { # file what step-name...: the step whose final evaluation has the lowest FID-10K; prints and stores "step fid"
    local file=runs/conditioning/$1.txt what=$2; shift 2
    if [ -e "$file" ]; then cat "$file"; return; fi
    local best="" bf="" s f
    for s in "$@"; do
        done_ok "${s}_eval" || { echo ">>> $what: $s not evaluated yet; not decided" >&2; return 1; }
        f=$(best_fid "$(run_of "$s")/eval_final")
        echo "    $s: FID-10K $f" >&2
        if [ -z "$bf" ] || python -c "import sys; sys.exit(0 if float('$f') < float('$bf') else 1)"; then best=$s; bf=$f; fi
    done
    echo "$best $bf" | tee "$file"
    echo ">>> $what: $best (FID-10K $bf)" >&2
}
train_eval() { # step-name training-args...: a full run and its final guided evaluation
    local n=$1; shift
    step "$n" python train_jit.py $BASE "$@"
    done_ok "$n" && evaluate "${n}_eval" "$(run_of "$n")" eval_final
}

# 1-2. per arm: screen, then the aux-group lr at the best screen setting
for arm in incontext kvtable; do
    A=$(arm_flags $arm)
    names=()
    for blr in 1e-4 2e-4 4e-4; do for ns in 0.5 1.0; do
        n="cond_${arm}_blr${blr}_noise${ns}"; names+=("$n")
        train_eval "$n" $W192 $A $(flags_of "$n") $FULL
    done; done
    best1=$(pick "${arm}_screen" "$arm screen (W192, converged)" "${names[@]}") || continue
    b=${best1%% *}
    for m in 0.5 2; do
        n="${b}_aux${m}"
        train_eval "$n" $W192 $A $(flags_of "$n") $FULL
    done
    pick "${arm}_best" "$arm best of screen + aux (W192, converged)" "$b" "${b}_aux0.5" "${b}_aux2" >/dev/null || continue
done

# 3. compute: each arm's best at convergence, and at matched training compute (the longer arm's kept checkpoint nearest the
#    shorter arm's final compute)
bi=$(cut -d' ' -f1 runs/conditioning/incontext_best.txt 2>/dev/null); bk=$(cut -d' ' -f1 runs/conditioning/kvtable_best.txt 2>/dev/null)
if [ -z "$bi" ] || [ -z "$bk" ]; then echo "=== arms not both decided (incontext '$bi', kvtable '$bk'): stopping"; exit 1; fi
cost() { # run-dir -> "final-step PFLOPs-per-step"
    python -c "
import json, sys, glob, re
r = json.load(open(sys.argv[1] + '/config.json'))['resolved']
it = max(int(re.search(r'diag_(\d+)\.json$', f).group(1)) for f in glob.glob(sys.argv[1] + '/diagnostics/diag_*.json')) # the final eval
print(it, r['fwd_gflops'] * 3 * r['effective_batch_size'] / 1e6)" "$1"
}
read -r ii pi_ <<< "$(cost "$(run_of "$bi")")"; read -r ik pk_ <<< "$(cost "$(run_of "$bk")")"
ci=$(python -c "print($ii * $pi_)"); ck=$(python -c "print($ik * $pk_)")
echo ">>> at convergence: incontext $bi: $ii updates, $ci PF | kvtable $bk: $ik updates, $ck PF"
if python -c "import sys; sys.exit(0 if $ck < $ci else 1)"; then long=$bi; lp=$pi_; target=$ck; else long=$bk; lp=$pk_; target=$ci; fi
d=$(run_of "$long")
kept=$(python -c "
import glob, re, sys
target, per = float(sys.argv[2]), float(sys.argv[3])
its = sorted(int(re.search(r'ckpt_(\d+)\.pt$', f).group(1)) for f in glob.glob(sys.argv[1] + '/ckpt_*.pt'))
print(min(its, key=lambda i: abs(i * per - target)) if its else '')" "$d" "$target" "$lp")
if [ -n "$kept" ]; then
    echo ">>> matched compute: $long at update $kept ($(python -c "print($kept * $lp)") PF) vs the other arm's $target PF at convergence"
    evaluate "${long}_matched_eval" "$d" eval_matched "$d/ckpt_${kept}.pt"
fi

# 4. confirm at S (full data) and at W192 on N/8, both until converged
for arm in incontext kvtable; do
    s=$([ $arm = incontext ] && echo "$bi" || echo "$bk")
    train_eval "cond_${arm}_S" $S $(arm_flags $arm) $(flags_of "$s") $FULL
    train_eval "cond_${arm}_W192_frac0.125" $W192 $(arm_flags $arm) $(flags_of "$s") $N8
done

# 5. fresh FID-50K of each arm's best W192 run at its chosen scale (seed 2: samples neither the search nor the 10K re-score used)
for s in "$bi" "$bk"; do
    sd=$(run_of "$s"); c=$(best_cfg "$sd/eval_final")
    step "${s}_fid50k" python eval_jit.py --ckpt="$sd/ckpt.pt" --data_dir=$DATA --phema_std=0.10 --cfgs=$c --final_num_images=50000 \
        --seed=2 --out_dir="$sd/eval_50k"
done

echo ">>> summary (guided FID; W192 unless noted)"
for s in "$bi" "$bk"; do echo "    $s: FID-10K $(best_fid "$(run_of "$s")/eval_final"), FID-50K $(best_fid "$(run_of "$s")/eval_50k")"; done
[ -n "$kept" ] && echo "    $long at update $kept (matched compute, $target PF): FID-10K $(best_fid "$d/eval_matched")"
for s in cond_incontext_S cond_kvtable_S cond_incontext_W192_frac0.125 cond_kvtable_W192_frac0.125; do
    echo "    $s: FID-10K $(best_fid "$(run_of "$s")/eval_final")"
done
echo "=== $(date '+%F %T') conditioning queue done"
