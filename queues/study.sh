#!/usr/bin/env bash
# The pixels-vs-DiTo-latents allocation study (plan of 2026-11-26, lean tuning of 2026-11-27), end to end, resumable: rerun
# this script after any interruption and it skips finished steps, resumes interrupted training from its last checkpoint and
# re-reads every decision it already made (runs/study/decisions/*.txt; delete one to redo that decision and what follows).
# Question: at a given data size and TOTAL TRAINING compute (tokenizer training + dataset encoding + generator training;
# inference is not part of this study), are pixels or DiTo latents the better conditional generator, and what allocation
# (generator size, training length, tokenizer investment) is best? Predicted from widths up to S, validated at B.
#
# Fixed: 128px center crops + flips; generators batch 1024, 500-update warm-up then constant lr, Muon (match_rms_adamw: the
# AdamW lr carries over) + AdamW side groups. Pixel generator: JiT /16, clean-image prediction. Latent generator: the same
# blocks on DiTo f8c4 latents (4 x 16 x 16, patch 2) with the SiT / DiT recipe the DiTo authors use for their latent model
# (velocity prediction, uniform t, flow matching, AdamW 1e-4 at batch 256 -> its own lr search here at batch 1024). DiTo
# trained as the paper (dito_model.py) at batch 64, Muon from the paper's AdamW lr. Widths W192 / W256 / S (12 x 192 / 256 /
# 384, heads of 64); B (12 x 768) held out. Data N, N/2, N/4, N/8 (nested, class-stratified: subsets.py).
# Full runs: 100 full-data epochs (subsets: the same number of updates), earlier only if the curve flattens (< 2% over the
# last third at 2 consecutive evals). Every eval (every 5 full-data epochs) of every run: unguided FID-10k AND guided FID-2K
# (post-hoc EMA 0.10, used directly; a 6-scale CFG search the first time, then the previous best scale +- 0.5), a kept
# checkpoint, post-hoc EMA snapshots (sigma_rel 0.02 / 0.05 / 0.10, bfloat16), and predictions of the converged (epoch-100)
# unguided and guided FID (predict_fid.py). End of every run: CFG search (6 scales, 2K, EMA 0.10), then post-hoc EMA 0.02 /
# 0.05 at that scale, then the best pair re-scored at FID-10K on fresh samples (B and seeds: also fresh FID-50K). Nothing
# else is scored after training.
# Tuning candidates (every search below): train with the same in-loop evals and stop when their predicted converged guided
# FID has settled (< 2% change between consecutive evals, twice; after 20, by 50 full-data epochs) or, after 20, when it is
# > 10% worse than the search's best so far. Ranked by that prediction (tune_rank.py); the winner resumes to 100 epochs and is
# its matrix cell (nothing trained twice); its prediction is checked against what it reaches (prediction_check.json).
# Searches: learning rate like the CFG search (hp_search.py: from the literature value, x2 until bracketed, then log
# bisection toward the better neighbour); weight decay as the averaging timescale tau = 1 / (lr wd) in full runs (none, tau = 1,
# then 0.25 or 4; every run gets wd = 1 / (lr tau T), so lr x wd transfers across widths and data sizes).
# Decisions, in order (runs/study/decisions/):
#   D1  conditioning: in-context (conditioning.sh's run, lr 1e-4, noise 1.0) vs one KV-prefix run at the same setting, full
#       runs; final guided FID at matched training compute (the costlier arm's kept checkpoint), within 2% -> the cheaper arm
#   D2  pixel noise scale: 0.5 vs 1.0 (conditioning.sh's two finished in-context runs), final guided FID
#   D3  pixel lr (W192): from 2e-4 (JiT's AdamW lr at batch 1024, kept by Muon match_rms), 4 candidates (1e-4 = the conditioning
#       run when it has the same setting)
#   D4  pixel weight decay (W192): 2 candidates
#   D5  pixel bottleneck at S: proportional 64 vs fixed 128 (earlier 20-epoch evidence: 77.85 vs 82.69), 2 candidates
#   D6  pixel lr width transfer at S: D3's runner-up vs D5's winner, 1 candidate (the winner's lr: W256, S, B)
#   D7  tokenizer lr (DiTo-$TOK_SIZE, batch 64 as the paper): from 1e-4 (the paper's), 3 candidates, each trained until its
#       predicted rFID at T100 settles (25k-75k updates) and judged by a W192 latent probe (a candidate) on it; the winner and
#       N/2, N/4, N/8 at its lr then train to the study's investments: T_early = T25 (25k updates), T_full = T100 (100k)
#   D8  latent lr (W192, full data, T_full): from 2e-4 (SiT / DiTo's AdamW 1e-4 at batch 256, sqrt batch scaling; Muon), 4
#   D9  latent weight decay: 2 candidates
#   D10 N/8 lr per family (W192): from the full-data lr, 3 candidates each (N/4, N/2: the full-data lr)
# Order (the hypothesis at full data as early as possible): D1-D9 (all tuning on full data, incl. the full-data tokenizer to
# T100), then the FULL-DATA STUDY: its matrix (N x {W192, W256, S} x {pixel, latent T_early, latent T_full}; 3 cells are tuning
# winners), the frontier, B forecasts frozen, B (pixel, latent T_early, latent T_full) and its validation. Only then the smaller
# data sizes: their tokenizers (each on its own subset, at D7's lr, to T25 / T100), D10, their matrices (x {W192, S}), the
# frontier over all data; then seeds (RUN_SEEDS=1), L (RUN_L=1).
# Matrix cells are predicted too: each trains until its predicted converged guided FID settles (>= 20 epochs), and is extended
# to the full 100 epochs only where the pixel-vs-latent decision could depend on it (extend_plan.py: where the two families'
# best predicted guided FIDs over total training compute are within 10% and the budget lies past what the run measured).
#
#   tmux new -s study
#   bash queues/study.sh 2>&1 | tee -a runs/queue_logs/queue_study.log
#   TOK_SIZE=B bash queues/study.sh ...        # DiTo-B tokenizers (the paper's size; ~4x DiTo-S's training compute)
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
STUDY=${STUDY:-runs/study}
TOK_SIZE=${TOK_SIZE:-S}
RUN_SEEDS=${RUN_SEEDS:-1}
RUN_L=${RUN_L:-0}
FRACS="1 0.5 0.25 0.125"
mkdir -p runs/queue_logs "$STUDY/decisions" "$STUDY/args" "$STUDY/runs"
GPU_LOCK=runs/queue_logs/gpu.lock
D=$STUDY/decisions
source queues/fast_data.sh # the image arrays on local NVMe for this queue (removed on exit; falls back to $DATA)
CAP=100 # full-data epochs (subsets: the same number of updates): a full run, unless the curve flattens first
TUNE=20 # full-data epochs of a tuning candidate
TOK_T100=100000 # the study's tokenizer investments (the plan's T25 / T100): 25k and 100k updates at batch 64
TOK_T25=25000
TOK_TUNE=30000 # tokenizer updates of an lr candidate
EVAL="--phema_stds=0.02,0.05,0.10 --phema_dtype=bfloat16 --fid_num_images=10000 --guided_eval=True --converge_tol=0.02 --converge_patience=2 --warmup_updates=500"
PIX_BASE="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA $EVAL --diag=False --trend_plots=False --progress_gif=False" # (the diagnostic suite does not feed the study)
LAT_BASE="--data_dir=$DATA --recipe=sit --optimizer_type=muon --patch_size=2 $EVAL" # SiT / DiT objective (v, uniform t), Muon
width_flags() { echo "--hidden_size=$1 --num_heads=$(( $1 / 64 ))"; }
len_flags() { # epochs data-fraction -> run length, eval spacing and minimum length in full-data epochs (subsets: same updates)
    python -c "e, f = float('$1'), float('$2'); print(f'--epochs={int(round(e / f))} --eval_every_epochs={int(round(5 / f))} --converge_min_epochs={int(round(20 / f))}' + ('' if f >= 1 else f' --train_frac={f:g}'))"
}
length_flags() { len_flags $CAP "$1"; }

# ---- machinery ------------------------------------------------------------------------------------------------------------
step() { # step name command...: skipped if done; output to its log and the screen; interrupted training resumes
    local name=$1; shift
    local log="runs/queue_logs/$name.log" done="runs/queue_logs/$name.done"
    if [ -e "$done" ]; then echo "skip $name (done)"; return; fi
    local extra=() mode=">"
    case "$2" in train_jit.py|train_latent.py|train_dito.py)
        if [ -e "$log" ]; then
            local dir; dir=$(grep -m1 '^run:  ' "$log" | sed 's/^run:  //')
            if [ -n "$dir" ] && [ -e "$dir/ckpt.pt" ]; then extra=(--init_from=resume); mode=">>"; fi
        fi ;;
    esac
    echo "=== $(date '+%F %T') start $name ${extra[*]+${extra[*]}}: $*"
    local guarded=(flock "$GPU_LOCK" bash -c '[ -e "$0" ] && exit 0; exec "$@"' "$done" "$@" ${extra[@]+"${extra[@]}"})
    local tee_args=(); [ "$mode" = ">>" ] && tee_args=(-a)
    "${guarded[@]}" 2>&1 | tee ${tee_args[@]+"${tee_args[@]}"} "$log"
    local rc=${PIPESTATUS[0]}
    [ $rc -eq 0 ] && touch "$done"
    echo "=== $(date '+%F %T') end $name (exit $rc)"
}
done_ok() { [ -e "runs/queue_logs/$1.done" ]; }
need() { for s in "$@"; do done_ok "$s" || { echo "=== waiting on $s (not finished): stopping here; rerun to continue"; exit 1; }; done; }
rdir() { echo "$STUDY/runs/$1"; }
eval_fid() { # an evaluation directory's re-scored FID (or the search's best)
    python -c "
import json, glob, sys
fs = glob.glob(sys.argv[1] + '/eval_*.json')
if not fs: print(''); sys.exit()
r = json.load(open(fs[0])); f = r.get('final', {})
v = [x['fid'] for x in f.values() if isinstance(x, dict) and 'fid' in x] or list(r['fid'].values())
print(min(v))" "$1"
}
eval_cfg() { python -c "import json, glob, sys; print(json.load(open(glob.glob(sys.argv[1] + '/eval_*.json')[0]))['best_fid_cfg'])" "$1"; }
lt() { python -c "import sys; sys.exit(0 if float('$1') < float('$2') else 1)"; }
final_iter() { python -c "
import glob, re, sys
its = [int(re.search(r'diag_(\d+)\.json$', f).group(1)) for f in glob.glob(sys.argv[1] + '/diagnostics/diag_*.json')]
print(max(its) if its else '')" "$1"; }
decide() { # name what id... : the run whose final evaluation has the lowest FID-10K; stored as "id fid"
    local f=$D/$1.txt what=$2; shift 2
    if [ -e "$f" ]; then cat "$f"; return; fi
    local best="" bf="" id v
    for id in "$@"; do
        v=$(eval_fid "$(rdir "$id")/eval_final")
        [ -n "$v" ] || { echo "=== $what: $id has no final evaluation yet; stopping" >&2; exit 1; }
        echo "    $id: FID-10K $v" >&2
        if [ -z "$bf" ] || lt "$v" "$bf"; then best=$id; bf=$v; fi
    done
    echo "$best $bf" | tee "$f"
    echo ">>> $what: $best (FID-10K $bf)" >&2
}

# generator runs: a run = id + family + args file; the args rebuild the network for its evaluations
train_gen() { # id family(pixel|latent) args...: a full run, then its final evaluation
    local id=$1 fam=$2; shift 2
    local dir; dir=$(rdir "$id")
    echo "$*" > "$STUDY/args/$id.txt"
    if [ "$fam" = pixel ]; then step "$id" python train_jit.py $PIX_BASE "$@" --out_dir="$dir"
    else step "$id" python train_latent.py $LAT_BASE "$@" --out_dir="$dir"; fi
    done_ok "$id" || return 1
    final_eval "$id" "$fam"
}
eval_one() { # step id family ckpt-iter sigma n-search n-final out-dir [extra]: one guided search (or a fixed scale)
    local name=$1 id=$2 fam=$3 it=$4 sig=$5 ns=$6 nf=$7 out=$8 extra=${9:-}
    local dir; dir=$(rdir "$id")
    if [ "$fam" = pixel ]; then
        step "$name" python eval_jit.py --ckpt="$dir/ckpt_$it.pt" --data_dir=$DATA --phema_std=$sig --sweep_num_images=$ns \
            --final_num_images=$nf --out_dir="$out" $extra
    else
        step "$name" python train_latent.py $LAT_BASE $(cat "$STUDY/args/$id.txt") --out_dir="$dir" --init_from=resume --eval_phema=$sig \
            --eval_at=$it --fid_num_images=$ns --eval_final_images=$([ "$nf" -gt "$ns" ] && echo "$nf" || echo 0) --eval_out_dir="$out" $extra
    fi
}
final_eval() { # id family [ckpt-iter] [tag]: CFG first (6-scale search, 2K, post-hoc EMA 0.10), then the post-hoc EMA (0.02 / 0.05)
               # at that scale, then the best pair re-scored at FID-10K on fresh samples -> {run}/eval_{tag}, {tag}_choice.txt
    local id=$1 fam=$2 dir; dir=$(rdir "$id")
    local it=${3:-$(final_iter "$dir")} tag=${4:-final}
    eval_one "${id}_${tag}${it}_cfg" "$id" "$fam" "$it" 0.10 2000 2000 "$dir/eval_${tag}_cfg" "--cfg_search=6"
    local c; c=$(eval_cfg "$dir/eval_${tag}_cfg") || return 1
    local fixed; fixed=$([ "$fam" = pixel ] && echo "--cfgs=$c" || echo "--eval_cfgs=$c")
    local bs=0.10 bf s v; bf=$(eval_fid "$dir/eval_${tag}_cfg")
    for s in 0.02 0.05; do
        eval_one "${id}_${tag}${it}_ema$s" "$id" "$fam" "$it" "$s" 2000 2000 "$dir/eval_${tag}_ema$s" "$fixed"
        v=$(eval_fid "$dir/eval_${tag}_ema$s"); [ -n "$v" ] || return 1
        if lt "$v" "$bf"; then bf=$v; bs=$s; fi
    done
    eval_one "${id}_${tag}${it}" "$id" "$fam" "$it" "$bs" 10000 10000 "$dir/eval_$tag" "$fixed --seed=1"
    done_ok "${id}_${tag}${it}" || return 1
    echo "$bs $c" > "$dir/${tag}_choice.txt"
    echo "$it" > "$dir/${tag}_at.txt"
}
final_done() { # id: its final evaluation is at its current last checkpoint
    local dir; dir=$(rdir "$1")
    [ -e "$dir/final_at.txt" ] && [ "$(cat "$dir/final_at.txt")" = "$(final_iter "$dir")" ]
}
fid50k() { # id family: fresh FID-50K at the final choice (seed 2)
    local id=$1 fam=$2 dir; dir=$(rdir "$id")
    final_done "$id" || return 1
    read -r s c < "$dir/final_choice.txt"
    local it; it=$(final_iter "$dir")
    if [ "$fam" = pixel ]; then eval_one "${id}_50k" "$id" pixel "$it" "$s" 50000 50000 "$dir/eval_50k" "--cfgs=$c --seed=2"
    else eval_one "${id}_50k" "$id" latent "$it" "$s" 50000 50000 "$dir/eval_50k" "--eval_cfgs=$c --seed=2"; fi
}
manifest() { # id family width frac tokenizer-ckpt|-: one line per discovery / validation run (frontier.py)
    local m=$STUDY/manifest.jsonl
    grep -q "\"id\": \"$1\"" "$m" 2>/dev/null && return
    python -c "
import json, sys
id_, fam, w, frac, tok = sys.argv[1:6]
rec = {'id': id_, 'family': fam, 'width': int(w), 'data_frac': float(frac), 'run': sys.argv[6], 'tokenizer': None if tok == '-' else tok,
       'tokenizer_pflops': 0.0, 'encode_pflops': 0.0}
if tok != '-': # the latent run's own record of its tokenizer's measured training compute and the dataset encoding (train_latent.py)
    cfg = json.load(open(sys.argv[6] + '/config.json'))['resolved']
    rec['tokenizer_pflops'] = cfg.get('vae_train_pflops', 0.0)
    rec['encode_pflops'] = cfg.get('cached_encoding_pflops', 0.0)
print(json.dumps(rec))" "$1" "$2" "$3" "$4" "$5" "$(rdir "$1")" >> "$m"
}

# ---- 0. subsets, preflight ---------------------------------------------------------------------------------------------------
step study_subsets python subsets.py --data_dir=$DATA --fracs=1,0.5,0.25,0.125
if [ "${PREFLIGHT:-1}" = 1 ]; then # every new code path for a few minutes, so a crash shows now rather than days in (runs/study/preflight)
    P=$STUDY/preflight
    step pre_pixel python train_jit.py $PIX_BASE $(width_flags 192) --kv_prefix_len=32 --kv_prefix_class_tokens=True --aux_lr_mult=0.5 \
        --train_frac=0.125 --max_updates=200 --eval_every_epochs=1000 --budget_keep=1/2,1 --fid_num_images=1000 --out_dir=$P/pixel
    step pre_pixel_eval python eval_jit.py --ckpt=$P/pixel/ckpt_200.pt --data_dir=$DATA --phema_std=0.10 --cfg_search=3 \
        --sweep_num_images=1000 --final_num_images=2000 --out_dir=$P/pixel/eval
    step pre_dito python train_dito.py --data_dir=$DATA --size=$TOK_SIZE --max_updates=400 --eval_every=200 --eval_images=1000 \
        --converge_tol=0.02 --converge_min_updates=200 --out_dir=$P/dito
    step pre_latents python prepare_latents.py --data_dir=$DATA --vae=dito:$P/dito/ckpt_400.pt
    step pre_latent python train_latent.py $LAT_BASE $(width_flags 192) --vae=dito:$P/dito/ckpt_400.pt --kv_prefix_len=32 \
        --kv_prefix_class_tokens=True --prediction=x --dropout=0.1 --train_frac=0.125 --max_updates=200 --eval_every_epochs=1000 \
        --budget_keep=1/2,1 --fid_num_images=1000 --out_dir=$P/latent
    step pre_latent_eval python train_latent.py $LAT_BASE $(width_flags 192) --vae=dito:$P/dito/ckpt_400.pt --kv_prefix_len=32 \
        --kv_prefix_class_tokens=True --prediction=x --dropout=0.1 --train_frac=0.125 --out_dir=$P/latent --init_from=resume \
        --eval_phema=0.10 --eval_at=200 --cfg_search=3 --fid_num_images=1000 --eval_final_images=2000 --eval_out_dir=$P/latent/eval
    need pre_pixel pre_pixel_eval pre_dito pre_latents pre_latent pre_latent_eval
    echo "=== preflight passed (delete runs/queue_logs/pre_*.done to rerun it; PREFLIGHT=0 skips it)"
fi

# ---- tuning machinery -------------------------------------------------------------------------------------------------------
# A tuning candidate trains with in-loop guided FID at every eval and stops when its predicted converged (epoch-100) guided FID
# has settled (< 2% change between consecutive evals, twice), after 20 and by 50 full-data epochs, or as soon as (after 20) its
# prediction is > 10% worse than the best finished candidate of the same search. Candidates are ranked by that prediction
# (tune_rank.py); the winner resumes to the full 100 epochs and becomes its matrix cell. Nothing is scored after training
# except each finished run's final evaluation.
SETTLE="--settle_tol=0.02 --settle_patience=2 --settle_min_epochs=20 --settle_max_epochs=50"
CAND_MAX=50
KILL=1.10
short() { # id family frac kill(or 0) args...: a tuning candidate
    local id=$1 fam=$2 frac=$3 kill=$4; shift 4
    echo "$*" > "$STUDY/args/$id.txt"
    local k=""; python -c "import sys; sys.exit(0 if float('$kill') > 0 else 1)" && k="--settle_kill=$kill"
    if [ "$fam" = pixel ]; then step "$id" python train_jit.py $PIX_BASE "$@" $(len_flags $CAND_MAX $frac) $SETTLE $k --out_dir="$(rdir "$id")"
    else step "$id" python train_latent.py $LAT_BASE "$@" $(len_flags $CAND_MAX $frac) $SETTLE $k --out_dir="$(rdir "$id")"; fi
}
kill_at() { # candidate ids...: KILL x the best predicted converged guided FID among those that have finished (0 if none)
    python -c "
import sys, os, predict_fid
best = None
for i in sys.argv[2:]:
    d = os.path.join(sys.argv[1], 'runs', i)
    if not os.path.exists(os.path.join(d, 'config.json')): continue
    p = predict_fid.predict(predict_fid.history(d, 'val/guided_fid', $CAND_MAX), $CAP)
    if p is not None and (best is None or p < best): best = p
print(best * $KILL if best else 0)" "$STUDY" "$@"
}
rank() { # name id...: the candidate with the best predicted converged guided FID (tune_rank.py); prints its id
    local name=$1; shift
    [ -e "$D/$name.txt" ] || python tune_rank.py --study=$STUDY --name=$name --target_epochs=$CAP --max_epochs=$CAND_MAX "$@" >&2 || return 1
    cut -d' ' -f1 "$D/$name.txt"
}
extend() { # id family frac [decision name]: resume a winning candidate to full length, its final evaluation, the prediction check
    local id=$1 fam=$2 frac=$3 name=${4:-} dir; dir=$(rdir "$id")
    if [ "$fam" = pixel ]; then step "${id}_full" python train_jit.py $PIX_BASE $(cat "$STUDY/args/$id.txt") $(length_flags $frac) --out_dir="$dir" --init_from=resume
    else step "${id}_full" python train_latent.py $LAT_BASE $(cat "$STUDY/args/$id.txt") $(length_flags $frac) --out_dir="$dir" --init_from=resume; fi
    done_ok "${id}_full" || return 1
    final_eval "$id" "$fam"
    [ -e "$dir/settled.json" ] && python -c "
import json, sys, predict_fid
d = sys.argv[1]; s = json.load(open(d + '/settled.json'))
g = predict_fid.history(d, 'val/guided_fid')
if g and s.get('pred_guided'):
    m = g[-1][1]
    print(f\">>> prediction check {sys.argv[2]}: predicted converged guided FID {s['pred_guided']:.2f} at {s['epochs']:.0f} epochs, measured {m:.2f} at {g[-1][0]:.0f} ({s['pred_guided'] / m - 1:+.1%})\")
    json.dump({'predicted': s['pred_guided'], 'measured': m, 'stopped_at_epochs': s['epochs']}, open(d + '/prediction_check.json', 'w'))" "$dir" "$id"
    return 0
}
scale_lr() { python -c "print(f'{float(\"$1\") * float(\"$2\"):g}')"; } # lr factor
lr_of() { grep -o -- '--blr=[^ ]*' "$STUDY/args/$1.txt" | tail -1 | cut -d= -f2; }
LR_BUDGET=4 # candidates per learning-rate search (x2 until bracketed, then log bisection toward the better neighbour)
T_FULL=$(( CAP * 1251 )) # updates of a full run (every data size: the same number of updates)
wd_of() { python -c "s, lr = float('$1'), float('$2'); print(f'{s / (lr * $T_FULL):.3g}' if s else '0')"; } # decay strength s, lr -> weight decay
# weight decay as the averaging timescale tau = 1 / (lr wd) in units of a full run (s = 1 / tau): the quantity that transfers
# across widths and data sizes (AdamW timescale scaling: Wang & Aitchison; Bergsma et al.), so every run gets wd = s / (lr T)
lr_search() { # name family frac start budget prefix alias(value=id or -) fixed-args...: adaptive lr search; prints the winner id
    local name=$1 fam=$2 frac=$3 start=$4 budget=$5 prefix=$6 al=$7; shift 7
    local hs="--study=$STUDY --prefix=$prefix --start=$start --step=2 --budget=$budget --target_epochs=$CAP --max_epochs=$CAND_MAX"
    [ "$al" = - ] || hs="$hs --alias=$al"
    local k v
    for k in $(seq 1 $budget); do
        v=$(python hp_search.py next $hs) || return 1
        case $v in done) break ;; wait) echo "=== $name: a candidate has not finished; stopping" >&2; return 1 ;; esac
        if [ "$al" != - ] && python -c "import sys; sys.exit(0 if abs(float('$v') - float('${al%%=*}')) < 1e-12 else 1)"; then
            : # the alias run (a finished full run at this value) is the candidate
        else
            short "$prefix$v" "$fam" "$frac" "$(kill_at $(python hp_search.py ids $hs))" "$@" --blr=$v >&2
        fi
    done
    rank "$name" $(python hp_search.py ids $hs)
}
wd_search() { # name family frac lr baseline-id prefix fixed-args...: weight decay s in {0 (baseline), 1, then 4 or 0.25}; prints the winner id
    local name=$1 fam=$2 frac=$3 lr=$4 base=$5 prefix=$6; shift 6
    short "${prefix}1" "$fam" "$frac" "$(kill_at "$base")" "$@" --blr=$lr --weight_decay=$(wd_of 1 $lr) >&2
    local w1; w1=$(rank "${name}_1" "$base" "${prefix}1") || return 1
    local s2; s2=$([ "$w1" = "${prefix}1" ] && echo 4 || echo 0.25)
    short "${prefix}$s2" "$fam" "$frac" "$(kill_at "$base" "${prefix}1")" "$@" --blr=$lr --weight_decay=$(wd_of $s2 $lr) >&2
    rank "$name" "$base" "${prefix}1" "${prefix}$s2"
}
s_of() { case $1 in *_wd*) echo "${1##*_wd}" ;; *) echo 0 ;; esac; } # winner id -> its decay strength

# ---- 1. conditioning (D1) and pixel noise (D2) ---------------------------------------------------------------------------------
# the in-context runs conditioning.sh started (lr 1e-4; noise 1.0 is finished here if it is not), linked into the study
link_cond() { # study-id conditioning-step args: finish (if needed) and link a conditioning.sh run
    local id=$1 st=$2 args=$3 dir
    dir=$(grep -m1 '^run:  ' "runs/queue_logs/$st.log" 2>/dev/null | sed 's/^run:  //')
    [ -n "$dir" ] || { echo "=== $st: no run directory in runs/queue_logs/$st.log; stopping"; exit 1; }
    step "$st" python train_jit.py $PIX_BASE $args $(length_flags 1) --out_dir="$dir"
    need "$st"
    [ -e "$(rdir "$id")" ] || ln -s "$(cd "$dir" && pwd)" "$(rdir "$id")"
    echo "$args" > "$STUDY/args/$id.txt"
    final_eval "$id" pixel
}
link_cond cond_incontext cond_incontext_blr1e-4_noise1.0 "$(width_flags 192) --in_context_len=32 --blr=1e-4 --noise_scale=1.0 --aux_lr_mult=1.0"
link_cond cond_incontext_n05 cond_incontext_blr1e-4_noise0.5 "$(width_flags 192) --in_context_len=32 --blr=1e-4 --noise_scale=0.5 --aux_lr_mult=1.0"
train_gen cond_kvtable pixel $(width_flags 192) --kv_prefix_len=32 --kv_prefix_class_tokens=True --blr=1e-4 --noise_scale=1.0 --aux_lr_mult=1.0 $(length_flags 1)
for i in cond_incontext cond_incontext_n05 cond_kvtable; do final_done $i || { echo "=== $i: no final evaluation yet; stopping"; exit 1; }; done
if [ ! -e "$D/conditioning.txt" ]; then
    # D1 at matched training compute: the arm that used more compute, at its kept checkpoint nearest the other arm's final compute
    read -r long short_ kept <<< "$(python -c "
import json, glob, re, sys
st = sys.argv[1]
def info(i):
    d = f'{st}/runs/{i}'; r = json.load(open(d + '/config.json'))['resolved']
    per = r['fwd_gflops'] * 3 * r['effective_batch_size'] / 1e6
    its = sorted(int(re.search(r'ckpt_(\d+)\.pt$', f).group(1)) for f in glob.glob(d + '/ckpt_*.pt'))
    return per, its
pi, ii = info('cond_incontext'); pk, ik = info('cond_kvtable')
(l, pl, il), (s, ps, is_) = sorted([('cond_incontext', pi, ii), ('cond_kvtable', pk, ik)], key=lambda x: -x[1] * x[2][-1])
print(l, s, min(il, key=lambda i: abs(i * pl - ps * is_[-1])))" "$STUDY")"
    final_eval "$long" pixel "$kept" matched
    [ -e "$(rdir "$long")/matched_at.txt" ] || { echo "=== $long: matched-compute evaluation not finished; stopping"; exit 1; }
    fl=$(eval_fid "$(rdir "$long")/eval_matched"); fs=$(eval_fid "$(rdir "$short_")/eval_final")
    if python -c "import sys; sys.exit(0 if float('$fl') < 0.98 * float('$fs') else 1)"; then w=$long; else w=$short_; fi
    echo "${w#cond_}" > "$D/conditioning.txt"
    echo ">>> D1 conditioning at matched training compute: $long (update $kept) $fl vs $short_ (final) $fs -> ${w#cond_} (within 2%: the cheaper arm)"
fi
ARM=$(cat "$D/conditioning.txt")
if [ ! -e "$D/noise.txt" ]; then
    f1=$(eval_fid "$(rdir cond_incontext)/eval_final"); f05=$(eval_fid "$(rdir cond_incontext_n05)/eval_final")
    n=$(lt "$f05" "$f1" && echo 0.5 || echo 1.0); echo "$n" > "$D/noise.txt"
    echo ">>> D2 pixel noise scale (in-context, lr 1e-4, final guided FID-10K): 1.0 $f1 vs 0.5 $f05 -> $n"
fi
NOISE=$(cat "$D/noise.txt")
arm_flags() { case $1 in incontext) echo "--in_context_len=32" ;; kvtable) echo "--kv_prefix_len=32 --kv_prefix_class_tokens=True" ;; esac; }
COND=$(arm_flags "$ARM")
echo "=== conditioning: $ARM; pixel noise scale $NOISE"

# ---- 2. pixel lr (D3) and weight decay (D4), W192, full data ------------------------------------------------------------------------
# start: JiT's AdamW lr at batch 1024 (2e-4; Muon match_rms keeps the AdamW lr). The finished conditioning run is the 1e-4
# candidate when it has the same setting (same arm, noise, no decay)
AL=-; [ "$ARM" = kvtable ] && [ "$NOISE" = 1.0 ] && AL=1e-4=cond_kvtable
[ "$ARM" = incontext ] && [ "$NOISE" = 1.0 ] && AL=1e-4=cond_incontext
[ "$ARM" = incontext ] && [ "$NOISE" = 0.5 ] && AL=1e-4=cond_incontext_n05
LW1=$(lr_search pixel_lr pixel 1 2e-4 $LR_BUDGET tune_pix_lr $AL $(width_flags 192) $COND --noise_scale=$NOISE) || exit 1
PIX_LR=$(lr_of "$LW1")
PW=$(wd_search pixel_wd pixel 1 $PIX_LR "$LW1" tune_pix_wd $(width_flags 192) $COND --noise_scale=$NOISE) || exit 1
PIX_S=$(s_of "$PW")
[ -e "$STUDY/runs/$PW/settled.json" ] && { extend "$PW" pixel 1 pixel_wd || exit 1; }
echo "=== pixel (W192): lr $PIX_LR, weight decay s $PIX_S (wd $(wd_of $PIX_S $PIX_LR)); cell $PW"

# ---- 3. pixel S: bottleneck (D5), then the lr's width transfer (D6) -----------------------------------------------------------------
PS="$COND --noise_scale=$NOISE"
short tune_pixS_bn128 pixel 1 0 $(width_flags 384) --bottleneck_dim=128 $PS --blr=$PIX_LR --weight_decay=$(wd_of $PIX_S $PIX_LR)
short tune_pixS_bn64 pixel 1 "$(kill_at tune_pixS_bn128)" $(width_flags 384) --bottleneck_dim=64 $PS --blr=$PIX_LR --weight_decay=$(wd_of $PIX_S $PIX_LR)
BW=$(rank pixel_S_bottleneck tune_pixS_bn64 tune_pixS_bn128) || exit 1
BN_RULE=$([ "$BW" = tune_pixS_bn64 ] && echo proportional || echo fixed)
bn_of() { if [ "$BN_RULE" = proportional ]; then echo $(( ($1 + 3) / 6 )); else echo 128; fi; } # width / 6, rounded
gen_width() { if [ "$1" = pixel ]; then echo "$(width_flags $2) --bottleneck_dim=$(bn_of $2)"; else width_flags $2; fi; }
echo "=== pixel bottleneck rule: $BN_RULE (W192 $(bn_of 192), W256 $(bn_of 256), S $(bn_of 384), B $(bn_of 768))"
NB=$(python -c "import json; r = json.load(open('$D/pixel_lr.json'))['rows']; print(r[1]['id'])") # the W192 search's runner-up lr
NLR=$(lr_of "$NB")
short tune_pixS_lr$NLR pixel 1 "$(kill_at "$BW")" $(gen_width pixel 384) $PS --blr=$NLR --weight_decay=$(wd_of $PIX_S $NLR)
SW=$(rank pixel_S_lr "$BW" tune_pixS_lr$NLR) || exit 1
extend "$SW" pixel 1 pixel_S_lr || exit 1
S_LR=$(lr_of "$SW")
echo "=== pixel lr at S: $S_LR (W192: $PIX_LR); used for W256, S and B"

# ---- 4. tokenizer lr (D7) and the full-data tokenizer ---------------------------------------------------------------------------
tok_id() { echo "tok_${TOK_SIZE}_frac$1_lr$2"; } # data fraction, lr
latents() { # tokenizer checkpoint: encode the dataset (both flips) and its rFID
    local ck=$1 tag; tag=$(echo "$ck" | sed 's|.*/runs/||; s|/ckpt_|_|; s|\.pt$||; s|/|_|g')
    [ -e "$ck" ] || { echo "=== tokenizer checkpoint $ck does not exist: stopping"; exit 1; }
    step "lat_$tag" python prepare_latents.py --data_dir=$DATA --vae=dito:$ck
    step "rfid_$tag" python latent_rfid.py --data_dir=$DATA --vae=dito:$ck
}
study_tokenizer() { # frac: the study's tokenizer for this data size (its own subset, the chosen lr) to T100; T25 / T100 latents
    local id; id=$(tok_id $1 $TOK_LR)
    step "${id}_full" python train_dito.py --data_dir=$DATA --size=$TOK_SIZE --lr=$TOK_LR --max_updates=$TOK_T100 --eval_every=12500 \
        $( [ "$1" = 1 ] || echo "--train_frac=$1" ) --out_dir="$(rdir "$id")" $( [ -e "$(rdir "$id")/ckpt.pt" ] && echo --init_from=resume )
    need "${id}_full"
    latents "$(rdir "$id")/ckpt_$TOK_T25.pt"; latents "$(rdir "$id")/ckpt_$TOK_T100.pt"
    echo "$(rdir "$id")/ckpt_$TOK_T25.pt" > "$D/tok_early_frac$1.txt"; echo "$(rdir "$id")/ckpt_$TOK_T100.pt" > "$D/tok_full_frac$1.txt"
}
tok_early() { cat "$D/tok_early_frac$1.txt"; }
tok_full() { cat "$D/tok_full_frac$1.txt"; }
# candidates: from the paper's AdamW lr (1e-4 at the paper's batch 64; Muon match_rms keeps it), adaptive, 3; each tokenizer trains
# until its predicted rFID at T100 settles (25k-75k updates), then a W192 latent probe (a tuning candidate) runs on its last checkpoint
PROBE="$(width_flags 192) $COND --blr=2e-4"
TS="--study=$STUDY --prefix=probe_tok_${TOK_SIZE}_frac1_lr --start=1e-4 --step=2 --budget=3 --target_epochs=$CAP --max_epochs=$CAND_MAX"
for k in 1 2 3; do
    lr=$(python hp_search.py next $TS) || exit 1
    case $lr in done) break ;; wait) echo "=== tokenizer lr: a probe has not finished; stopping"; exit 1 ;; esac
    tid=$(tok_id 1 $lr)
    step "$tid" python train_dito.py --data_dir=$DATA --size=$TOK_SIZE --lr=$lr --max_updates=75000 --eval_every=12500 \
        --settle_tol=0.02 --settle_target_updates=$TOK_T100 --settle_min_updates=25000 --settle_max_updates=75000 --out_dir="$(rdir "$tid")"
    need "$tid"
    ck=$(ls "$(rdir "$tid")"/ckpt_*.pt | sed 's/.*ckpt_\([0-9]*\)\.pt/\1/' | sort -n | tail -1); ck=$(rdir "$tid")/ckpt_$ck.pt
    latents "$ck"
    short probe_$tid latent 1 "$(kill_at $(python hp_search.py ids $TS))" $PROBE --vae=dito:$ck
done
TW=$(rank tok_lr $(python hp_search.py ids $TS)) || exit 1
TOK_LR=${TW##*_lr}
study_tokenizer 1 # the winning candidate resumes to T100
echo "=== tokenizer: DiTo-$TOK_SIZE at lr $TOK_LR"

# ---- 5. latent lr (D8) and weight decay (D9), W192, full data, T_full -------------------------------------------------------------------
# start: SiT / the DiTo authors' AdamW 1e-4 at batch 256, sqrt batch scaling to 1024 = 2e-4 (Muon match_rms keeps the AdamW lr)
LL1=$(lr_search latent_lr latent 1 2e-4 $LR_BUDGET tune_lat_lr - $(width_flags 192) $COND --vae=dito:$(tok_full 1)) || exit 1
LAT_LR=$(lr_of "$LL1")
LW=$(wd_search latent_wd latent 1 $LAT_LR "$LL1" tune_lat_wd $(width_flags 192) $COND --vae=dito:$(tok_full 1)) || exit 1
LAT_S=$(s_of "$LW")
extend "$LW" latent 1 latent_wd || exit 1
echo "=== latent (W192): lr $LAT_LR, weight decay s $LAT_S (wd $(wd_of $LAT_S $LAT_LR))"
recipe() { # family frac width -> its recipe: the tuned lr (pixel: S's at width >= 256), times the N/8 factor at N/8; wd from the decay timescale
    local lr; if [ "$1" = pixel ]; then lr=$([ "$3" -ge 256 ] && echo "$S_LR" || echo "$PIX_LR"); else lr=$LAT_LR; fi
    [ "$2" = 0.125 ] && lr=$(scale_lr $lr "$(cat "$D/$1_lr_factor_frac0.125.txt")")
    local s; s=$([ "$1" = pixel ] && echo "$PIX_S" || echo "$LAT_S")
    if [ "$1" = pixel ]; then echo "$PS --blr=$lr --weight_decay=$(wd_of $s $lr)"; else echo "$COND --blr=$lr --weight_decay=$(wd_of $s $lr)"; fi
}

# ---- matrix machinery: predict-and-stop cells, extended only where the pixel-vs-latent decision could depend on them ------------
GRID_SETTLE="--settle_tol=0.02 --settle_patience=2 --settle_min_epochs=20 --settle_max_epochs=$CAP"
grid_cell() { # id family frac args...: train until the predicted converged guided FID settles (or the full length); final evaluation
    local id=$1 fam=$2 frac=$3; shift 3
    final_done "$id" && return 0
    echo "$*" > "$STUDY/args/$id.txt"; echo "$fam" > "$STUDY/args/$id.fam"
    if [ "$fam" = pixel ]; then step "$id" python train_jit.py $PIX_BASE "$@" $(length_flags $frac) $GRID_SETTLE --out_dir="$(rdir "$id")"
    else step "$id" python train_latent.py $LAT_BASE "$@" $(length_flags $frac) $GRID_SETTLE --out_dir="$(rdir "$id")"; fi
    done_ok "$id" || return 1
    final_eval "$id" "$fam"
}
extend_cell() { # id frac: resume a matrix run to the full length (no stopping rule), then its final evaluation at the new end
    local id=$1 frac=$2 fam; fam=$(cat "$STUDY/args/$id.fam" 2>/dev/null || echo pixel); local dir; dir=$(rdir "$id")
    if [ "$fam" = pixel ]; then step "${id}_ext" python train_jit.py $PIX_BASE $(cat "$STUDY/args/$id.txt") $(length_flags $frac) --out_dir="$dir" --init_from=resume
    else step "${id}_ext" python train_latent.py $LAT_BASE $(cat "$STUDY/args/$id.txt") $(length_flags $frac) --out_dir="$dir" --init_from=resume; fi
    done_ok "${id}_ext" && final_eval "$id" "$fam"
}
matrix() { # frac: the cells of one data size, then up to 3 extension passes (extend_plan.py), then the manifest
    local f=$1 ws; ws=$([ "$f" = 1 ] && echo "192 256 384" || echo "192 384")
    local ids="" w sys_ fam tok extra id
    for w in $ws; do
        for sys_ in pixel early full; do
            if [ $sys_ = pixel ]; then fam=pixel; tok=-; extra=""
            else fam=latent; tok=$([ $sys_ = early ] && tok_early $f || tok_full $f); extra="--vae=dito:$tok"; fi
            id=m_${sys_}_w${w}_frac$f
            # the tuning winners are these cells (same family, width, data, recipe, tokenizer): reused, not retrained
            [ $w = 192 ] && [ $sys_ = pixel ] && [ $f = 1 ] && id=$PW
            [ $w = 384 ] && [ $sys_ = pixel ] && [ $f = 1 ] && id=$SW
            [ $w = 192 ] && [ $sys_ = full ] && [ $f = 1 ] && id=$LW
            [ $w = 192 ] && [ $sys_ = pixel ] && [ $f = 0.125 ] && id=$(cut -d' ' -f1 "$D/pixel_lr_frac0.125.txt")
            [ $w = 192 ] && [ $sys_ = full ] && [ $f = 0.125 ] && id=$(cut -d' ' -f1 "$D/latent_lr_frac0.125.txt")
            [ -e "$STUDY/args/$id.fam" ] || echo "$fam" > "$STUDY/args/$id.fam"
            if ! final_done "$id"; then grid_cell $id $fam $f $(gen_width $fam $w) $(recipe $fam $f $w) $extra || return 1; fi
            ids="$ids $id"
            echo "$id $fam $w $tok $sys_" >> "$D/.cells_frac$f.tmp"
        done
    done
    local pass ext
    for pass in 1 2 3; do
        ext=$(python extend_plan.py --study=$STUDY --frac=$f --full_epochs=$CAP --margin=${EXT_MARGIN:-0.10} $ids) || return 1
        [ -z "$ext" ] && break
        echo ">>> matrix data $f, extension pass $pass: $ext"
        for id in $ext; do extend_cell $id $f || return 1; done
    done
    sort -u "$D/.cells_frac$f.tmp" | while read -r id fam w tok sys_; do
        final_done "$id" && manifest "$id" "$([ $fam = pixel ] && echo pixel || echo latent-$sys_)" $w $f "$tok"
    done
    rm -f "$D/.cells_frac$f.tmp"
}

# ---- 6. the full-data study: matrix, frontier, frozen B forecasts, B (the hypothesis at full data, as early as possible) --------------
matrix 1 || exit 1
step study_frontier_full python frontier.py --study=$STUDY
if [ ! -e "$STUDY/predictions_w768.json" ]; then
    python frontier.py --study=$STUDY --predict_width=768 || exit 1
    echo ">>> B forecasts frozen: $STUDY/predictions_w768.json ($(cat "$STUDY/predictions_w768.json.sha256"))"
fi
sha256sum -c <(echo "$(cat "$STUDY/predictions_w768.json.sha256")  $STUDY/predictions_w768.json") >/dev/null 2>&1 \
    || { echo "=== predictions_w768.json changed after freezing: stopping"; exit 1; }
for sys_ in pixel early full; do
    if [ $sys_ = pixel ]; then fam=pixel; tok=-; extra=""; else fam=latent; tok=$([ $sys_ = early ] && tok_early 1 || tok_full 1); extra="--vae=dito:$tok"; fi
    id=b_${sys_}
    train_gen $id $fam $(gen_width $fam 768) $(length_flags 1) $(recipe $fam 1 768) $extra || continue
    fid50k $id $fam
    final_done "$id" && manifest "$id" "$([ $fam = pixel ] && echo pixel || echo latent-$sys_)" 768 1 "$tok"
done
step study_b_validation python -c "
import json, glob, os, sys, numpy as np
sys.path.insert(0, '.')
import predict_fid
study = sys.argv[1]; pred = json.load(open(study + '/predictions_w768.json'))
out = {}
for sys_, fam in (('pixel', 'pixel'), ('early', 'latent-early'), ('full', 'latent-full')):
    run = f'{study}/runs/b_{sys_}'
    if fam not in pred['families'] or not os.path.isdir(run): continue
    curve = pred['families'][fam]['curve']
    T = np.array([c['presentations'] for c in curve]); F = np.array([c['fid'] for c in curve])
    rows = [{'epochs': e, 'measured': v, 'forecast': float(np.exp(np.interp(np.log(e * predict_fid.FULL_TRAIN), np.log(T), np.log(F))))}
            for e, v in predict_fid.history(run, 'val/guided_fid')]
    f50 = glob.glob(run + '/eval_50k/eval_*.json')
    out[fam] = {'curve': rows, 'fid50k': (lambda r: min(v['fid'] for v in r['final'].values() if isinstance(v, dict)) if r.get('final') else min(r['fid'].values()))(json.load(open(f50[0]))) if f50 else None}
json.dump(out, open(study + '/b_validation.json', 'w'), indent=1)
for fam, r in out.items():
    print(f'>>> B {fam}: FID-50K {r[\"fid50k\"]} | guided curve, forecast / measured: ' + ', '.join(f\"{a['forecast']:.1f}/{a['measured']:.1f}\" for a in r['curve'][-4:]))
" "$STUDY"
step study_frontier_full_b python frontier.py --study=$STUDY
echo "=== full-data study done: $STUDY/frontier.json, frontier.png, b_validation.json"

# ---- 7. the smaller data sizes: tokenizers, N/8 lr (D10), matrices ------------------------------------------------------------------
for f in 0.5 0.25 0.125; do study_tokenizer $f; done
for fam in pixel latent; do
    base=$([ $fam = pixel ] && echo "$PIX_LR" || echo "$LAT_LR"); s=$([ $fam = pixel ] && echo "$PIX_S" || echo "$LAT_S")
    NS="--study=$STUDY --prefix=tune_${fam}_frac0.125_lr --start=$base --step=2 --budget=3 --target_epochs=$CAP --max_epochs=$CAND_MAX"
    for k in 1 2 3; do
        lr=$(python hp_search.py next $NS) || exit 1
        case $lr in done) break ;; wait) echo "=== N/8 $fam lr: a candidate has not finished; stopping"; exit 1 ;; esac
        kl=$(kill_at $(python hp_search.py ids $NS))
        if [ $fam = pixel ]; then short tune_${fam}_frac0.125_lr$lr pixel 0.125 "$kl" $(gen_width pixel 192) $PS --blr=$lr --weight_decay=$(wd_of $s $lr)
        else short tune_${fam}_frac0.125_lr$lr latent 0.125 "$kl" $(width_flags 192) $COND --blr=$lr --weight_decay=$(wd_of $s $lr) --vae=dito:$(tok_full 0.125); fi
    done
    w=$(rank ${fam}_lr_frac0.125 $(python hp_search.py ids $NS)) || exit 1
    extend "$w" $fam 0.125 ${fam}_lr_frac0.125 || exit 1
    echo "$fam" > "$STUDY/args/$w.fam"
    echo "$(python -c "print(float('$(lr_of "$w")') / float('$base'))")" > "$D/${fam}_lr_factor_frac0.125.txt"
done
for f in 0.5 0.25 0.125; do matrix $f || exit 1; done
step study_frontier_all python frontier.py --study=$STUDY

# ---- 8. seeds for the most consequential comparison -----------------------------------------------------------------------------
if [ "$RUN_SEEDS" = 1 ]; then
    need b_pixel_50k b_early_50k b_full_50k
    fe=$(eval_fid "$(rdir b_early)/eval_50k"); ff=$(eval_fid "$(rdir b_full)/eval_50k")
    LB=$(lt "$fe" "$ff" && echo early || echo full)
    echo ">>> seeds: pixel B vs latent B ($LB tokenizer)"
    for seed in 1 2; do
        train_gen b_pixel_seed$seed pixel $(gen_width pixel 768) $(length_flags 1) $(recipe pixel 1 768) --seed=$seed && fid50k b_pixel_seed$seed pixel
        # the tokenizer retrained end to end with this seed, to the same number of updates as the study's checkpoint
        tck=$([ $LB = early ] && tok_early 1 || tok_full 1); tit=$(echo "$tck" | sed 's/.*ckpt_\([0-9]*\)\.pt/\1/')
        tid=$(tok_id 1 $TOK_LR)_seed$seed
        step "$tid" python train_dito.py --data_dir=$DATA --size=$TOK_SIZE --lr=$TOK_LR --max_updates=$tit --eval_every=12500 \
            --seed=$seed --out_dir="$(rdir "$tid")"
        done_ok "$tid" || continue
        sck=$(rdir "$tid")/ckpt_$tit.pt
        latents "$sck"
        train_gen b_${LB}_seed$seed latent $(width_flags 768) $(length_flags 1) $(recipe latent 1 768) --vae=dito:$sck --seed=$seed && fid50k b_${LB}_seed$seed latent
    done
    python -c "
import json, glob, sys, numpy as np
study, lb = sys.argv[1], sys.argv[2]
def f50(run):
    fs = glob.glob(f'{study}/runs/{run}/eval_50k/eval_*.json')
    if not fs: return None
    r = json.load(open(fs[0])); fin = [v['fid'] for v in r.get('final', {}).values() if isinstance(v, dict)]
    return min(fin) if fin else min(r['fid'].values())
for name, runs in (('pixel B', ['b_pixel', 'b_pixel_seed1', 'b_pixel_seed2']), (f'latent B ({lb})', [f'b_{lb}', f'b_{lb}_seed1', f'b_{lb}_seed2'])):
    v = [x for x in (f50(r) for r in runs) if x is not None]
    print(f'>>> {name}: FID-50K ' + ', '.join(f'{x:.2f}' for x in v) + (f'  mean {np.mean(v):.2f} +- {np.std(v, ddof=1):.2f}' if len(v) > 1 else ''))
" "$STUDY" "$LB" | tee "$D/seeds_summary.txt"
fi

# ---- 9. L (depth transfer) ------------------------------------------------------------------------------------------------------
if [ "$RUN_L" = 1 ]; then
    LFLAGS="--depth=24 --hidden_size=1024 --num_heads=16"
    train_gen l_pixel pixel --model=JiT-L/16 --bottleneck_dim=$(bn_of 1024) --batch_size=512 --gradient_accumulation_steps=2 --compile_mode=blocks $(length_flags 1) $(recipe pixel 1 1024) && fid50k l_pixel pixel
    tok=$(tok_full 1)
    train_gen l_full latent $LFLAGS --gradient_accumulation_steps=2 $(length_flags 1) $(recipe latent 1 1024) --vae=dito:$tok && fid50k l_full latent
fi
echo "=== $(date '+%F %T') study done (or waiting on the steps above); results: $STUDY/frontier.json, frontier.png, ledger.csv, b_validation.json"
