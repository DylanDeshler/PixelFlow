#!/usr/bin/env bash
# The pixels-vs-DiTo-latents allocation study (plan of 2026-11-26), end to end, resumable: rerun this script after any
# interruption and it skips finished steps, resumes interrupted training from its last checkpoint and re-reads every
# decision it already made (runs/study/decisions/*.txt; delete one to redo that decision and what follows from it).
# Question: at a given data size and TOTAL TRAINING compute (tokenizer training + dataset encoding + generator training;
# inference is not part of this study), are pixels or DiTo latents the better conditional generator, and what allocation
# (generator size, training length, tokenizer investment) is best? Predicted from widths up to S, validated at B.
#
# Fixed: 128px center crops + flips; generators batch 1024, 500-update warm-up then constant lr, Muon (match_rms_adamw) +
# AdamW side groups; pixel generator JiT /16 with clean-image prediction; latent generator the same blocks on DiTo f8c4
# latents (4 x 16 x 16, patch 2); DiTo trained as the paper (dito_model.py: its loss, LayerNorm latent, noise sync, 50-step
# Euler decoding) at batch 64 with Muon from the paper's AdamW lr 1e-4 (match_rms_adamw: the AdamW lr carries over).
# Widths W192 / W256 / S (12 x 192 / 256 / 384, heads of 64); B (12 x 768) held out. Data N, N/2, N/4, N/8 (nested,
# class-stratified: subsets.py).
# Run length: generators train for 100 full-data epochs (subsets: the same number of updates), stopping earlier only if their
# own curve flattens (jit_common.converged: < 2% improvement over the last third of training at 2 consecutive evals);
# tokenizers likewise by rFID, capped at the paper's 300k updates. frontier.py treats budgets past a run's end as extrapolated. Generator evals every 5 full-data epochs: unguided
# FID-10k, a kept checkpoint and post-hoc EMA snapshots (sigma_rel 0.02 / 0.05 / 0.10, bfloat16) each.
# Evaluation policy (one throughout): guided FID with post-hoc EMA and a fixed budget of 6 guidance scales (uniform 1/3/5
# then bisection; CFG at all t), searched with 2K samples, the pick re-scored at FID-10K on fresh samples:
#   anchors  each run's kept checkpoints nearest 1/16, 1/8, 1/4, 1/2, 1 of its length, sigma_rel 0.10 (the fitted curves)
#   joint    the final checkpoint: EMA x CFG searched jointly (each sigma_rel's 6-scale search at 2K, the best pair re-scored
#            at 10K fresh)
#   decisive fresh FID-50K (seed 2) for B and the seed replicates
# Stages (each resumable; later stages read earlier decisions):
#   0  subset manifests; a preflight of every new code path (a few minutes each; PREFLIGHT=0 skips it)
#   1  conditioning: queues/conditioning.sh (JiT in-context tokens vs the KV-prefix class table, W192, each arm's lr / noise
#      / aux search, confirmations); the winner by guided FID at matched compute (the cheaper arm when within 2%) carries
#      into every generator, pixel and latent
#   1b pixel bottleneck rule: fixed 128 vs proportional (width / 6) at S to convergence (earlier 20-epoch evidence favours
#      proportional: 77.85 vs 82.69); the winner applies to every pixel width
#   2  tokenizers (DiTo-$TOK_SIZE): full data at lr 5e-5 / 1e-4 / 2e-4, each judged by a downstream probe (a W192 latent
#      generator to convergence on its latents, guided FID); N/8 checks the neighbours of the chosen lr the same way; N/2
#      and N/4 use the full-data lr. Every tokenizer trains to rFID convergence; its kept checkpoints are the investment
#      axis: T_full (converged) and T_early (the checkpoint nearest a quarter of that). Latents (both flips) and rFID for each
#   3  latent recipe (W192, full data, T_full): Muon lr 1e-4 / 2e-4 / 4e-4, then clean-latent (x) vs velocity prediction,
#      then JiT-style vs uniform t, then the aux lr x0.5 / x2, each a decision on the previous best
#   4  data dependence (W192, N/4 and N/8, per family): the full-data recipe vs lr x0.5, x2 and dropout 0.1, all to
#      convergence; N/2 uses the full-data recipe. A pilot of the latent family with the other conditioning arm is recorded
#   5  the discovery matrix: N x {W192, W256, S}, N/2, N/4, N/8 x {W192, S}; each cell pixel, latent T_early, latent T_full
#      (27 trajectories; identical earlier runs are reused); anchors + joint evaluation each; a run manifest for frontier.py
#   6  frontiers, backtests and the cost ledger (frontier.py); then the B forecasts frozen (predictions_w768.json + sha256,
#      never overwritten)
#   7  B on full data: pixel, latent T_early, latent T_full (existing full-data tokenizers), anchors + joint + FID-50K;
#      forecast vs measured written to b_validation.json
#   8  seeds (RUN_SEEDS=1): the most consequential comparison (pixel B vs the better latent B) with 2 more end-to-end seeds
#      each, the tokenizer retrained for every latent seed; FID-50K
#   9  L on full data (RUN_L=1): pixel and the better latent family, a depth-transfer test
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
CAP=100 # full-data epochs (subsets: the same number of updates): the run length unless the curve flattens first
TOK_CAP=300000 # tokenizer updates: the paper's schedule, as a cap
EVAL="--phema_stds=0.02,0.05,0.10 --phema_dtype=bfloat16 --fid_num_images=10000 --converge_tol=0.02 --converge_patience=2 --warmup_updates=500"
PIX_BASE="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA $EVAL --diag=False --trend_plots=False --progress_gif=False" # (the diagnostic suite does not feed the study)
LAT_BASE="--data_dir=$DATA --recipe=jit --patch_size=2 $EVAL"
width_flags() { echo "--hidden_size=$1 --num_heads=$(( $1 / 64 ))"; }
length_flags() { # data fraction -> epochs cap, eval spacing and minimum length, all in full-data epochs
    python -c "f=float('$1'); print(f'--epochs={int(round($CAP / f))} --eval_every_epochs={int(round(5 / f))} --converge_min_epochs={int(round(20 / f))}' + ('' if f >= 1 else f' --train_frac={f:g}'))"
}

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
anchors() { # run dir -> the kept checkpoints' iterations nearest 1/16 .. 1 of its length (unique)
    python -c "
import glob, re, sys
its = sorted(int(re.search(r'ckpt_(\d+)\.pt$', f).group(1)) for f in glob.glob(sys.argv[1] + '/ckpt_*.pt'))
if not its: sys.exit()
end = its[-1]
print(' '.join(str(i) for i in sorted({min(its, key=lambda i: abs(i - f * end)) for f in (1/16, 1/8, 1/4, 1/2, 1)})))" "$1"
}
decide() { # name what id... : the run whose joint evaluation has the lowest FID; stored as "id fid"
    local f=$D/$1.txt what=$2; shift 2
    if [ -e "$f" ]; then cat "$f"; return; fi
    local best="" bf="" id v
    for id in "$@"; do
        v=$(eval_fid "$(rdir "$id")/eval_joint")
        [ -n "$v" ] || { echo "=== $what: $id has no joint evaluation yet; stopping" >&2; exit 1; }
        echo "    $id: FID-10K $v" >&2
        if [ -z "$bf" ] || lt "$v" "$bf"; then best=$id; bf=$v; fi
    done
    echo "$best $bf" | tee "$f"
    echo ">>> $what: $best (FID-10K $bf)" >&2
}

# generator runs: a run = id + family + args file; the args rebuild the network for its evaluations
train_gen() { # id family(pixel|latent) args...: train to convergence, then anchors + joint evaluation
    local id=$1 fam=$2; shift 2
    local dir; dir=$(rdir "$id")
    echo "$*" > "$STUDY/args/$id.txt"
    if [ "$fam" = pixel ]; then step "$id" python train_jit.py $PIX_BASE "$@" --out_dir="$dir"
    else step "$id" python train_latent.py $LAT_BASE "$@" --out_dir="$dir"; fi
    done_ok "$id" || return 1
    evaluate_run "$id" "$fam"
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
evaluate_run() { # id family: anchors (sigma 0.10) and the joint EMA x CFG search at the final checkpoint
    local id=$1 fam=$2 dir it s
    dir=$(rdir "$id")
    for it in $(anchors "$dir"); do
        eval_one "${id}_a$it" "$id" "$fam" "$it" 0.10 2000 10000 "$dir/eval_anchor_$it" "--cfg_search=6"
    done
    it=$(final_iter "$dir")
    local best="" bf="" bs=""
    for s in 0.02 0.05 0.10; do
        eval_one "${id}_j$s" "$id" "$fam" "$it" "$s" 2000 2000 "$dir/eval_joint_s$s" "--cfg_search=6"
        local v; v=$(eval_fid "$dir/eval_joint_s$s")
        [ -n "$v" ] || return 1
        if [ -z "$bf" ] || lt "$v" "$bf"; then bf=$v; bs=$s; fi
    done
    local c; c=$(eval_cfg "$dir/eval_joint_s$bs")
    if [ "$fam" = pixel ]; then eval_one "${id}_joint" "$id" pixel "$it" "$bs" 10000 10000 "$dir/eval_joint" "--cfgs=$c --seed=1"
    else eval_one "${id}_joint" "$id" latent "$it" "$bs" 10000 10000 "$dir/eval_joint" "--eval_cfgs=$c --seed=1"; fi
    echo "$bs $c" > "$dir/joint_choice.txt"
}
fid50k() { # id family: fresh FID-50K at the joint choice (seed 2)
    local id=$1 fam=$2 dir; dir=$(rdir "$id")
    read -r s c < "$dir/joint_choice.txt"
    local it; it=$(final_iter "$dir")
    if [ "$fam" = pixel ]; then eval_one "${id}_50k" "$id" pixel "$it" "$s" 50000 50000 "$dir/eval_50k" "--cfgs=$c --seed=2"
    else eval_one "${id}_50k" "$id" latent "$it" "$s" 50000 50000 "$dir/eval_50k" "--eval_cfgs=$c --seed=2"; fi
}
manifest() { # id family width frac tokenizer-ckpt|-: one line per discovery / validation run (frontier.py)
    local m=$STUDY/manifest.jsonl
    grep -q "\"id\": \"$1\"" "$m" 2>/dev/null && return
    python -c "
import json, sys, torch
id_, fam, w, frac, tok = sys.argv[1:6]
rec = {'id': id_, 'family': fam, 'width': int(w), 'data_frac': float(frac), 'run': sys.argv[6], 'tokenizer': None if tok == '-' else tok,
       'tokenizer_pflops': 0.0, 'encode_pflops': 0.0}
if tok != '-':
    ck = torch.load(tok, map_location='cpu', weights_only=False)
    rec['tokenizer_pflops'] = ck['train_pflops']
    cfg = json.load(open(sys.argv[6] + '/config.json'))['resolved']
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

# ---- 1. conditioning --------------------------------------------------------------------------------------------------------
bash queues/conditioning.sh || { echo "=== conditioning stage not finished: stopping here; rerun to continue"; exit 1; }
if [ ! -e "$D/conditioning.txt" ]; then
    bi=$(cut -d' ' -f1 runs/conditioning/incontext_best.txt); bk=$(cut -d' ' -f1 runs/conditioning/kvtable_best.txt)
    run_of() { grep -m1 '^run:  ' "runs/queue_logs/$1.log" | sed 's/^run:  //'; }
    fi_=$(eval_fid "$(run_of "$bi")/eval_final"); fk=$(eval_fid "$(run_of "$bk")/eval_final")
    # the matched-compute pair: the longer arm's checkpoint at the shorter arm's final compute vs the shorter arm at convergence
    long=$(ls runs/queue_logs/*_matched_eval.done 2>/dev/null | head -1 | sed 's|runs/queue_logs/||; s|_matched_eval.done||')
    fm=$(eval_fid "$(run_of "$long")/eval_matched")
    if [ "$long" = "$bi" ]; then a_long=incontext; f_long=$fm; f_short=$fk; short=kvtable; else a_long=kvtable; f_long=$fm; f_short=$fi_; short=incontext; fi
    # winner at matched compute; within 2% -> the cheaper (shorter-compute) arm
    if python -c "import sys; a, b = float('$f_long'), float('$f_short'); sys.exit(0 if a < b * 0.98 else 1)"; then arm=$a_long; else arm=$short; fi
    s=$([ $arm = incontext ] && echo "$bi" || echo "$bk")
    echo "$arm $s" > "$D/conditioning.txt"
    echo ">>> conditioning: $arm ($s); matched compute: $a_long $f_long vs $short $f_short; at convergence incontext $fi_, kvtable $fk"
fi
read -r ARM COND_STEP < "$D/conditioning.txt"
arm_flags() { case $1 in incontext) echo "--in_context_len=32" ;; kvtable) echo "--kv_prefix_len=32 --kv_prefix_class_tokens=True" ;; esac; }
lr_flags_of() { # conditioning step name -> --blr --noise_scale --aux_lr_mult
    local n=$1 aux=1.0
    case $n in *_aux*) aux=${n##*_aux}; n=${n%_aux*} ;; esac
    echo "--blr=$(echo "$n" | sed 's/.*_blr\([^_]*\)_noise.*/\1/') --noise_scale=${n##*_noise} --aux_lr_mult=$aux"
}
COND=$(arm_flags "$ARM")
OTHER=$([ "$ARM" = incontext ] && echo kvtable || echo incontext)
PIX_RECIPE="$COND $(lr_flags_of "$COND_STEP")" # the pixel recipe for full data (W192 search of stage 1)
echo "=== conditioning: $ARM; pixel recipe: $PIX_RECIPE"

# ---- 1b. pixel bottleneck rule: JiT's fixed 128 vs proportional to width (width / 6, JiT-B's 768 / 128) ---------------------------
# Earlier evidence (scaling.sh, row-8 stack, S, 20 epochs): proportional 64 FID-10k 77.85 vs fixed 128 82.69. Confirmed here with
# the study's recipe at convergence, at S (the largest discovery width); the winning rule applies to every pixel width.
train_gen bn_S_fixed pixel $(width_flags 384) --bottleneck_dim=128 $(length_flags 1) $PIX_RECIPE
train_gen bn_S_prop pixel $(width_flags 384) --bottleneck_dim=64 $(length_flags 1) $PIX_RECIPE
r=$(decide bottleneck "pixel bottleneck at S (fixed 128 vs proportional 64)" bn_S_prop bn_S_fixed) || exit 1
BN_RULE=$([ "${r%% *}" = bn_S_prop ] && echo proportional || echo fixed)
bn_of() { if [ "$BN_RULE" = proportional ]; then echo $(( ($1 + 3) / 6 )); else echo 128; fi; } # width / 6, rounded
gen_width() { # family width -> width flags (+ the pixel bottleneck by the rule)
    if [ "$1" = pixel ]; then echo "$(width_flags $2) --bottleneck_dim=$(bn_of $2)"; else width_flags $2; fi
}
echo "=== pixel bottleneck rule: $BN_RULE (W192 $(bn_of 192), W256 $(bn_of 256), S $(bn_of 384), B $(bn_of 768))"

# ---- 2. tokenizers -----------------------------------------------------------------------------------------------------------
tok_id() { echo "tok_${TOK_SIZE}_frac$1_lr$2"; } # data fraction, lr
train_tok() { # frac lr [seed]
    local id; id=$(tok_id "$1" "$2")${3:+_seed$3}
    step "$id" python train_dito.py --data_dir=$DATA --size=$TOK_SIZE --lr=$2 --max_updates=$TOK_CAP --eval_every=10000 \
        --converge_tol=0.02 --converge_min_updates=30000 $( [ "$1" = 1 ] || echo "--train_frac=$1" ) ${3:+--seed=$3} --out_dir="$(rdir "$id")"
}
tok_ckpts() { # tokenizer id -> "T_early_ckpt T_full_ckpt"
    python -c "
import glob, re, sys
its = sorted(int(re.search(r'ckpt_(\d+)\.pt$', f).group(1)) for f in glob.glob(sys.argv[1] + '/ckpt_*.pt'))
end = its[-1]; early = min(its, key=lambda i: abs(i - end / 4))
print(f'{sys.argv[1]}/ckpt_{early}.pt {sys.argv[1]}/ckpt_{end}.pt')" "$(rdir "$1")"
}
latents() { # tokenizer checkpoint: encode the dataset (both flips) and its rFID
    local ck=$1 tag; tag=$(echo "$ck" | sed 's|.*/runs/||; s|/ckpt_|_|; s|\.pt$||; s|/|_|g')
    step "lat_$tag" python prepare_latents.py --data_dir=$DATA --vae=dito:$ck
    step "rfid_$tag" python latent_rfid.py --data_dir=$DATA --vae=dito:$ck
}
LAT_PROBE_RECIPE="$COND --blr=2e-4" # the latent generator before stage 3's tuning
probe() { # tokenizer id frac: a W192 latent generator on its converged checkpoint, to convergence (the downstream judgement)
    local tid=$1 frac=$2 full; full=$(tok_ckpts "$tid" | cut -d' ' -f2)
    latents "$full"
    train_gen "probe_$tid" latent $(width_flags 192) $(length_flags "$frac") $LAT_PROBE_RECIPE --vae=dito:$full
}
# 2a. full data: lr search judged by probes
for lr in 5e-5 1e-4 2e-4; do train_tok 1 $lr; done
need $(tok_id 1 5e-5) $(tok_id 1 1e-4) $(tok_id 1 2e-4)
for lr in 5e-5 1e-4 2e-4; do probe "$(tok_id 1 $lr)" 1; done
r=$(decide tok_lr_full "tokenizer lr, full data (probe guided FID)" probe_$(tok_id 1 5e-5) probe_$(tok_id 1 1e-4) probe_$(tok_id 1 2e-4)) || exit 1
TOK_LR=$(echo "$r" | cut -d' ' -f1 | sed 's/.*_lr//')
# 2b. N/8: the chosen lr and its neighbours
lo=$(python -c "print(f'{float(\"$TOK_LR\") / 2:g}')"); hi=$(python -c "print(f'{float(\"$TOK_LR\") * 2:g}')")
for lr in $lo $TOK_LR $hi; do train_tok 0.125 $lr; done
need $(tok_id 0.125 $lo) $(tok_id 0.125 $TOK_LR) $(tok_id 0.125 $hi)
for lr in $lo $TOK_LR $hi; do probe "$(tok_id 0.125 $lr)" 0.125; done
r=$(decide tok_lr_frac0.125 "tokenizer lr, N/8 (probe guided FID)" probe_$(tok_id 0.125 $lo) probe_$(tok_id 0.125 $TOK_LR) probe_$(tok_id 0.125 $hi)) || exit 1
TOK_LR_8=$(echo "$r" | cut -d' ' -f1 | sed 's/.*_lr//')
# 2c. N/2, N/4 at the full-data lr; the study's tokenizer per data size
for f in 0.5 0.25; do train_tok $f $TOK_LR; done
need $(tok_id 0.5 $TOK_LR) $(tok_id 0.25 $TOK_LR)
tok_of() { case $1 in 0.125) tok_id 0.125 "$TOK_LR_8" ;; *) tok_id "$1" "$TOK_LR" ;; esac; }
for f in $FRACS; do
    read -r early full <<< "$(tok_ckpts "$(tok_of $f)")"
    latents "$early"; latents "$full"
    echo "$early" > "$D/tok_early_frac$f.txt"; echo "$full" > "$D/tok_full_frac$f.txt"
done
tok_early() { cat "$D/tok_early_frac$1.txt"; }
tok_full() { cat "$D/tok_full_frac$1.txt"; }

# ---- 3. latent recipe (W192, full data, T_full) --------------------------------------------------------------------------------
W192=$(width_flags 192); TF1=$(tok_full 1)
LAT_COMMON="$COND --vae=dito:$TF1 $(length_flags 1)"
ids=""
for lr in 1e-4 2e-4 4e-4; do
    if [ "$lr" = 2e-4 ]; then id=probe_$(tok_id 1 "$TOK_LR") # the chosen tokenizer's probe is exactly this run (same latents, recipe, length)
    else id=rec_lat_lr$lr; train_gen $id latent $W192 $LAT_COMMON --blr=$lr; fi
    ids="$ids $id"
done
r=$(decide rec_lat_lr "latent lr" $ids) || exit 1; best=${r%% *}
LAT_LR=$(grep -o -- '--blr=[^ ]*' "$STUDY/args/$best.txt" | tail -1)
train_gen rec_lat_x latent $W192 $LAT_COMMON $LAT_LR --prediction=x
r=$(decide rec_lat_pred "latent prediction (v vs x)" "$best" rec_lat_x) || exit 1; best_pred=${r%% *}
PRED=$([ "$best_pred" = rec_lat_x ] && echo "--prediction=x" || echo "")
train_gen rec_lat_tuni latent $W192 $LAT_COMMON $LAT_LR $PRED --t_sampling=uniform
r=$(decide rec_lat_t "latent t sampling (logit-normal vs uniform)" "$best_pred" rec_lat_tuni) || exit 1; best_t=${r%% *}
TS=$([ "$best_t" = rec_lat_tuni ] && echo "--t_sampling=uniform" || echo "")
for m in 0.5 2; do train_gen rec_lat_aux$m latent $W192 $LAT_COMMON $LAT_LR $PRED $TS --aux_lr_mult=$m; done
r=$(decide rec_lat_aux "latent aux lr" "$best_t" rec_lat_aux0.5 rec_lat_aux2) || exit 1; best_aux=${r%% *}
AUX=$(grep -o -- '--aux_lr_mult=[^ ]*' "$STUDY/args/$best_aux.txt" | tail -1)
LAT_RECIPE="$COND $LAT_LR $PRED $TS $AUX"
echo "$LAT_RECIPE" > "$D/latent_recipe_full.txt"
echo "=== latent recipe (full data): $LAT_RECIPE"
# the latent pilot with the other conditioning arm (recorded, not switched)
train_gen pilot_lat_$OTHER latent $W192 $(arm_flags "$OTHER") --vae=dito:$TF1 $(length_flags 1) $LAT_LR $PRED $TS $AUX
echo ">>> latent pilot: $ARM $(eval_fid "$(rdir "$best_aux")/eval_joint") vs $OTHER $(eval_fid "$(rdir pilot_lat_$OTHER)/eval_joint")" | tee "$D/latent_pilot.txt"

# ---- 4. data dependence (W192, N/4 and N/8) ------------------------------------------------------------------------------------
scale_lr() { # recipe factor -> the recipe with --blr scaled
    echo "$1" | python -c "
import re, sys
r = sys.stdin.read().strip(); f = float(sys.argv[1])
print(re.sub(r'--blr=([^ ]+)', lambda m: f'--blr={float(m.group(1)) * f:g}', r))" "$2"
}
for f in 0.25 0.125; do
    for fam in pixel latent; do
        base=$([ $fam = pixel ] && echo "$PIX_RECIPE" || echo "$COND $LAT_LR $PRED $TS $AUX --vae=dito:$(tok_full $f)")
        drop=$([ $fam = pixel ] && echo "--attn_dropout=0.1 --proj_dropout=0.1" || echo "--dropout=0.1")
        ids=""
        for v in base lrhalf lrdouble drop; do
            case $v in base) r="$base" ;; lrhalf) r=$(scale_lr "$base" 0.5) ;; lrdouble) r=$(scale_lr "$base" 2) ;; drop) r="$base $drop" ;; esac
            id=dd_${fam}_frac${f}_$v
            train_gen $id $fam $(gen_width $fam 192) $(length_flags $f) $r
            ids="$ids $id"
        done
        r=$(decide dd_${fam}_frac$f "$fam recipe at data $f" $ids) || exit 1; best=${r%% *}
        sed "s|--vae=dito:[^ ]*||" "$STUDY/args/$best.txt" | python -c "
import re, sys
a = sys.stdin.read()
keep = re.findall(r'--(?:in_context_len|kv_prefix_len|kv_prefix_class_tokens|blr|noise_scale|aux_lr_mult|prediction|t_sampling|attn_dropout|proj_dropout|dropout)=[^ ]+', a)
print(' '.join(keep))" > "$D/recipe_${fam}_frac$f.txt"
    done
done
recipe() { # family frac -> its recipe flags (N and N/2: the full-data recipe)
    if [ -e "$D/recipe_$1_frac$2.txt" ]; then cat "$D/recipe_$1_frac$2.txt"
    elif [ "$1" = pixel ]; then echo "$PIX_RECIPE"; else echo "$LAT_RECIPE"; fi
}

# ---- 5. discovery matrix -----------------------------------------------------------------------------------------------------
widths_of() { [ "$1" = 1 ] && echo "192 256 384" || echo "192 384"; }
for f in $FRACS; do # full data first (its width ladder), then the smaller data sizes
    for w in $(widths_of $f); do
        for sys_ in pixel early full; do
            if [ $sys_ = pixel ]; then fam=pixel; tok=-; extra=""
            else fam=latent; tok=$([ $sys_ = early ] && tok_early $f || tok_full $f); extra="--vae=dito:$tok"; fi
            id=m_${sys_}_w${w}_frac$f
            # identical earlier runs are reused (same family, width, data, recipe and tokenizer): the latent W192 full-data T_full
            # cell is stage 3's winner; the W192 N/4 and N/8 pixel / T_full cells are stage 4's winners
            if [ $w = 192 ] && [ $sys_ = full ] && [ $f = 1 ]; then id=$(cut -d' ' -f1 "$D/rec_lat_aux.txt"); fi
            if [ $w = 192 ] && [ $sys_ != early ] && { [ $f = 0.25 ] || [ $f = 0.125 ]; }; then id=$(cut -d' ' -f1 "$D/dd_${fam}_frac$f.txt"); fi
            if [ $w = 384 ] && [ $sys_ = pixel ] && [ $f = 1 ]; then id=$(cut -d' ' -f1 "$D/bottleneck.txt"); fi # stage 1b's winner
            if done_ok "${id}_joint"; then :
            else train_gen $id $fam $(gen_width $fam $w) $(length_flags $f) $(recipe $fam $f) $extra || continue; fi
            done_ok "${id}_joint" && manifest "$id" "$([ $fam = pixel ] && echo pixel || echo latent-$sys_)" $w $f "$tok"
        done
    done
done

# ---- 6. frontiers, backtests, ledger; B forecasts frozen ---------------------------------------------------------------------
step study_frontier_discovery python frontier.py --study=$STUDY
if [ ! -e "$STUDY/predictions_w768.json" ]; then
    python frontier.py --study=$STUDY --predict_width=768 || exit 1
    echo ">>> B forecasts frozen: $STUDY/predictions_w768.json ($(cat "$STUDY/predictions_w768.json.sha256"))"
fi
sha256sum -c <(echo "$(cat "$STUDY/predictions_w768.json.sha256")  $STUDY/predictions_w768.json") >/dev/null 2>&1 \
    || { echo "=== predictions_w768.json changed after freezing: stopping"; exit 1; }

# ---- 7. B on full data ------------------------------------------------------------------------------------------------------
for sys_ in pixel early full; do
    if [ $sys_ = pixel ]; then fam=pixel; tok=-; extra=""; else fam=latent; tok=$([ $sys_ = early ] && tok_early 1 || tok_full 1); extra="--vae=dito:$tok"; fi
    id=b_${sys_}
    train_gen $id $fam $(gen_width $fam 768) $(length_flags 1) $(recipe $fam 1) $extra || continue
    fid50k $id $fam
    done_ok "${id}_joint" && manifest "$id" "$([ $fam = pixel ] && echo pixel || echo latent-$sys_)" 768 1 "$tok"
done
step study_b_validation python -c "
import json, glob, os, sys, numpy as np
study = sys.argv[1]; pred = json.load(open(study + '/predictions_w768.json'))
out = {}
for sys_, fam in (('pixel', 'pixel'), ('early', 'latent-early'), ('full', 'latent-full')):
    run = f'{study}/runs/b_{sys_}'
    if fam not in pred['families'] or not os.path.isdir(run): continue
    cfg = json.load(open(run + '/config.json'))['resolved']
    curve = pred['families'][fam]['curve']
    T = np.array([c['presentations'] for c in curve]); F = np.array([c['fid'] for c in curve])
    rows = []
    for d in sorted(glob.glob(run + '/eval_anchor_*')):
        it = int(d.rsplit('_', 1)[1]); fs = glob.glob(d + '/eval_*.json')
        if not fs: continue
        r = json.load(open(fs[0])); fin = [v['fid'] for v in r.get('final', {}).values() if isinstance(v, dict)]
        meas = min(fin) if fin else min(r['fid'].values())
        t = it * cfg['effective_batch_size']
        rows.append({'iter': it, 'presentations': t, 'measured': meas, 'forecast': float(np.exp(np.interp(np.log(t), np.log(T), np.log(F))))})
    f50 = glob.glob(run + '/eval_50k/eval_*.json')
    out[fam] = {'anchors': rows, 'fid50k': (lambda r: min(v['fid'] for v in r['final'].values() if isinstance(v, dict)) if r.get('final') else min(r['fid'].values()))(json.load(open(f50[0]))) if f50 else None}
json.dump(out, open(study + '/b_validation.json', 'w'), indent=1)
for fam, r in out.items():
    print(fam, 'FID-50K', r['fid50k'], '| anchors (forecast vs measured):', ', '.join(f\"{a['forecast']:.1f}/{a['measured']:.1f}\" for a in r['anchors']))
" "$STUDY"
step study_frontier_b python frontier.py --study=$STUDY

# ---- 8. seeds for the most consequential comparison -----------------------------------------------------------------------------
if [ "$RUN_SEEDS" = 1 ]; then
    need b_pixel_50k b_early_50k b_full_50k
    fe=$(eval_fid "$(rdir b_early)/eval_50k"); ff=$(eval_fid "$(rdir b_full)/eval_50k")
    LB=$(lt "$fe" "$ff" && echo early || echo full)
    echo ">>> seeds: pixel B vs latent B ($LB tokenizer)"
    for seed in 1 2; do
        train_gen b_pixel_seed$seed pixel $(gen_width pixel 768) $(length_flags 1) $(recipe pixel 1) --seed=$seed && fid50k b_pixel_seed$seed pixel
        # the tokenizer retrained end to end with this seed, to the same number of updates as the study's checkpoint
        tck=$([ $LB = early ] && tok_early 1 || tok_full 1); tit=$(echo "$tck" | sed 's/.*ckpt_\([0-9]*\)\.pt/\1/')
        tid=$(tok_of 1)_seed$seed
        step "$tid" python train_dito.py --data_dir=$DATA --size=$TOK_SIZE --lr=$TOK_LR --max_updates=$tit --eval_every=10000 \
            --seed=$seed --out_dir="$(rdir "$tid")"
        done_ok "$tid" || continue
        sck=$(rdir "$tid")/ckpt_$tit.pt
        latents "$sck"
        train_gen b_${LB}_seed$seed latent $(width_flags 768) $(length_flags 1) $(recipe latent 1) --vae=dito:$sck --seed=$seed && fid50k b_${LB}_seed$seed latent
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
    train_gen l_pixel pixel --model=JiT-L/16 --bottleneck_dim=$(bn_of 1024) --batch_size=512 --gradient_accumulation_steps=2 --compile_mode=blocks $(length_flags 1) $(recipe pixel 1) && fid50k l_pixel pixel
    tok=$(tok_full 1)
    train_gen l_full latent $LFLAGS --gradient_accumulation_steps=2 $(length_flags 1) $(recipe latent 1) --vae=dito:$tok && fid50k l_full latent
fi
echo "=== $(date '+%F %T') study done (or waiting on the steps above); results: $STUDY/frontier.json, frontier.png, ledger.csv, b_validation.json"
