#!/usr/bin/env bash
# Scaling plan (2026-11-25): does end-to-end pixel diffusion scale better than latent diffusion? At 128px, model sizes S / B / L,
# each pipeline with its better recipe (decided at S), every run long enough to fit its FID curve (fit_fid.py) and project it.
#   pixel   JiT /16 on pixels (train_jit.py)
#   latent  the same blocks on MAR's ImageNet-only KL-16 latents, 16 x 8 x 8, patch 1 (train_latent.py), the stronger and
#           in-domain VAE (SD-VAE was 18 FID worse unguided; rFID 8.17 vs 8.99)
#   recipes plain  JiT's recipe (Muon, 5-epoch warm-up, clip 1, no loss weighting; the jit recipe in train_latent.py)
#           stack  row 8: class-table KV prefix + head gate + full AttnRes + adaLN-single, warm-up 1, learned weight, no clip
# Both: 64 tokens, the same trunk FLOPs per image, evals every 5 epochs (FID-10k + FD-DINOv2), milestone checkpoints every
# 20 epochs, constant lr after warm-up (so each run is one curve). Decisions use FID-10k (unguided) at 20 epochs and are
# written to runs/scaling/*.txt (delete one to re-decide).
#   0. decisions at S (width 384, 6 heads, depth 12; ~1/4 of B's FLOPs), 20 epochs each:
#      a. pixel bottleneck, on the stack: JiT's 128 vs the proportional width / 6 (JiT-B's 768 / 128) = 64 -> bottleneck.txt
#         (L then uses 128, or 1024 / 6 = 170). Latent patches are 16-d, so the embedding doesn't compress there
#      b. pixel recipe: plain (at a's bottleneck) vs a's winner -> pixel_recipe.txt
#      c. latent recipe: plain vs stack -> latent_recipe.txt
#   1. B, 120 epochs, each pipeline's recipe. Pixel stack = the existing class-table run (120 epochs, no FD-DINOv2: it predates
#      it), so only its guided sweep runs; pixel plain would be trained. Latent B is trained
#   2. S: each pipeline's winning 20-epoch run resumed to 80 epochs (nothing re-run)
#   3. L (width 1024, 16 heads, depth 24, prefix from block 8 as JiT-L), 40 epochs, 512 x 2 accumulation: ~3.5x B's FLOPs,
#      roughly 1-1.5 days each; check the first log lines for img/s and memory
#   after each final run: a guided sweep of its last checkpoint, cfg 2 / 3 / 4 / 5 (a uniform grid; FID-10k, CFG at all t)
#   at the end: fit_fid.py over all final runs -> runs/scaling/fid_projection.{png,json}
# Shares the queues' GPU lock; resumes via .done markers and checkpoints. Skip sizes with SIZES (default "S B L"; the S
# decisions always run first, everything else needs them).
#
# Every step's output goes to its own log (runs/queue_logs/<step>.log) and to the screen. Run it inside tmux so it survives a
# disconnect, keeping the queue's own output too:
#   tmux new -s scaling
#   bash queues/scaling.sh 2>&1 | tee -a runs/queue_logs/queue_scaling.log
#   (detach: Ctrl-b d; reattach: tmux attach -t scaling)
#   SIZES="S B" bash queues/scaling.sh ...          # without the L runs
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
SIZES=${SIZES:-"S B L"}
mkdir -p runs/queue_logs runs/scaling
GPU_LOCK=runs/queue_logs/gpu.lock
CFGS=2.0,3.0,4.0,5.0
EVAL="--fd_dino=True --keep_ckpt_every_epochs=20"

# recipes per pipeline
PIX_PLAIN="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA $EVAL"
PIX_STACK="$PIX_PLAIN --attn_res_chunk=8 --warmup_epochs=1 --loss_t_weight=learned --loss_weight_normalize=False --grad_clip=0 \
--attn_res=full --adaln_single=True --kv_prefix_len=32 --kv_prefix_class_tokens=True --attn_gate=head"
LAT_PLAIN="--data_dir=$DATA --vae=mar-kl16 --patch_size=1 $EVAL"
LAT_STACK="$LAT_PLAIN --attn_res=full --adaln_single=True --attn_res_chunk=8 --kv_prefix_len=32 --kv_prefix_class_tokens=True \
--attn_gate=head --loss_t_weight=learned --warmup_epochs=1 --grad_clip=0"
# sizes (L: the JiT-L preset on pixels, its shape given explicitly on latents; the stack's prefix starts at block 8 there)
PIX_S="--hidden_size=384 --num_heads=6"
LAT_S="--hidden_size=384 --num_heads=6"
PIX_L="--model=JiT-L/16 --batch_size=512 --gradient_accumulation_steps=2 --compile_mode=blocks --attn_res_kernel=True"
LAT_L="--depth=24 --hidden_size=1024 --num_heads=16 --gradient_accumulation_steps=2"
LAT_L_STACK="--kv_prefix_start=8"
PIXEL_B_STACK_RUN=runs/in1k_128/jit-b16_icl0/muon_attngate-head_kvprefix32_kvprefixclass_losst-learned_no-lossnorm_warmup1_clip0_attnres-full_adalnsingle

step() { # step name command...: skipped if done (re-checked once the GPU lock is held); interrupted training resumes
    local name=$1; shift
    local log="runs/queue_logs/$name.log" done="runs/queue_logs/$name.done"
    if [ -e "$done" ]; then echo "skip $name (done)"; return; fi
    local extra=() mode=">"
    if [ -e "$log" ] && { [ "$2" = "train_jit.py" ] || [ "$2" = "train_latent.py" ]; }; then
        local dir; dir=$(grep -m1 '^run:  ' "$log" | sed 's/^run:  //')
        if [ -n "$dir" ] && [ -e "$dir/ckpt.pt" ]; then extra=(--init_from=resume); mode=">>"; fi
    fi
    echo "=== $(date '+%F %T') start $name ${extra[*]+${extra[*]}}: $*"
    local guarded=(flock "$GPU_LOCK" bash -c '[ -e "$0" ] && exit 0; exec "$@"' "$done" "$@" ${extra[@]+"${extra[@]}"})
    # output to the step's log and the screen (tee); the exit code is the command's, not tee's
    local tee_args=(); [ "$mode" = ">>" ] && tee_args=(-a)
    "${guarded[@]}" 2>&1 | tee ${tee_args[@]+"${tee_args[@]}"} "$log"
    local rc=${PIPESTATUS[0]}
    [ $rc -eq 0 ] && touch "$done"
    echo "=== $(date '+%F %T') end $name (exit $rc)"
}
run_of() { grep -m1 '^run:  ' "runs/queue_logs/$1.log" 2>/dev/null | sed 's/^run:  //'; }
done_ok() { [ -e "runs/queue_logs/$1.done" ]; }
has() { case " $SIZES " in *" $1 "*) return 0 ;; esac; return 1; }
fid_at_epoch() { # a run's in-loop FID-10k at the eval nearest an epoch
    python -c "
import json, glob, sys, os
run, ep = sys.argv[1], float(sys.argv[2])
ipe = json.load(open(os.path.join(run, 'config.json')))['resolved']['iters_per_epoch']
d = [(abs(int(f.rsplit('_', 1)[1][:-5]) / ipe - ep), f) for f in glob.glob(os.path.join(run, 'diagnostics', 'diag_*.json'))]
print(json.load(open(min(d)[1]))['scalars'].get('val/fid', '') if d and min(d)[0] < 0.5 else '')" "$1" "$2"
}
decide() { # file what stepA valueA stepB valueB: the lower 20-epoch FID-10k wins (ties -> A); writes its value to the file
    local file=runs/scaling/$1.txt what=$2 sa=$3 va=$4 sb=$5 vb=$6
    [ -e "$file" ] && return
    done_ok "$sa" && done_ok "$sb" || return
    local fa fb w; fa=$(fid_at_epoch "$(run_of "$sa")" 20); fb=$(fid_at_epoch "$(run_of "$sb")" 20)
    if [ -z "$fa" ] || [ -z "$fb" ]; then echo ">>> $what: missing a 20-epoch FID ($va: '$fa', $vb: '$fb'); not decided"; return; fi
    if python -c "import sys; sys.exit(0 if float('$fb') < float('$fa') else 1)"; then w=$vb; else w=$va; fi
    echo "$w" > "$file"
    echo ">>> $what at S, 20 epochs: $va FID-10k $fa, $vb $fb -> $w"
}
sweep_pixel() { # step-name run-dir: guided sweep of the run's last checkpoint
    [ -n "$2" ] && step "$1" python eval_jit.py --ckpt="$2/ckpt.pt" --data_dir=$DATA --cfgs=$CFGS \
        --sweep_num_images=0 --final_num_images=10000 --out_dir="$2/eval_cfg_final"
}
sweep_latent() { # step-name run-dir training-args...: the same args rebuild the network
    local name=$1 d=$2; shift 2
    [ -n "$d" ] && step "$name" python train_latent.py "$@" --out_dir="$d" --init_from=resume --eval_cfgs=$CFGS --fid_interval_min=0.0
}

# 0. decisions at S, 20 epochs
step scaling_pixel_S_stack_bn128 python train_jit.py $PIX_STACK $PIX_S --bottleneck_dim=128 --epochs=20
step scaling_pixel_S_stack_bn64 python train_jit.py $PIX_STACK $PIX_S --bottleneck_dim=64 --epochs=20
decide bottleneck "pixel bottleneck (stack)" scaling_pixel_S_stack_bn128 128 scaling_pixel_S_stack_bn64 proportional
BN=$(cat runs/scaling/bottleneck.txt 2>/dev/null)
S_BN=$([ "$BN" = proportional ] && echo 64 || echo 128)
L_BN=$([ "$BN" = proportional ] && echo 170 || echo 128)
S_STACK_STEP=$([ "$BN" = proportional ] && echo scaling_pixel_S_stack_bn64 || echo scaling_pixel_S_stack_bn128)
if [ -n "$BN" ]; then
    step scaling_pixel_S_plain python train_jit.py $PIX_PLAIN $PIX_S --bottleneck_dim=$S_BN --epochs=20
    decide pixel_recipe "pixel recipe" "$S_STACK_STEP" stack scaling_pixel_S_plain plain
fi
step scaling_latent_S_stack python train_latent.py $LAT_STACK $LAT_S --epochs=20
step scaling_latent_S_plain python train_latent.py $LAT_PLAIN $LAT_S --epochs=20
decide latent_recipe "latent recipe" scaling_latent_S_stack stack scaling_latent_S_plain plain
PR=$(cat runs/scaling/pixel_recipe.txt 2>/dev/null)
LR=$(cat runs/scaling/latent_recipe.txt 2>/dev/null)
if [ -z "$PR" ] || [ -z "$LR" ]; then echo "decisions incomplete (bottleneck '$BN', pixel '$PR', latent '$LR'): stopping"; exit 1; fi
PIX_R=$([ "$PR" = stack ] && echo "$PIX_STACK" || echo "$PIX_PLAIN")
LAT_R=$([ "$LR" = stack ] && echo "$LAT_STACK" || echo "$LAT_PLAIN")
echo "=== recipes: pixel $PR (bottleneck $BN), latent $LR"

# 1. B, 120 epochs
PIXEL_B_RUN=""
if has B; then
    if [ "$PR" = stack ]; then
        PIXEL_B_RUN=$PIXEL_B_STACK_RUN
    else
        step scaling_pixel_B python train_jit.py $PIX_R --epochs=120
        done_ok scaling_pixel_B && PIXEL_B_RUN=$(run_of scaling_pixel_B)
    fi
    sweep_pixel scaling_pixel_B_cfg "$PIXEL_B_RUN"
    step scaling_latent_B python train_latent.py $LAT_R --epochs=120
    done_ok scaling_latent_B && sweep_latent scaling_latent_B_cfg "$(run_of scaling_latent_B)" $LAT_R --epochs=120
fi

# 2. S: the winning 20-epoch runs resumed to 80 epochs
if has S; then
    s_pix_step=$([ "$PR" = stack ] && echo "$S_STACK_STEP" || echo scaling_pixel_S_plain)
    step scaling_pixel_S python train_jit.py $PIX_R $PIX_S --bottleneck_dim=$S_BN --epochs=80 --out_dir="$(run_of "$s_pix_step")" --init_from=resume
    done_ok scaling_pixel_S && sweep_pixel scaling_pixel_S_cfg "$(run_of scaling_pixel_S)"
    s_lat_step=scaling_latent_S_$LR
    step scaling_latent_S python train_latent.py $LAT_R $LAT_S --epochs=80 --out_dir="$(run_of "$s_lat_step")" --init_from=resume
    done_ok scaling_latent_S && sweep_latent scaling_latent_S_cfg "$(run_of scaling_latent_S)" $LAT_R $LAT_S --epochs=80
fi

# 3. L, 40 epochs
if has L; then
    step scaling_pixel_L python train_jit.py $PIX_R $PIX_L --bottleneck_dim=$L_BN --epochs=40
    done_ok scaling_pixel_L && sweep_pixel scaling_pixel_L_cfg "$(run_of scaling_pixel_L)"
    lat_l="$LAT_R $LAT_L $([ "$LR" = stack ] && echo "$LAT_L_STACK")"
    step scaling_latent_L python train_latent.py $lat_l --epochs=40
    done_ok scaling_latent_L && sweep_latent scaling_latent_L_cfg "$(run_of scaling_latent_L)" $lat_l --epochs=40
fi

# projections over the final runs (fit_fid.py needs >= 3 evals per run)
dirs=() labels=()
add() { [ -n "$2" ] && [ -e "$2/config.json" ] && { dirs+=("$2"); labels+=("$1"); }; }
add "pixel S ($PR)" "$(run_of scaling_pixel_S)"; add "pixel B ($PR)" "$PIXEL_B_RUN"; add "pixel L ($PR)" "$(run_of scaling_pixel_L)"
add "latent S ($LR)" "$(run_of scaling_latent_S)"; add "latent B ($LR)" "$(run_of scaling_latent_B)"; add "latent L ($LR)" "$(run_of scaling_latent_L)"
if [ ${#dirs[@]} -gt 0 ]; then
    python fit_fid.py "${dirs[@]}" --labels="$(IFS=,; echo "${labels[*]}")" --out=runs/scaling/fid_projection
fi
echo "=== $(date '+%F %T') scaling queue done"
