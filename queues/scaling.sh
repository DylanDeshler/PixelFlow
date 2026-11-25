#!/usr/bin/env bash
# Scaling plan (2026-11-25): does end-to-end pixel diffusion scale better than latent diffusion? Stack vs stack: the same
# improvements on both pipelines (row 8 = class-table KV prefix + gate + full AttnRes + adaLN-single + Muon / warm-up 1 /
# learned weight / no clipping), at 128px, model sizes S / B / L, every run long enough to fit its FID curve (fit_fid.py)
# and project it to longer training.
#   pixel   JiT /16 on pixels (train_jit.py)
#   latent  the same blocks on MAR's ImageNet-only KL-16 latents, 16 x 8 x 8, patch 1 (train_latent.py), the stronger and
#           in-domain VAE (SD-VAE was 18 FID worse unguided)
# Both: 64 tokens, the same trunk FLOPs per image, evals every 5 epochs (FID-10k + FD-DINOv2), milestone checkpoints every
# 20 epochs (later guided sweeps), constant lr after warm-up (so each run is one curve).
#   1. latent B, 120 epochs: matches the existing pixel class-table run (runs/in1k_128/jit-b16_icl0/..._kvprefixclass_...,
#      120 epochs, no FD-DINOv2: it predates it). Its 20-epoch eval also says whether the stack transfers to latent
#      (plain latent: 39.94 unguided at 20 epochs)
#   2. S (width 384, 6 heads, depth 12; ~1/4 of B's FLOPs), 80 epochs: pixel, then latent
#   3. L (width 1024, 16 heads, depth 24, the prefix from block 8 as JiT-L), 40 epochs, as 512 x 2 accumulation: pixel, then
#      latent. ~3.5x B's FLOPs: the longest step (roughly 1-1.5 days each); check the first log lines for img/s and memory
#   after each run: a guided sweep of its final checkpoint, cfg 2 / 2.5 / 3 / 4 / 5 (FID-10k, CFG at all t)
#   at the end: fit_fid.py over all runs -> runs/scaling/fid_projection.{png,json}
# The JiT paper's bottleneck (128) is kept at every size (it uses 128 for B and L, 256 only from H).
# Shares the queues' GPU lock; resumes via .done markers and checkpoints. Skip sizes with SIZES (default "B S L").
#
#   nohup bash queues/scaling.sh > runs/queue_logs/queue_scaling.log 2>&1 &
#   SIZES="B S" nohup bash queues/scaling.sh ...          # without the L runs
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
SIZES=${SIZES:-"B S L"}
mkdir -p runs/queue_logs runs/scaling
GPU_LOCK=runs/queue_logs/gpu.lock
CFGS=2.0,2.5,3.0,4.0,5.0

# the stack, per pipeline (row 8)
PIX="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA --attn_res_chunk=8 --warmup_epochs=1 --loss_t_weight=learned \
--loss_weight_normalize=False --grad_clip=0 --attn_res=full --adaln_single=True --kv_prefix_len=32 --kv_prefix_class_tokens=True \
--attn_gate=head --fd_dino=True --keep_ckpt_every_epochs=20"
LAT="--data_dir=$DATA --vae=mar-kl16 --patch_size=1 --attn_res=full --adaln_single=True --attn_res_chunk=8 --kv_prefix_len=32 \
--kv_prefix_class_tokens=True --attn_gate=head --loss_t_weight=learned --warmup_epochs=1 --grad_clip=0 --fd_dino=True \
--keep_ckpt_every_epochs=20"
# sizes (same flags on both pipelines; L: the JiT-L preset on pixels, its shape and prefix start given explicitly on latents)
PIX_S="--hidden_size=384 --num_heads=6"
LAT_S="--hidden_size=384 --num_heads=6"
PIX_L="--model=JiT-L/16 --batch_size=512 --gradient_accumulation_steps=2 --compile_mode=blocks --attn_res_kernel=True"
LAT_L="--depth=24 --hidden_size=1024 --num_heads=16 --kv_prefix_start=8 --gradient_accumulation_steps=2"
PIXEL_B_RUN=runs/in1k_128/jit-b16_icl0/muon_attngate-head_kvprefix32_kvprefixclass_losst-learned_no-lossnorm_warmup1_clip0_attnres-full_adalnsingle

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
    if [ "$mode" = ">>" ]; then "${guarded[@]}" >> "$log" 2>&1; else "${guarded[@]}" > "$log" 2>&1; fi
    local rc=$?
    [ $rc -eq 0 ] && touch "$done"
    echo "=== $(date '+%F %T') end $name (exit $rc)"
}
run_of() { grep -m1 '^run:  ' "runs/queue_logs/$1.log" 2>/dev/null | sed 's/^run:  //'; }
done_ok() { [ -e "runs/queue_logs/$1.done" ]; }

pixel() { # size-tag epochs size-args: train, then sweep the final checkpoint
    local tag=$1 ep=$2; shift 2
    step "scaling_pixel_$tag" python train_jit.py $PIX "$@" --epochs=$ep
    local d; d=$(run_of "scaling_pixel_$tag")
    if done_ok "scaling_pixel_$tag" && [ -n "$d" ]; then
        step "scaling_pixel_${tag}_cfg" python eval_jit.py --ckpt="$d/ckpt.pt" --data_dir=$DATA --cfgs=$CFGS \
            --sweep_num_images=0 --final_num_images=10000 --out_dir="$d/eval_cfg_final"
    fi
}
latent() { # size-tag epochs size-args
    local tag=$1 ep=$2; shift 2
    step "scaling_latent_$tag" python train_latent.py $LAT "$@" --epochs=$ep
    local d; d=$(run_of "scaling_latent_$tag")
    if done_ok "scaling_latent_$tag" && [ -n "$d" ]; then
        step "scaling_latent_${tag}_cfg" python train_latent.py $LAT "$@" --epochs=$ep --out_dir="$d" --init_from=resume \
            --eval_cfgs=$CFGS --fid_interval_min=0.0
    fi
}

for size in $SIZES; do
    case $size in
        B) latent B 120
           step scaling_pixel_B_cfg python eval_jit.py --ckpt="$PIXEL_B_RUN/ckpt.pt" --data_dir=$DATA --cfgs=$CFGS \
               --sweep_num_images=0 --final_num_images=10000 --out_dir="$PIXEL_B_RUN/eval_cfg_final" ;; # the existing run's last checkpoint
        S) pixel S 80 $PIX_S; latent S 80 $LAT_S ;;
        L) pixel L 40 $PIX_L; latent L 40 $LAT_L ;;
        *) echo "unknown size $size (B, S or L)" ;;
    esac
done

# projections over every finished or partial run (fit_fid.py needs >= 3 evals per run)
dirs=() labels=()
add() { [ -n "$2" ] && [ -e "$2/config.json" ] && { dirs+=("$2"); labels+=("$1"); }; }
add "pixel S" "$(run_of scaling_pixel_S)"; add "pixel B" "$PIXEL_B_RUN"; add "pixel L" "$(run_of scaling_pixel_L)"
add "latent S" "$(run_of scaling_latent_S)"; add "latent B" "$(run_of scaling_latent_B)"; add "latent L" "$(run_of scaling_latent_L)"
if [ ${#dirs[@]} -gt 0 ]; then
    python fit_fid.py "${dirs[@]}" --labels="$(IFS=,; echo "${labels[*]}")" --out=runs/scaling/fid_projection
fi
echo "=== $(date '+%F %T') scaling queue done"
