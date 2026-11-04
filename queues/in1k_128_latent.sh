#!/usr/bin/env bash
# in1k_128 (2026-11-04): a latent-space baseline at JiT-B/16's scale (train_latent.py), and the Dispersive Loss on it with the
# paper's exact settings. Two questions:
#   1. how do a latent model's hidden representations compare with pixel-space JiT's, probed the same way (kNN / linear on
#      mean-pooled blocks 4 / 8 / 12 at five noise levels, the same images)? It's also the in1k_128 latent baseline.
#   2. does the Dispersive Loss help in latent space in our code? If yes, the implementation is validated and its failure on
#      JiT (every variant, 2026-11-03/04) is about pixel space / end-to-end training / AttnRes; if not, our setup is suspect.
# The network is JiT-B's blocks on SD-VAE (ft-EMA) latents, 2 x 2 patches: 64 tokens, width 768, depth 12, like JiT-B/16 at
# 128px; v-prediction (SiT). RECIPE=jit (default: the JiT baseline's Muon recipe, compare with runs/in1k_128/jit-b16_icl0/muon,
# FID-10k 81.0 at 20 epochs) or RECIPE=sit (SiT's AdamW recipe).
#   0. prepare_latents.py: encodes in1k_128 train + val with both flips (~5.2 GB; needs `pip install diffusers` and the VAE
#      weights from the hub)
#   1. the latent baseline, 20 epochs
#   2. the same + Dispersive Loss: plain InfoNCE-L2 on every token of block 3 (depth // 4), lambda 0.5, tau 0.5, 20 epochs
# Shares the queues' GPU lock; resumes via .done markers and checkpoints.
#
#   nohup bash queues/in1k_128_latent.sh > runs/queue_logs/queue_latent.log 2>&1 &
#   RECIPE=sit nohup bash queues/in1k_128_latent.sh ...
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
RECIPE=${RECIPE:-jit}
mkdir -p runs/queue_logs
GPU_LOCK=runs/queue_logs/gpu.lock
step() { # step name command...: skipped if done (re-checked once the GPU lock is held); interrupted training resumes
    local name=$1; shift
    local log="runs/queue_logs/$name.log" done="runs/queue_logs/$name.done"
    if [ -e "$done" ]; then echo "skip $name (done)"; return; fi
    local extra=() mode=">"
    if [ -e "$log" ] && [ "$2" = "train_latent.py" ]; then
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

step latent_prepare python prepare_latents.py --data_dir=$DATA
step "latent_${RECIPE}" python train_latent.py --data_dir=$DATA --recipe=$RECIPE --epochs=20
step "latent_${RECIPE}_disp" python train_latent.py --data_dir=$DATA --recipe=$RECIPE --epochs=20 --disp_weight=0.5
echo "latent baseline ($RECIPE): FID-10k $(last_fid "latent_${RECIPE}"); + Dispersive Loss (paper settings): $(last_fid "latent_${RECIPE}_disp")"
echo "=== $(date '+%F %T') latent queue done"
