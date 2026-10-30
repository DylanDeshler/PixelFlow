#!/usr/bin/env bash
# in1k_128 queue (2026-10-29): critical batch, loss weighting, batch size, bottleneck, then the from-scratch calibration.
# Sequential; each step logs to runs/queue_logs/{name}.log and a failed step doesn't stop the queue. Rerunning the same
# command resumes: finished steps (runs/queue_logs/{name}.done) are skipped, and an interrupted training step continues
# from its last checkpoint (train_jit.py saves one every epoch; its log names the run directory). calibrate.py resumes its
# own finished steps; a training run it was in the middle of restarts. Every training run is
# JiT-B/16 + Muon, 20 epochs, matched to the existing baseline runs/in1k_128/jit-b16_icl0/muon (batch 1024, constant lr, so
# --epochs=20 has the same schedule as that run's first 20 epochs) and compared to it at its 5 / 10 / 15 / 20-epoch evals.
#
#   nohup bash queues/in1k_128_weighting_batch.sh > runs/queue_logs/queue.log 2>&1 &
#   QUEUE_FROM=4 nohup bash queues/in1k_128_weighting_batch.sh ...     # skip steps before 4 (steps are numbered below)
#   rm runs/queue_logs/wiener.done                                      # make a finished step run again (from scratch)
# Rough times on one L40S: step 1 ~15 min, each 20-epoch run ~3.5-4 h at batch 1024 and ~4-5 h at batch 256, step 9 ~7 h.
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
FROM=${QUEUE_FROM:-1}
BASE="config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA --epochs=20"
mkdir -p runs/queue_logs
step() { # step N name command...
    local n=$1 name=$2; shift 2
    local log="runs/queue_logs/$name.log" done="runs/queue_logs/$name.done"
    if [ "$n" -lt "$FROM" ]; then echo "skip $n $name (before QUEUE_FROM)"; return; fi
    if [ -e "$done" ]; then echo "skip $n $name (done)"; return; fi
    local extra=() mode=">"
    if [ -e "$log" ] && [ "$2" = "train_jit.py" ]; then
        # an interrupted training run: resume it if it saved a checkpoint (its log's "run:  <dir>" line names the directory)
        local dir; dir=$(grep -m1 '^run:  ' "$log" | sed 's/^run:  //')
        if [ -n "$dir" ] && [ -e "$dir/ckpt.pt" ]; then extra=(--init_from=resume); mode=">>"; fi
    fi
    echo "=== $(date '+%F %T') start $n $name ${extra[*]:-}: $*"
    if [ "$mode" = ">>" ]; then "$@" "${extra[@]}" >> "$log" 2>&1; else "$@" > "$log" 2>&1; fi
    local rc=$?
    [ $rc -eq 0 ] && touch "$done"
    echo "=== $(date '+%F %T') end $n $name (exit $rc)"
}

# 1. critical batch of the plain vs Wiener-weighted training objective (mixed noise levels), on the p16 / p8 baselines
step 1 gradsnr python analyze_grad_snr.py --data_dir=$DATA --ts=0.1,0.3,0.6 --mixed_batches=48 \
    --ckpts=runs/in1k_128/jit-b16_icl0/muon,runs/in1k_128/jit-b16_p8_icl0/muon

# 2-3. loss weighting at the tuned schedule and batch (does either beat the baseline at matched epochs?)
step 2 wiener  python train_jit.py $BASE --loss_freq_weight=wiener
step 3 learned python train_jit.py $BASE --loss_t_weight=learned

# 4-5. batch size: 256 vs the baseline's 1024 at equal images seen (p16's critical batch is ~500). EMA decays per step are
# set for the same half-life in images (0.9999^(1/4), 0.9996^(1/4)). lr: train_jit scales it linearly with the batch
# (5e-5 at 256); step 5 uses the square-root rule instead (blr 4e-4 -> lr 1e-4)
step 4 bs256_linear python train_jit.py $BASE --batch_size=256 --ema_decay1=0.999975 --ema_decay2=0.9999 --run_name=bs256
step 5 bs256_sqrt   python train_jit.py $BASE --batch_size=256 --ema_decay1=0.999975 --ema_decay2=0.9999 --blr=4e-4 --run_name=bs256

# 6-7. bottleneck: does the 768-wide (no compression) patch embedding still lose to 128, with and without the weighting?
step 6 bn768        python train_jit.py $BASE --bottleneck_dim=768
step 7 bn768_wiener python train_jit.py $BASE --bottleneck_dim=768 --loss_freq_weight=wiener

# 8. both weightings together
step 8 wiener_learned python train_jit.py $BASE --loss_freq_weight=wiener --loss_t_weight=learned

# 9. the from-scratch procedure on a dataset where the answer is known (JiT's tuned P_mean -0.8 / P_std 0.8 at noise 0.5)
step 9 calibrate python calibrate.py --data_dir=$DATA --train_args="config/jit_in1k_128.py --optimizer_type=muon"
step 10 calib_reference python train_jit.py config/jit_in1k_128.py --optimizer_type=muon --data_dir=$DATA --epochs=10 \
    --eval_every_epochs=10 --warmup_epochs=1 --diag=False --out_dir=runs/calibration/in1k_128_reference

echo "=== $(date '+%F %T') queue done"
