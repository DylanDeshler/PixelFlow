#!/usr/bin/env bash
# in1k_128 sampler sweep (2026-11-02), inference only: SiT's finding that SDE sampling with a tuned noise level beats the
# ODE, tested with the ancestral sampler (DDIM eta family) against Heun at matched network calls (100: ancestral makes 1 per
# step, heun 2), plus the shifted step grid. Doubles as a test of the copying hypothesis: models that follow their noisy
# input more (higher copy / trajectory excess) should gain more from fresh noise per step. bn768 copies 2x the baseline
# (0.258 vs 0.127). Each eval is one FID-10k (cfg 1), ~3-4 min. Shares the queues' GPU lock; resumes via .done markers.
#
#   nohup bash queues/in1k_128_samplers.sh > runs/queue_logs/queue_samplers.log 2>&1 &
set -u
DATA=${JIT_DATA_ROOT:-/data}/in1k_128
mkdir -p runs/queue_logs
# one GPU: every step of every queue holds this lock while it runs, so the queues can be started in any order and take
# turns step by step (an earlier version only checked which queues were running at start, and three runs collided)
GPU_LOCK=runs/queue_logs/gpu.lock
step() { # step name command...
    local name=$1; shift
    local log="runs/queue_logs/$name.log" done="runs/queue_logs/$name.done"
    if [ -e "$done" ]; then echo "skip $name (done)"; return; fi
    echo "=== $(date '+%F %T') start $name: $*"
    flock "$GPU_LOCK" "$@" > "$log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && touch "$done"
    echo "=== $(date '+%F %T') end $name (exit $rc)"
}

for run in jit-b16_icl0/muon jit-b16_bn768_icl0/muon jit-b16_icl0/muon_losst-learned; do
    C="--ckpt=runs/in1k_128/$run/ckpt.pt --data_dir=$DATA --cfgs=1.0 --final_num_images=10000 --plot=False"
    tag=$(echo "$run" | tr '/' '_')
    step "smp_${tag}_heun50" python eval_jit.py $C --sampling_method=heun --num_sampling_steps=50
    step "smp_${tag}_heun50_shift" python eval_jit.py $C --sampling_method=heun --num_sampling_steps=50 --time_schedule=shift
    for eta in 0.25 0.5 1.0; do
        step "smp_${tag}_anc100_eta$eta" python eval_jit.py $C --sampling_method=ancestral --sampling_eta=$eta --num_sampling_steps=100
    done
done
echo "=== $(date '+%F %T') sampler queue done"
