#!/bin/bash
# REPA alignment-target comparison on one dataset tier: how much of DINOv2's benefit comes from
# the representation itself (ImageNet-only DINO / MAE) vs DINOv2's larger pretraining data.
# Runs land in runs/{dataset}/{model}/base and .../repa_encoder={enc}.
# usage: ./run_repa_targets.sh config/jit_in100_128.py [extra --key=value overrides]
set -e
CONFIG=$1; shift

python train_jit.py "$CONFIG" "$@"
for ENC in dino_vitb16 mae_vitb16 dinov2_vitb14; do
    python train_jit.py "$CONFIG" --repa_encoder="$ENC" "$@"
done
# iBOT IN-1k vs IN-22k (same recipe, more data) once the checkpoints are downloaded:
# python train_jit.py "$CONFIG" --repa_encoder=file:/path/to/ibot_vitb16_in1k.pth "$@"
# python train_jit.py "$CONFIG" --repa_encoder=file:/path/to/ibot_vitb16_in22k.pth "$@"
