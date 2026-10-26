#!/usr/bin/env bash
# Start analyze_patch_sizes.py once another job finishes.
#   ./wait_then_analyze.sh 12345          wait for PID 12345 to exit
#   ./wait_then_analyze.sh train_jit.py   wait until no process's command line matches the pattern
#   ./wait_then_analyze.sh                wait 30 min
# Detached, surviving logout: nohup ./wait_then_analyze.sh 12345 > analyze_patch_sizes.log 2>&1 &
set -euo pipefail

RUNS=/home/dylandeshler/pixels/PixelFlow/runs/in1k_128
P32="$RUNS/jit-b16_p32_icl0/muon/ckpt.pt"
P16="$RUNS/jit-b16_icl0/muon/ckpt.pt"
P8="$RUNS/jit-b16_p8_icl0/muon/ckpt.pt"
DATA=/data/in1k_128

cd "$(dirname "$0")"
for f in "$P32" "$P16" "$P8" "$DATA/fid_stats.npz"; do [ -f "$f" ] || { echo "missing: $f"; exit 1; }; done

if [ $# -eq 0 ]; then
    echo "$(date): waiting 30 min"
    sleep 1800
elif [[ $1 =~ ^[0-9]+$ ]]; then
    echo "$(date): waiting for PID $1"
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
else
    echo "$(date): waiting for processes matching '$1'"
    # this script and the shells that launched it can contain the pattern too: leave them out
    ancestors() { local p=$$; while [ "$p" -gt 1 ]; do echo "$p"; p=$(ps -o ppid= -p "$p" | tr -d ' '); done; }
    skip=$(ancestors)
    while pgrep -f "$1" | grep -vxqF "$skip"; do sleep 60; done
fi

echo "$(date): starting analysis"
python analyze_patch_sizes.py --ckpts="$P32,$P16,$P8" --data_dir="$DATA"
