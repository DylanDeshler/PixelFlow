#!/bin/bash
# Guided FID (CFG sweeps), one run after another (eval_jit.py, FID-10k per scale, EMA1, CFG at all t: the same settings as
# the row-7 and class-table 20-epoch sweeps):
#   1. 80 epochs, first (the headline long-training comparison): row 7 and the class table at ckpt_100080.pt, scales
#      2,3,4,5 (the best scale drops with training); 1.5 is added if the best is 2, 6 if it is 5. -> {run}/eval_cfg80/
#   2. every ladder row at 20 epochs: its ckpt.pt at iter 25020 (or a ckpt_25020.pt milestone; runs extended past 20
#      epochs without one are skipped with a note), scales 3,4,5; 6 and 7 added if the best is 5. -> {run}/eval_cfg20/
# Done steps are skipped on a rerun. A summary table at the end.
#
# $ bash queues/ladder_cfg.sh            (from the repo root, GPU free)
set -u
ROOT=${ROOT:-runs/in1k_128}
DATA=${DATA:-/data/in1k_128}
RUNS=(
  "ref|jit-b16/no-lossnorm"
  "0 adamw icl0|jit-b16_icl0/no-lossnorm"
  "1 muon|jit-b16_icl0/muon"
  "2 +warmup1|jit-b16_icl0/muon_warmup1"
  "3 +learned weight|jit-b16_icl0/muon_losst-learned_no-lossnorm_warmup1"
  "4 +clip0|jit-b16_icl0/muon_losst-learned_no-lossnorm_warmup1_clip0"
  "4b +adaln-single|jit-b16_icl0/muon_losst-learned_no-lossnorm_warmup1_clip0_adalnsingle"
  "5 +attnres|jit-b16_icl0/muon_losst-learned_no-lossnorm_warmup1_clip0_attnres-full_adalnsingle"
  "6 +kv-prefix32|jit-b16_icl0/muon_kvprefix32_losst-learned_no-lossnorm_warmup1_clip0_attnres-full_adalnsingle"
  "6' in-context32|jit-b16/muon_losst-learned_no-lossnorm_warmup1_clip0_attnres-full_adalnsingle"
)
# row 7 (17.13) and the class table (14.75) already have their 20-epoch sweeps
LONG=(
  "7 +gate (80 ep)|jit-b16_icl0/muon_attngate-head_kvprefix32_losst-learned_no-lossnorm_warmup1_clip0_attnres-full_adalnsingle"
  "class table (80 ep)|jit-b16_icl0/muon_attngate-head_kvprefix32_kvprefixclass_losst-learned_no-lossnorm_warmup1_clip0_attnres-full_adalnsingle"
)

ckpt_for() { # the 20-epoch checkpoint of run dir $1, or empty
  local d=$1
  if [ -f "$d/ckpt_25020.pt" ]; then echo "$d/ckpt_25020.pt"; return; fi
  [ -f "$d/ckpt.pt" ] || return
  local it
  it=$(python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('iter_num'))" "$d/ckpt.pt" 2>/dev/null)
  if [ "$it" = "25020" ]; then echo "$d/ckpt.pt"; else echo "skip:$it"; fi
}

best_cfg() { python -c "import json,sys; r=json.load(open(sys.argv[1])); print(min(r['fid'], key=lambda k: r['fid'][k]))" "$1"; }

# 1. 80 epochs
for entry in "${LONG[@]}"; do
  label=${entry%%|*}; rel=${entry#*|}; d="$ROOT/$rel"; ck="$d/ckpt_100080.pt"; out="$d/eval_cfg80"
  if [ ! -f "$ck" ]; then echo "== $label: no $ck -- skipped"; continue; fi
  if [ -f "$out/eval_10k_heun50.json" ]; then echo "== $label: already done"; continue; fi
  echo "== $label: $ck"
  python eval_jit.py --ckpt="$ck" --data_dir="$DATA" --cfgs=2.0,3.0,4.0,5.0 --sweep_num_images=0 --final_num_images=10000 --out_dir="$out" || continue
  b=$(best_cfg "$out/eval_10k_heun50.json")
  if [ "$b" = "2.0" ] || [ "$b" = "5.0" ]; then
    extra=$([ "$b" = "2.0" ] && echo "1.5" || echo "6.0")
    mv "$out/eval_10k_heun50.json" "$out/eval_10k_heun50_cfg2345.json"
    python eval_jit.py --ckpt="$ck" --data_dir="$DATA" --cfgs="$b,$extra" --sweep_num_images=0 --final_num_images=10000 --out_dir="$out"
  fi
done

# 2. ladder rows at 20 epochs
for entry in "${RUNS[@]}"; do
  label=${entry%%|*}; rel=${entry#*|}; d="$ROOT/$rel"
  ck=$(ckpt_for "$d")
  if [ -z "$ck" ] || [[ "$ck" == skip:* ]]; then echo "== $label: no 20-epoch checkpoint in $d (${ck#skip:}) -- skipped"; continue; fi
  out="$d/eval_cfg20"
  if [ -f "$out/eval_10k_heun50.json" ]; then echo "== $label: already done"; continue; fi
  echo "== $label: $ck"
  python eval_jit.py --ckpt="$ck" --data_dir="$DATA" --cfgs=3.0,4.0,5.0 --sweep_num_images=0 --final_num_images=10000 --out_dir="$out" || continue
  if [ "$(best_cfg "$out/eval_10k_heun50.json")" = "5.0" ]; then
    mv "$out/eval_10k_heun50.json" "$out/eval_10k_heun50_cfg345.json"
    python eval_jit.py --ckpt="$ck" --data_dir="$DATA" --cfgs=5.0,6.0,7.0 --sweep_num_images=0 --final_num_images=10000 --out_dir="$out"
  fi
done

summary() { # $1 = label, $2 = results dir
  python - "$1" "$2" <<'EOF'
import json, sys, glob, os
label, out = sys.argv[1], sys.argv[2]
fid = {}
for f in glob.glob(os.path.join(out, 'eval_10k_heun50*.json')):
    fid.update(json.load(open(f))['fid'])
if fid:
    b = min(fid, key=fid.get)
    print(f"  {label:<20} best {fid[b]:.2f} at cfg {b}   (" + ', '.join(f'{k}: {v:.2f}' for k, v in sorted(fid.items(), key=lambda kv: float(kv[0]))) + ')')
else:
    print(f"  {label:<20} -")
EOF
}
echo; echo "summary, 80 epochs (best guided FID-10k):"
for entry in "${LONG[@]}"; do summary "${entry%%|*}" "$ROOT/${entry#*|}/eval_cfg80"; done
echo "summary, 20 epochs (best guided FID-10k; row 7 17.13 and class table 14.75 from their earlier sweeps):"
for entry in "${RUNS[@]}"; do summary "${entry%%|*}" "$ROOT/${entry#*|}/eval_cfg20"; done
