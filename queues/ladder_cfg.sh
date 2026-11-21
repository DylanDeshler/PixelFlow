#!/bin/bash
# Guided FID (CFG sweeps), one run after another (eval_jit.py, FID-10k per scale, EMA1, CFG at all t: the same settings as
# the row-7 and class-table 20-epoch sweeps):
#   1. 80 epochs, first (the headline long-training comparison): row 7 and the class table, scales 2,3,4,5 (the best
#      scale drops with training). -> {run}/eval_cfg80/
#   2. every ladder row at 20 epochs, scales 3,4,5. -> {run}/eval_cfg20/
# Checkpoint of a run at E epochs (iteration E x 1251): the milestone ckpt_<iter>.pt if it exists, else ckpt.pt if the
# iteration stored in it is exactly that (loaded and checked), else the run is skipped with a note.
# Guidance is capped at 5 (no extension). The summary reports, per run, the trade-off pick as eval_jit.py makes it (the
# highest IS among the scales whose FID is within 5% of the best FID, over all scales scored <= 5): its FID, IS and scale. Scales already scored are never re-scored; each eval_jit call's results are kept as
# eval_10k_heun50_cfg<scales>.json and merged for the summary.
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

ckpt_for() { # run dir $1, iteration $2: the milestone ckpt_$2.pt, else ckpt.pt if its stored iteration is $2, else "skip:<iter>"
  local d=$1 want=$2
  if [ -f "$d/ckpt_$want.pt" ]; then echo "$d/ckpt_$want.pt"; return; fi
  [ -f "$d/ckpt.pt" ] || { echo "skip:no ckpt.pt"; return; }
  local it
  it=$(python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('iter_num'))" "$d/ckpt.pt" 2>/dev/null)
  if [ "$it" = "$want" ]; then echo "$d/ckpt.pt"; else echo "skip:ckpt.pt is at iter $it, want $want"; fi
}

missing() { # $1 out dir, $2 comma list: the scales of $2 not yet scored there (comma list, possibly empty)
  python - "$1" "$2" <<'PYX'
import json, sys, glob, os
done = set()
for f in glob.glob(os.path.join(sys.argv[1], 'eval_10k_heun50*.json')):
    done |= {float(k) for k in json.load(open(f))['fid']}
print(','.join(c for c in sys.argv[2].split(',') if float(c) not in done))
PYX
}

score() { # $1 ckpt, $2 out dir, $3 comma list of scales: one eval_jit call, its results kept under their own name
  python eval_jit.py --ckpt="$1" --data_dir="$DATA" --cfgs="$3" --sweep_num_images=0 --final_num_images=10000 --out_dir="$2" || return 1
  mv "$2/eval_10k_heun50.json" "$2/eval_10k_heun50_cfg${3//,/_}.json"
}


for d0 in $(find "$ROOT" -type d \( -name eval_cfg20 -o -name eval_cfg80 \) 2>/dev/null); do # results of the old script: one name per call
  [ -f "$d0/eval_10k_heun50.json" ] || continue
  python - "$d0" <<'PYX'
import json, os, sys
d = sys.argv[1]
src = os.path.join(d, 'eval_10k_heun50.json')
scales = sorted(json.load(open(src))['fid'], key=float)
os.replace(src, os.path.join(d, 'eval_10k_heun50_cfg' + '_'.join(scales) + '.json'))
PYX
done

# 1. 80 epochs
for entry in "${LONG[@]}"; do
  label=${entry%%|*}; rel=${entry#*|}; d="$ROOT/$rel"; out="$d/eval_cfg80"
  ck=$(ckpt_for "$d" 100080)
  if [[ "$ck" == skip:* ]]; then echo "== $label: no 80-epoch checkpoint in $d (${ck#skip:}) -- skipped"; continue; fi
  todo=$(missing "$out" 2.0,3.0,4.0,5.0)
  if [ -z "$todo" ]; then echo "== $label: done"; continue; fi
  echo "== $label: $ck, scales $todo"
  score "$ck" "$out" "$todo"
done

# 2. ladder rows at 20 epochs
for entry in "${RUNS[@]}"; do
  label=${entry%%|*}; rel=${entry#*|}; d="$ROOT/$rel"
  ck=$(ckpt_for "$d" 25020)
  if [[ "$ck" == skip:* ]]; then echo "== $label: no 20-epoch checkpoint in $d (${ck#skip:}) -- skipped"; continue; fi
  out="$d/eval_cfg20"
  todo=$(missing "$out" 3.0,4.0,5.0)
  if [ -z "$todo" ]; then echo "== $label: done"; continue; fi
  echo "== $label: $ck, scales $todo"
  score "$ck" "$out" "$todo"
done

summary() { # $1 = label, $2 = results dir
  python - "$1" "$2" <<'EOF'
import json, sys, glob, os
label, out = sys.argv[1], sys.argv[2]
m = {}
for f in glob.glob(os.path.join(out, 'eval_10k_heun50*.json')):
    r = json.load(open(f))
    for k in r['fid']:
        m[float(k)] = (r['fid'][k], r['is'].get(k), r.get('precision', {}).get(k), r.get('recall', {}).get(k))
ok = [c for c in m if c <= 5]
if ok:
    best = min(m[c][0] for c in ok)
    pick = max((c for c in ok if m[c][0] <= 1.05 * best), key=lambda c: m[c][1]) # eval_jit's trade-off rule (fid_tolerance 0.05)
    print(f"  {label:<20} FID {m[pick][0]:6.2f}  IS {m[pick][1]:6.1f}  at cfg {pick:g}   (best FID {best:.2f}; scales scored <= 5: {', '.join(f'{c:g}' for c in sorted(ok))})")
else:
    print(f"  {label:<20} -")
EOF
}
echo; echo "summary, 80 epochs (FID-10k / IS trade-off pick: highest IS within 5% of the best FID, guidance <= 5):"
for entry in "${LONG[@]}"; do summary "${entry%%|*}" "$ROOT/${entry#*|}/eval_cfg80"; done
echo "summary, 20 epochs (same rule; row 7 and the class table from their earlier sweeps, see the ladder):"
for entry in "${RUNS[@]}"; do summary "${entry%%|*}" "$ROOT/${entry#*|}/eval_cfg20"; done
