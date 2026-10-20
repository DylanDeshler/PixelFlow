# cross_patch

Single-GPU ImageNet-1k pretraining + frozen-feature evaluation for trying out ViT encoder ideas against DINO.

| file | what |
| --- | --- |
| `vit.py` | baseline ViT (key-compatible with official DINO weights), DINO head, multi-crop wrapper, DINO loss. Defines the encoder interface new models must implement. |
| `train_dino.py` | DINO pretraining. Defaults = official ViT-S/16 100 epoch recipe (effective batch 512). Logs k-NN accuracy every ~10 epochs. |
| `eval_linear.py` | final eval of a checkpoint: full 20-NN + DINOv2-style linear probe sweep (52 heads, one backbone pass). |
| `data.py` / `evaluation.py` | ImageNet loading, DINO augmentations, eval transforms, feature extraction, k-NN classifier. |
| `prepare_imagenet.py` | downloads ImageNet from Hugging Face (or uses the official tarballs) and writes the ImageFolder layout. |
| `configurator.py` | nanoGPT-style `--key=value` overrides of the config globals. |

## Workflow

```bash
pip install -r requirements.txt
# accept the terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k first (auto-approved)
huggingface-cli login
python prepare_imagenet.py --source=hf --raw_dir=/data/imagenet_hf --out_dir=/data/imagenet
rm -rf /data/imagenet_hf   # optional, the parquet shards aren't needed once extracted

# 1. check the eval pipeline reproduces published numbers (should land near 74.5 k-NN / 77.0 linear)
python eval_linear.py --ckpt=dino_vits16 --data_dir=/data/imagenet

# 2. baseline + your idea, same budget
python train_dino.py --data_dir=/data/imagenet --out_dir=dino_vits16_100ep
python train_dino.py --data_dir=/data/imagenet --out_dir=my_idea_100ep   # after swapping `net` in train_dino.py

# 3. final numbers
python eval_linear.py --ckpt=my_idea_100ep/ckpt.pt --data_dir=/data/imagenet
```

To try a new encoder, put it in its own file (or in `vit.py`) with `embed_dim`, `num_prefix_tokens`,
`forward(x) -> [B, D]` and `get_intermediate_layers(x, n)`, then change the `from vit import VisionTransformer as net`
line in `train_dino.py`. The checkpoint records which class it was, so `eval_linear.py` needs no changes.

## Reference numbers (ImageNet-1k val top-1, as reported by the authors)

| model | epochs | k-NN | linear |
| --- | --- | --- | --- |
| DINO ViT-S/16, this repo's default recipe | 100 | 69.3 | 74.0 |
| DINO ViT-S/16 | 800 | 74.5 | 77.0 |
| DINO ViT-B/16 | 400 | 76.1 | 78.2 |
| DINOv2 ViT-S/14 (distilled from ViT-g, LVD-142M data) | - | 79.0 | 81.1 |
| DINOv2 ViT-B/14 (distilled from ViT-g, LVD-142M data) | - | 82.1 | 84.5 |

DINOv2 numbers come from a much larger curated dataset, so they're an upper bound rather than a like-for-like comparison.
Re-measuring the official checkpoints with `eval_linear.py` (`--ckpt=dino_vits16`, `dino_vitb16`, `dinov2_vits14`, ...) gives
reference points from exactly the same pipeline as your models.

## Compute notes

- DINO reports 1.75 days on 8 GPUs for the 100 epoch ViT-S/16 run. Expect several days on a single H100, and only if
  the data loader keeps up. Multi-crop decoding needs roughly 32+ CPU cores at full speed; watch `imgs_per_sec`. If
  you're CPU bound, `prepare_imagenet.py --short_side=320` (resized copies) or pillow-simd help a lot.
- For iterating on ideas, compare against a baseline trained with the same short schedule (e.g. `--epochs=20`), and run the full
  100 epochs only for the promising ones. The k-NN curve in wandb usually ranks methods correctly well before the end.
- `ckpt.pt` is overwritten about once an epoch (atomically), so if a hosted GPU gets preempted, rerun with `--init_from=resume`.
- If `batch_size=256` doesn't fit, lower it and raise `gradient_accumulation_steps` to keep the effective batch at 512.
  The ViT has no BatchNorm, so accumulation is exactly equivalent.

# JiT: pixel-space flow matching

`model_jit.py`, `denoiser.py`, `util/` are vendored from [LTH14/JiT](https://github.com/LTH14/JiT) (MIT, `LICENSE_JiT`).
The only local changes make the model device-agnostic (their code hard-coded `.cuda()`); computation and checkpoint keys are unchanged.

| file | what |
| --- | --- |
| `train_jit.py` | single-GPU JiT training (paper defaults), FID tracking, optional REPA (`--repa_encoder=...`) |
| `eval_jit.py` | FID/IS with a CFG sweep, for our checkpoints and JiT's released ones |
| `prepare_jit_data.py` | center-cropped uint8 memmaps (ImageNet-1k or ImageNet-100, any resolution) |
| `fid.py` | in-memory FID/IS (numerically the same as JiT's setup), precision/recall, reference stats and features |
| `diagnostics.py` | data-efficiency diagnostics run at every FID eval (see below) |
| `repa.py` | frozen alignment targets (DINO, MAE, DINOv2, any DINO-format checkpoint), projector, loss |
| `masking.py` | MAE-style token masking (`--mask_ratio`), same flow-matching loss, masked patches predicted from context |
| `config/jit_*.py`, `config/model_*.py` | starting configs for each dataset tier, and model configs to compose with them |
| `run_repa_targets.sh` | the alignment-target comparison on one tier |

## Configuration

Every setting is a global in the config block at the top of `train_jit.py`. Set them from config files (applied left to right)
and/or `--key=value` (applied last), without editing the script. Keys and types are checked, so typos and wrong types are errors.
Paths are derived, not configured per run:

- data: `{data_root}/{dataset}`. `data_root` defaults to `$JIT_DATA_ROOT` or `/data`; tier configs only set `dataset`.
- run directory: `{runs_root}/{dataset}/{model}/{changes}`. `{model}` is the preset plus changed architecture fields
  (`jit-b16_d6_w384_h6`); `{changes}` lists every setting that differs from the script defaults and affects training
  (`noise_scale=0.5_repa_encoder=dino_vitb16`, or `base`). Budget, eval, logging, diagnostics and system settings are left out, so
  the same command with `--init_from=resume` (and e.g. more `--epochs`) continues the same run. Add `--run_name=seed2` for repeats.
  `--out_dir` / `--data_dir` still override the derived paths.

```bash
export JIT_DATA_ROOT=/data
python train_jit.py config/jit_in100_128.py                                   # runs/in100_128/jit-b16/base
python train_jit.py config/jit_in100_128.py config/model_jit_s16.py           # runs/in100_128/jit-b16_w384_h6/base
python train_jit.py config/jit_in100_128.py --model=JiT-L/16 --depth=16 --batch_size=128   # runs/in100_128/jit-l16_d16/base
python train_jit.py config/jit_in100_128.py --noise_scale=0.75                # runs/in100_128/jit-b16/noise_scale=0.75
```

The model is a JiT preset (`--model=JiT-B/16`, `JiT-L/16`, ...) with any architecture field overridden: `depth`, `hidden_size`,
`num_heads`, `patch_size`, `bottleneck_dim`, `in_context_len` (0 disables the in-context class tokens), `in_context_start`,
`mlp_ratio`. A value of -1 keeps the preset's. Custom architectures get a descriptive name such as `JiT-B/16(depth=6,hidden_size=384)`
and are stored in the checkpoint, so `eval_jit.py` and `--init_from=resume` rebuild them automatically. Each run writes its full
resolved config to `{out_dir}/config.json`.

## Dataset tiers

| tier | images | JiT-B/16 tokens | rough single-H100 cost (200 epochs, ignoring eval) |
| --- | --- | --- | --- |
| `in100_128` | ~128k | 8x8 | a few hours |
| `in100_256` | ~128k | 16x16 | ~half a day |
| `in1k_128` | 1.28M | 8x8 | ~half a day to a day |
| `in1k_256` | 1.28M | 16x16 | ~1.5-2 days (paper: FID 4.37) |

ImageNet-100 here is every 10th class. All tiers keep 16px patches (768-d), JiT's high-dimensional regime; at small patch
sizes every prediction target works and results stop transferring. The costs are FLOP-based estimates, so measure your own
`imgs_per_sec`. On ImageNet-100, check whether many epochs start to memorize the training set (FID is computed against it).

```bash
python prepare_jit_data.py --data_dir=/data/imagenet --out_dir=/data/in100_128 --img_size=128 --subset=in100
python prepare_jit_data.py --data_dir=/data/imagenet --out_dir=/data/in100_128 --img_size=128 --subset=in100 --split=val
python fid.py --data_dir=/data/in100_128          # FID stats + Inception features for train and val

python train_jit.py config/jit_in100_128.py                                   # -> runs/in100_128/jit-b16/base
python train_jit.py config/jit_in100_128.py --repa_encoder=dino_vitb16        # -> runs/in100_128/jit-b16/repa_encoder=dino_vitb16
./run_repa_targets.sh config/jit_in100_128.py         # none / DINO (IN-1k) / MAE (IN-1k) / DINOv2 (LVD-142M)
python eval_jit.py --ckpt=runs/in100_128/jit-b16/base/ckpt.pt --data_dir=/data/in100_128 --cfgs=1.0,1.5,2.0,2.5,3.0
```

`eval_jit.py --cfgs=1.0:4.0:0.25` sweeps guidance (comma lists work too). It reports the best-FID scale, a trade-off pick (highest IS
with FID within `--fid_tolerance`, default 5%, of the best) and the FID/IS Pareto frontier, and saves `cfg_sweep_*.png`: FID, IS and
precision/recall vs CFG, plus the FID-IS trade-off curve. Sweep at 10k images, then re-score the picks with `--final_num_images=50000`.

`eval_jit.py --save_samples=png|npz` keeps every generated image in `{out_dir}/samples/`, as a PNG folder (`{index}_class{label}.png`, JiT's own format)
or an ADM-evaluator `.npz` (`arr_0` images plus `labels`), with a `_classes.json` mapping labels to ImageNet indices and names.
Training saves fixed-noise 8x8 sample grids, captioned with class names, to `samples/` at every eval.

Noise scale follows JiT's rule `img_size / 256` (`--noise_scale` overrides). The paper always scales patch size with
resolution (256 tokens), so the 128px tiers with 16px patches are outside what it tested. Calibrate `noise_scale` on the
`in100_128` baseline before comparing methods there.

## Muon

`--optimizer_type=muon` (`muon.py`) uses Muon for the 2D weight matrices inside the transformer blocks and AdamW for the
rest (patch / time / class embeddings, output layer, gains, biases, REPA projector, mask token).

- `--muon_adjust_lr=match_rms_adamw` (default): Moonshot's scaling ("Muon is Scalable for LLM Training"), update x 0.2 sqrt(max(A, B)),
  so the update RMS matches AdamW's and Muon uses the AdamW `lr` and `weight_decay` as is.
- `--muon_adjust_lr=original`: Keller Jordan's reference, update x sqrt(max(1, A/B)) with `--muon_lr` (0.02) and `--muon_weight_decay` (0).
- Both: momentum 0.95 with Nesterov, 5 Newton-Schulz steps in bf16, matching `torch.optim.Muon` within bf16 noise.
- `--muon_split_fused` (default on): fused weights are orthogonalized per component: q / k / v, the two SwiGLU input projections,
  and the 6 adaLN projections (as modded-nanogpt does for q / k / v).
- `--muon_adaln=False` moves the zero-initialized adaLN projections to AdamW, for the ablation.

## Token masking

`--mask_ratio=0.5` replaces half the patch tokens of a masked sample with a learned mask token (after the patch embedding, so
position is kept), in a `--mask_prob` fraction of each batch (default 0.5; the rest trains unmasked, as at inference). There is no
extra loss term: the usual flow-matching loss on every patch makes masked patches an inpainting-from-context task. JiT's v-loss
weights the clean-image error by 1/(1-t)^2 (up to 400x), so with the default `--mask_weighting=effective_t` masked patches are weighted
as t=0, the noise level they actually have, rather than letting their irreducible error dominate at low noise (`jit` keeps the
literal weighting, for the ablation). Evaluation and sampling never mask. If masking works by forcing longer-range context, the
locality diagnostics (window ablation, attention distance at high noise) should move along with FID.

## Verify the pipeline against the paper first

1. `prepare_jit_data.py --subset=in1k --img_size=256` downloads JiT's official reference stats. Then `fid.py` on that dir
   prints the FID between our reference stats and theirs. It should be ~0, which confirms the preprocessing matches.
2. Download the released JiT-B/16 checkpoint (Dropbox link in their README) and run `eval_jit.py` on it with their
   evaluation flags (50k images, their CFG scale/interval). It should land near the paper's 3.66.

Each eval also reports the training objective on fixed train and held-out subsets (`eval/train_loss`, `eval/val_loss`, same noise
every eval), a single FID-10k of the EMA weights at CFG 1, and sample grids at CFG 1 and CFG 3 (`samples/samples_{iter}.png`,
`samples/samples_{iter}_cfg3.png`; the CFG-3 grid is one 64-image batch, no FID; `--sample_cfg`, 0 disables). Use `eval_jit.py` for guided
FIDs and CFG sweeps. With diagnostics on, `diagnostics/probes_{iter}.png` shows k-NN and linear probe
accuracy as block x noise-level matrices.

Training-time FIDs use no CFG, 10k samples and a fixed seed, so the same noise and labels are used at every eval and across runs.
They're for comparing runs with each other. Paper-comparable numbers need `eval_jit.py` with 50k samples and CFG.

## Data-efficiency diagnostics

With `diag=True` (default; needs the `--split=val` data and `fid.py` run after it), every FID eval also logs to wandb
and `{out_dir}/diagnostics/diag_{iter}.json`:

| # | what | wandb keys | reading it |
| --- | --- | --- | --- |
| 1 | loss per noise level on fixed train / held-out images (raw weights) | `noise_loss/*`, `noise_xmse/*`, `noise_gap/*` | slow high-noise (small t) bins: semantics bottleneck; slow low-noise bins: detail; per-bin gap: where it memorizes |
| 2 | precision / recall (k=3, Inception) | `val/precision`, `val/recall` | low recall with fine precision: coverage / data-limited |
| 3 | raw vs EMA weights (off by default, `--diag_raw=True`: a second full FID run) | `val/*_raw` | EMA much better: noisy gradients (batch size, variance reduction, EMA horizon) |
| 4 | grad norms, update/weight ratio per block (every `log_interval`), adaLN gates, attention entropy | `grad_norm/*`, `update_ratio/*`, `gate_*/*`, `attn_entropy/*` | blocks that barely update or are gated off; collapsed or uniform attention |
| 4b | locality: mean attention distance of patch queries (relative to uniform attention) and attention mass on the in-context class tokens, per block; windowed-attention ablation: held-out loss increase when attention is confined to non-overlapping k x k patch tiles in every layer (`diag_window_tiles`, default 1,2,4) | `attn_dist/*`, `attn_prefix_mass/*`, `window_loss_increase/*`, heatmaps | an increase near 0 at high noise (small t) means the model isn't using context beyond the tile where global structure is decided |
| 5 | k-NN + linear probe of mean-pooled tokens (blocks at 1/3, 2/3 and the end of the model, t=1, 0.75, 0.5, null label) | `probe/*` | whether / when semantic features emerge, and whether FID gains follow them |
| 6 | relative error per frequency band and noise level; sample vs real spectrum | `freq_err/*`, `spectrum_log_ratio/*`, heatmaps | capacity spent on (or missing) high frequencies |
| 7 | memorization: nearest-training-image distance of samples vs held-out images | `mem/nn_ratio`, `mem/near_dup_rate` | ratio < 1 or near-dup rate well above 1%: samples copy training images |

In the windowed ablation the in-context class tokens may only attend to each other, so they can't relay image content between
tiles; a test confirms that perturbing one tile leaves every other tile's output exactly unchanged. The model never saw masked
attention in training, so part of any increase is distribution shift. Compare across noise levels, training time and methods
rather than reading absolute values.

JiT's convention is t=1 clean and t=0 pure noise. The eval runs two extra sample batches (raw weights) plus about 40k
instrumented forward passes, so it costs roughly 2-3x a plain FID eval; `--diag_raw=False` or `--diag=False` cut that.

## Training-trend figures

`plot_trends.py` turns the per-eval diagnostics into trends over training (x = images seen, so runs with different batch
sizes line up by compute), written to `{run}/trends/`:

```bash
python plot_trends.py runs/in100_128/jit-b16/base
python plot_trends.py runs/in100_128/jit-b16/base runs/in100_128/jit-b16/mask_ratio=0.5 runs/in100_128/jit-b16_bn256/base --out=trends_compare
```

| figure | question it answers |
| --- | --- |
| `convergence.png` | FID (log-log), train / held-out loss, precision / recall, memorization: is progress steady, is it starting to copy? |
| `noise_levels.png` | held-out loss per noise level vs the first eval, recent improvement per level, train/held-out gap: which noise levels still learn |
| `detail.png` | per-band error at low and high noise, sample vs real spectrum: is fine detail improving or stuck (e.g. at the patch-bottleneck floor)? |
| `semantics_locality.png` | probe accuracy per block, windowed-attention loss increase, attention distance: do representations and global context use grow? |
| `optimization.png` | loss, gradient norm, update/weight ratio per parameter group (from `train_log.jsonl`): are some blocks barely learning? |
| `compare.png` / `.csv` | the key curves and latest values across runs |

`train_log.jsonl` (loss, lr, gradient norms, update ratios every `log_interval`) is written by `train_jit.py` alongside wandb.

## Noise-schedule calibration

`noise_calibration.py` models the data as a stationary Gaussian (decorrelated channels, measured Fourier spectrum) and computes,
in closed form, how the best possible linear denoiser's R^2 (coarse, power-weighted) and the information revealed about the
quantized signal (fine, count-weighted) grow with noise level. Noise scale s and the logit-normal mean are interchangeable:
s at P_mean mu is the same set of noise levels as 1 at mu - ln(s).

```bash
python noise_calibration.py --data=/data/in100_128 --ref_data=/data/in1k_256 --ref_noise_scale=1.0 --plot=calib.png
python noise_calibration.py --data=clips.npy --quant_bits=16 --noise_scale=1.0     # other modalities: [N, C, *spatial] arrays
```

- **Within a modality** (new resolution or dataset vs a tuned reference): use the *shared-content match*. Checks: synthetic
  natural-spectrum images reproduce sqrt(m/n) (128/256/512px -> 0.504 / 1.000 / 1.978). Real ImageNet photos (Imagenette) give 0.532 for 128px vs JiT's 256px recipe.
- **New modality, no reference**: the logit-normal fits to the two transition curves only bracket a good setting. On real
  ImageNet at 256px they give P_mean -3.0 (coarse) and +4.3 (fine), and JiT's tuned -0.8 sits ~30% of the way between them.
  Sweep inside that bracket and judge on samples (training losses aren't comparable across noise settings). The per-noise-level
  held-out loss diagnostic shows whether training mass is landing where the loss is still reducible.
