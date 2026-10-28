# cross_patch

Single-GPU ImageNet-1k pretraining + frozen-feature evaluation for trying out ViT encoder ideas against DINO.

| file | what |
| --- | --- |
| `vit.py` | baseline ViT (key-compatible with official DINO weights), DINO head, multi-crop wrapper, DINO loss. Defines the encoder interface new models must implement. |
| `train_dino.py` | DINO pretraining. Defaults = official ViT-S/16 100 epoch recipe (effective batch 512). Logs k-NN accuracy every ~10 epochs. |
| `eval_linear.py` | final eval of a checkpoint (DINO, official weights, or a JiT `train_jit.py` checkpoint): full 20-NN + DINOv2-style linear probe sweep (one backbone pass; JiT: depth-attention / attention-pooling heads, BatchNorm'd features). |
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
| `repa.py` | frozen alignment targets (DINO, MAE, DINOv2, timm models, any DINO-format checkpoint); REPA's MLP projector and PixelREPA's masked transformer adapter |
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
  as short tokens (`muon`, `noise0.75_pmean-1`, `repa-dino_vitb16`, `mask0.5`, or `base`). Budget, eval, logging, diagnostics and system settings are left out, so
  the same command with `--init_from=resume` (and e.g. more `--epochs`) continues the same run. Add `--run_name=seed2` for repeats.
  `--out_dir` / `--data_dir` still override the derived paths.

```bash
export JIT_DATA_ROOT=/data
python train_jit.py config/jit_in100_128.py                                   # runs/in100_128/jit-b16/base
python train_jit.py config/jit_in100_128.py config/model_jit_s16.py           # runs/in100_128/jit-b16_w384_h6/base
python train_jit.py config/jit_in1k_128.py config/model_jit_b8.py             # 8px patches (--patch_size=8): runs/in1k_128/jit-b16_p8_icl0/base
python train_jit.py config/jit_in100_128.py --model=JiT-L/16 --depth=16 --batch_size=128   # runs/in100_128/jit-l16_d16/base
python train_jit.py config/jit_in100_128.py --noise_scale=0.75                # runs/in100_128/jit-b16/noise0.75
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
python train_jit.py config/jit_in100_128.py --repa_encoder=dino_vitb16        # -> runs/in100_128/jit-b16/repa-dino_vitb16
./run_repa_targets.sh config/jit_in100_128.py         # none / DINO (IN-1k) / MAE (IN-1k) / DINOv2 (LVD-142M)
python eval_jit.py --ckpt=runs/in100_128/jit-b16/base/ckpt.pt --data_dir=/data/in100_128 --cfgs=1.0,1.5,2.0,2.5,3.0
```

`eval_jit.py --cfgs=1.0:4.0:0.25` sweeps guidance (comma lists work too). It reports the best-FID scale, a trade-off pick (highest IS
with FID within `--fid_tolerance`, default 5%, of the best) and the FID/IS Pareto frontier, and saves `cfg_sweep_*.png`: FID, IS and
precision/recall vs CFG, plus the FID-IS trade-off curve. Each scale is swept at `--sweep_num_images` (default 10k). The best-FID and
trade-off picks are then re-scored at `--final_num_images` (default 50k), the number to report. `--sweep_num_images=0` sweeps at the full count.

`eval_jit.py --save_samples=png|npz` keeps every generated image of the final `--final_num_images` evaluation in `{out_dir}/samples/`, as a PNG folder (`{index}_class{label}.png`, JiT's own format)
or an ADM-evaluator `.npz` (`arr_0` images plus `labels`), with a `_classes.json` mapping labels to ImageNet indices and names.
Training saves fixed-noise 8x8 sample grids, captioned with class names, to `samples/` at every eval, and rebuilds
`samples/progress.gif` / `samples/progress_cfg3.gif` (one frame per eval, same seed and classes; also logged to wandb) so you can flip
through training progress. For an existing run: `python progress_gif.py runs/in100_128/jit-b16/base`.

Noise scale follows JiT's rule `img_size / 256` (`--noise_scale` overrides). The paper always scales patch size with
resolution (256 tokens), so the 128px tiers with 16px patches are outside what it tested. Calibrate `noise_scale` on the
`in100_128` baseline before comparing methods there.

## Representation alignment: REPA vs PixelREPA

Plain REPA (`--repa_adapter=mlp`, REPA's MLP projector, weight 0.5) hurts JiT: Shin et al. 2026 report FID 5.14 vs 4.37 for JiT-B/16
at 200 epochs. `--repa_adapter=mta` is their PixelREPA, ported from github.com/kaist-cvml/PixelREPA and checked to give outputs identical
to their adapter: a 2-block JiT-style adapter on the JiT tokens after 20% random token masking, aligned to DINOv2 with 1 - cosine at weight 0.1.
Their encoder setup is DINOv2-B rebuilt by timm at JiT's image and patch size:

```bash
python train_jit.py config/jit_in100_128.py --repa_encoder=timm:vit_base_patch14_dinov2.lvd142m --repa_adapter=mta
```

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

## Attention Residuals

`--attn_res=block` (or `full`) replaces JiT's residual stream with Attention Residuals (Kimi Team 2026, arXiv 2603.15031,
`AttnResState` in `model_jit.py`). Each sublayer (every attention and every MLP) gets its input as a softmax mix over
earlier outputs, alpha = softmax(w_l . RMSNorm(v_i)), instead of the plain sum. There is one zero-initialized pseudo-query
per sublayer, plus one for the final layer, so mixing starts uniform. The sources are:
- `full`: the patch embedding and every gated sublayer output.
- `block`: the embedding, the summed outputs of each completed block of `--attn_res_block_size` JiT blocks (default about
  8 blocks, the paper's setting), and the running partial sum of the current block.

The paper reports about 1.25x compute efficiency for Block AttnRes on LLMs. This is the first test on a pixel-space
denoiser. In-context class tokens are added to the embedding source, and the other sources are zero at those positions.
The architecture is stored in the checkpoint, so `eval_jit.py` and `eval_linear.py` rebuild it. Block hooks (REPA,
probes) see each block's output as the input the next block actually receives.

With diagnostics on, AttnRes runs also log their depth-mixing weights. `diagnostics/attn_res_alpha_{iter}.png`
shows the layer x source weights at the noisiest, middle and cleanest noise level. `attn_res_emb_weight/t*` is the
final layer's weight on the patch embedding per noise level, i.e. how much it reads the input directly.
`trends/attn_res.png` shows both over training. If the final layer's embedding weight already swings with t,
routing adapts through the content-dependent keys alone.

`--attn_res_adaln=True` conditions the depth mixing on time and class with adaLN on the queries:
w_l(c) = w_l * (1 + scale(c)) + shift(c). One zero-initialized projection is shared by all queries, about 1.2M
parameters on JiT-B, and it starts as plain AttnRes. The shift goes on the query side because on the keys it would
cancel in the softmax over sources. The motivation: the best mixing should depend on the noise level, leaning on the
embedding near t=1 (keep and edit the input) and on deep outputs at high noise.

## Gated bottleneck

`--gated_bottleneck=True` adds a full-rank patch embedding next to JiT's 128-d bottleneck, behind a per-channel gate
from the conditioning (time + class), adaLN-style and zero-initialized:
`embedding = proj2(proj1(patch)) + gate(c) * bypass(patch)`. It starts as exactly the bottleneck model. The gate can
open where the input's fine detail is useful (low noise) and stay shut where the patch is mostly noise. It adds about
1.2M parameters on JiT-B. The motivation: with the bottleneck, the model is several times worse than copying its
input in the finest bands at low noise. Removing the bottleneck (`--bottleneck_dim=768`) fixes that and lowers the
loss, but costs FID and mid/deep probe accuracy. Token masking replaces the whole embedding, bypass included. The
diagnostics log the gated path's RMS relative to the bottleneck path per noise level (`bypass_ratio/t*`), plotted
in `trends/gated_bottleneck.png`.

## Skip pathway

`--skip_pathway=True` adds a full-rank patch embedding of the noisy input that skips the trunk. It is added, behind
a zero-initialized adaLN-style gate from the conditioning, to the trunk's output just before the final layer:
`h_out = h + gate(c) * skip(patch(z))`, then `--skip_head_blocks=N` optional JiT blocks (default 0), then the final
layer. It starts as exactly the model without it. The difference from `--gated_bottleneck`: the trunk only ever sees the
bottlenecked embedding, at every noise level, so it can't solve low-noise steps by editing the input locally (what
cost the no-bottleneck run its mid / deep-block semantics). The input's detail reaches only the output stage. With 0
head blocks, the merge is linear (a learned per-patch filter of the input plus the trunk's prediction); head blocks let
attention combine detail with trunk features. `--skip_inject_block` sets where the skip enters: 0 = after the trunk (default), -k = before the last k blocks
(-1 = after block 11 of 12), k = after block k. Mid-trunk, it repurposes the last blocks as a detail-aware head at
no extra parameters. The probe peak sits around two-thirds depth, and later blocks act more like a decoder. Without
AttnRes it is added to the residual stream. With AttnRes it is added into that block's output source (the AttnRes
form of a residual add). A separate source would dilute the other sources' softmax weight even with the gate shut.
Every variant starts as exactly the model without the skip. Token masking zeroes masked patches in the skip as well.

`--skip_stop_grad=True` keeps the trunk trained exactly like vanilla JiT. Each training step also computes the
trunk-only prediction (the same network without the skip), and the loss is L(with skip) + L(trunk only), the same
objective on both. The with-skip branch continues from a detached copy of the trunk's state at the injection point.
So its loss only trains the skip pathway, the blocks after the injection point (shared with the trunk branch) and
the final layer, never the embeddings or the blocks before it. Works with every `--skip_inject_block`; repeating the
blocks after the injection point costs about (12 - k) / 12 extra compute for injection after block k. `train/loss` is
the with-skip loss (comparable to other runs) and `train/loss_trunk` the trunk-only one. Evaluation and sampling use
the with-skip prediction.

`--skip_final_layer=True` (with `--skip_stop_grad`) gives the with-skip prediction its own final layer, starting as an
exact copy. JiT's final layer then serves only the trunk-only branch. With injection after the trunk, the trunk is
then trained exactly like vanilla JiT, apart from gradient clipping on the combined norm. Mid-trunk, the blocks after
the injection point are still shared. With `--dcae_head`, the DC-AE head belongs to the sampled (with-skip) path only,
and the trunk-only prediction stays exactly vanilla JiT. `--dcae_trunk_head` gives the trunk-only prediction its own,
separate DC-AE head. With both flags each branch has its own head and nothing in the decoder is shared.
About +0.6M params on JiT-B.

`--skip_residual=True` (with `--skip_final_layer --skip_stop_grad`) makes the with-skip prediction
`x_trunk.detach() + correction`. The trunk's own prediction (its final layer, DC-AE head if any) is the base, and the
skip branch's final layer only learns a correction from the gated skip. It is zero at init, so the prediction starts
exactly at the trunk's. The with-skip loss trains only the correction and the skip modules; detaching keeps it away
from the trunk and its final layer. Samples and evaluation use trunk + correction, so both final layers contribute.
With injection after the trunk this costs one extra final-layer pass. Mid-trunk, the blocks after the injection point
also run twice at sampling time.

`--skip_decoder=pixel|subpatch` (with `--skip_stop_grad`; default `linear`) replaces the with-skip branch's linear
decoder with a decoder over units of the image: single pixels (`pixel`, 256 per patch, width 16) or 4x4 pixel blocks
(`subpatch`, 16 per patch, width 64; `--skip_decoder_dim`). It keeps the structure that makes `--skip_final_layer` work
and adds PixelDiT's per-token modulation:
- **Start:** `tokens = readout(norm(h)) + gate(c) * embed(noisy pixels)`. The readout is a linear decode of each
  patch's detached trunk token into its decoder tokens. The noisy pixels sit behind a zero-initialized gate from the
  conditioning, as the skip pathway's. So the decoder starts as a linear decode of the trunk, like `skip_final_layer`.
- **Refinement:** `--skip_decoder_blocks` (default 2) blocks each give every token its own shift / scale / gate from
  the conditioning token of its patch, then a gated SwiGLU MLP. With 0 blocks it is the linear skip decoder at unit
  granularity: a control for what the per-token modulation adds.
- An earlier version without the readout learned very slowly (with-skip loss 0.15 vs the trunk's 0.06 at 5k
  iterations): with modulation as the only route for the trunk's content, it first learns a local denoiser of each
  unit's noisy pixels.
- `--skip_decoder_modulation=False` turns the blocks into plain residual MLP blocks (no per-token adaLN from the
  trunk tokens), isolating what the modulation adds.
- `--skip_decoder_rank=r` makes the modulation projections low-rank. `subpatch` with rank 112 matches the best linear
  setup's parameter count. `pixel`'s readout alone is 3.1M.
- `--skip_decoder_conv` adds a 3x3 conv over neighbouring tokens, the only path across patch boundaries.
- `--skip_decoder_attnres` (needs `--attn_res`) lets each block condition on its own AttnRes mix of the detached trunk
  sources, with a learned query, instead of the trunk output only.
- Diagnostics log the gated noisy-pixel input relative to the readout as `skip_ratio/t*`.

`--skip_pathway_drop_prob=p` (default 0, off) drops the skip for a random fraction p of training samples, the way
label dropout works for CFG. On those samples the trunk has to reconstruct everything through the bottleneck. The
motivation: with the skip always on, the trunk's probes trailed at every noise level, even where the skip's gate was
shut. That suggests reconstructing through the bottleneck is part of how the trunk learns its representations.
Evaluation and sampling always use the skip. It's a training setting, not part of the architecture. Diagnostics log
the gated skip's RMS relative to the trunk output per noise level (`skip_ratio/t*`, `trends/skip_pathway.png`).

## DC-AE conv stem / head and overlapping patches

DC-AE style resampling (Chen et al. 2024) around JiT's patching. At JiT's widths every DC-AE shortcut is an exact
reshape (3 x 4^2 = 48 channels at the 4px level, 48 x 4^2 = 768 = one 16px patch), so each DC-AE layer is
shuffle(x + branch(x)). Every branch is a residual: channel RMSNorm, adaLN modulation from the conditioning, 3x3 conv
(-> `--dcae_width`, default 128), SiLU, `--dcae_kernel2` conv back. Its gate is zero-initialized, so the model starts
as exactly vanilla JiT.
- `--dcae_stem`: DC-AE downsampling, convs at the higher resolution of each stage. One branch at full resolution
  (3 channels, width / 4), PixelUnshuffle(4), one at the 4px level (48 channels), then back to an image for the
  unchanged bottleneck patch embedding, which does the 4 -> 16 step. The 128-d bottleneck stays; it now compresses
  nonlinear, cross-patch features. About +0.2M params on JiT-B.
- `--dcae_head`: DC-AE upsampling, convs at the lower resolution of each stage, on the predicted image. One branch at
  the token level (768 channels, a 3x3 over neighbouring tokens), PixelShuffle(4), one at the 4px level, PixelShuffle(4).
  It only sees the trunk's output (no input path), so it can synthesize detail but not copy it. About +2.9M params.
- `--patch_overlap=o`: each patch also sees o pixels of its neighbours. The patch embedding's first conv becomes
  (16 + 2o)^2 with stride 16. The final layer predicts (16 + 2o)^2 patches, blended by overlap-add (averaged where
  they overlap). About +0.9M params for o = 4.
- `--head_overlap=o`: the output side of `--patch_overlap` alone, with the patch embedding unchanged. Each token's
  final layer predicts a (16 + 2o)^2 patch and neighbouring predictions are overlap-added (averaged), i.e. a transposed
  conv with kernel 16 + 2o and stride 16 in place of the per-patch projection, so seams are blended by construction.
  It applies to every final layer: with `--skip_final_layer` both the sampled head and the trunk-only head overlap, so
  the trunk's training signal changes too.

`--dcae_norm=group` replaces each branch's per-pixel channel norm with GroupNorm (gcd(32, C) groups: 1 for the
3-channel stem branch, 16 for the 48-channel branches, 32 for the token-level head branch), as DC-AE uses. The
default `channel` normalizes each pixel across its channels. For the 3-channel full-resolution stem branch that keeps
only each pixel's colour direction and drops its brightness, which likely hurt the stem once its branches became
active. It stays the default only so existing DC-AE checkpoints keep loading.

All three combine freely, with each other and with the other options. With token masking, a DC-AE stem or
overlapping patches would let masked patches' pixels leak into neighbouring tokens. Masked patches are therefore
zeroed in the input image as well.

## LayerNorm

`--layernorm=True` makes every RMSNorm in the network a LayerNorm with a learnable gain and bias: the blocks'
pre-norms, QK-norm in attention, the final layer, the AttnRes key norms and the DC-AE branches. The PixelREPA adapter
is a separate module and keeps its own. Computed in fp32 and returned in the input's dtype like JiT's RMSNorm, so
activations stay bf16 under autocast. AttnRes scores with LayerNorm keys subtract each source's mean:
`(v . u - mean(v) sum(u)) / std(v)`. The key norm's bias adds the same constant to every source's score, so it cancels.

## Fixed-budget quadtree tokenization (QuadJiT)

`model_quadtree.py` implements `research_proposal_fixed_budget_quadtree_patch_allocation.md`. It's a separate model that
copies what it changes from JiT, so the baseline can't be affected. Every image is tokenized into exactly
`--quad_tokens` patches drawn from 3 levels (32 / 16 / 8 px at 128px; K = 16 + 3m), placed by a router, and the JiT-B
backbone runs on those K tokens. The backbone has the best config's options (`--attn_res`, `--adaln_single`) and no skip
pathway.

How the pieces work:
- **Embedding:** each level has its own bottleneck patch embedding and final layer. Every candidate patch is embedded,
  each image gathers its own K tokens, and each token's output patch is scattered back onto the canvas. Shapes are
  static, so whole-model compile works.
- **Positions** (proposal Section 3.1):
  - RoPE on the physical patch centres;
  - footprint-integrated sin-cos as the absolute embedding (a parent's embedding is the mean of its children's);
  - a learned level embedding;
  - proportional attention: +log(area) on every key's logits.
- **Routing** runs under no_grad and outside the compiled graph, every training step and every sampling call.
  - `--quad_router=dino`: frozen DINOv2 (`--quad_route_dino`, at one 14px DINO patch per 8px patch, so 224px) on a
    frozen baseline's clean-image estimate. The score is the angular divergence of each patch's four quadrant features
    times a PCA-1 foreground gate, standardized per level, allocated by hierarchical greedy.
  - `pixel`: gradient energy of the same estimate.
  - `random`: a pseudo-random function of the input, so the two halves of a CFG batch share a layout.
  - `uniform`: coarse-first. K=64 is uniform 16px patches and K=256 uniform 8px, so the uniform baselines run on exactly
    the same code path.
  - The baseline (`--quad_route_baseline`) must be a fixed checkpoint (its EMA1 weights are used). Copy the in1k_128
    run's `ckpt.pt`, or use a milestone `ckpt_{iter}.pt`; a `ckpt.pt` that is still being written changes under you.
  - It predicts with the null label, so both halves of a CFG batch get one layout.
- **The router is not part of the model's parameters.** It isn't trained, EMA'd or saved, and it's rebuilt from the
  architecture fields when a checkpoint loads. `eval_jit.py`, `diagnose_ckpt.py` and `eval_linear.py` work unchanged.
  DINOv2 comes from torch.hub, so the first run needs internet access or a populated `~/.cache/torch/hub`.
- **Cost:** the dino router adds a baseline forward and a DINOv2-B forward per step, roughly +50% on a JiT-B/16 step at
  batch 1024.
- **Diagnostics,** every eval (`layout/*`, and `trends/layout.png` from `plot_trends.py`):
  - **Area per patch size,** per noise level.
  - **Router fidelity** (`clean_agree_t*`): the fraction of the image tokenized at the same size as when the router's
    scorer is applied to the clean image (dino / pixel routers). It shows how much routing survives noise and sets
    `--quad_route_t_min` from data.
  - **Error by patch size** (`xmse_p*_t*`): held-out clean-image error of the pixels each size covers. Do fine tokens
    lower the error where they're placed, and what does a coarse region cost?
  - **From 64 fixed-seed samples:**
    - layout stability along sampling (`agree_final_t*`, `agree_prev_mean`, `settle_t`: the t from which the layout
      stays within 90% of its final form), which sets `--quad_route_every`;
    - seams (`seam`, `seam_cross`): neighbour differences across token boundaries / inside tokens, next to the same ratio
      on real held-out images with the router's near-clean layouts (`seam_real`, `seam_cross_real`);
    - detail per patch size (`hf_ratio_p*`): high-frequency energy of the samples / real images in the regions each size
      covers. Below 1 under 32px tokens is the texture-blur risk.
- **Sample grids:** each grid is also saved with token boundaries drawn on (`samples_{step}_layout.png`), and with the
  layouts at t ≈ 0.25 / 0.5 / 0.75 / last for 8 samples (`_layout_steps.png`). `eval_jit.py` does the same for its
  final grid. Boundaries are coloured by the finer of the two tokens they separate: 32px blue, 16px orange, 8px green.
- **Comparing runs:** `plot_trends.py` with several runs adds `budget.png`, FID at matched training images vs tokens per
  image and vs estimated training GFLOPs per image (router included).
- **The router by eye, before any training:** `viz_quadtree_router.py`.
  - **On clean held-out images:** DINOv2's PCA-1 foreground projection and gate, plus every router's layouts at every
    budget.
  - **With `--baseline`:** the layout routed on the baseline's estimate from noised images, next to the clean-image
    layout, with the agreement per t.

```bash
python viz_quadtree_router.py --data_dir=/data/in1k_128                                      # -> in1k_128_quadtree_router.png
python viz_quadtree_router.py --data_dir=/data/in1k_128 --baseline=$BASE --ts=0.3,0.5,0.7,0.9  # + _noisy.png
```

```bash
B="config/jit_in1k_128.py config/model_quadjit_b32_8.py --optimizer_type=muon --attn_res=full --adaln_single=True --keep_ckpt_every_epochs=10"
BASE=runs/in1k_128/jit-b16_icl0/<best run>/ckpt_<iter>.pt                     # frozen router baseline
python train_jit.py $B --quad_router=uniform                                   # floor: uniform 16px, K=64
python train_jit.py $B --quad_router=uniform --quad_tokens=256                 # ceiling: uniform 8px (or --batch_size=512 --gradient_accumulation_steps=2)
python train_jit.py $B --quad_router=random                                    # mixed-scale machinery, does routing matter
python train_jit.py $B --quad_router=dino --quad_route_baseline=$BASE          # the method, K=64
python train_jit.py $B --quad_router=dino --quad_route_baseline=$BASE --quad_tokens=100   # budgets: also 160
python train_jit.py $B --quad_router=dino --quad_route_baseline=$BASE --quad_route_t_min=0.5   # uniform 16px below t=0.5
python train_jit.py $B --quad_router=pixel --quad_route_baseline=$BASE         # non-semantic routing (optional)
python train_jit.py $B --quad_router=dino --quad_route_baseline=$BASE --quad_rope=integrated   # ablations: --quad_pos=center,
                                                                               # --quad_area_bias=False, --quad_level_embed=False
python eval_jit.py --ckpt=runs/in1k_128/quadjit-b32-8/<run> --data_dir=/data/in1k_128 --quad_route_every=4   # re-route every 4 calls
```

`config/model_jit_b8.py` (plain JiT with 8px patches) remains as a cross-check of the K=256 ceiling, using JiT's own
position embedding.

### A no-training estimate of the gain: `analyze_patch_sizes.py`

This uses three plain-JiT checkpoints at 32/16/8px patches, trained alike to the same step. Generation stitches each
step's estimate region by region from the three models according to a router's layout, and is scored by FID-10k. Every
configuration uses the same noise and labels. The full description is in the script's docstring.
- **anchors:** each model alone. The 16px model is the K=64 baseline, the 8px model the K=256 ceiling.
- **stitched:** FID per router × K. "gap closed" is the share of the 16px → 8px FID improvement reached. Routers:
  - **loss:** the no-external-representation router from proposal Section 7, a greedy on the largest expected loss
    reductions. It uses the 8px estimate in place of E[x | z_t], so each model's expected error over a region is
    ‖x_l − x8‖² plus a posterior-variance term that is the same for every model. That makes the measured gain with
    x8 in place of x the conditional mean a split-value regression head would learn. It needs no training, but it
    borrows the 8px model, so it is optimistic.
  - **dino, pixel:** on the 16px null-label estimate, as in training.
  - **random, uniform.**
  - **disagree** (optional): equals loss for 16px parents.
  - **`<router>_2level`:** 16/8px only. Layouts split the uniform 16px layout (K ≥ 64), and the 32px model is never
    run. With two levels the greedy is already exact.
  - **`<router>_pooled`:** the batch's B·K tokens go to its best 16→8px splits in any image, so tokens per image vary,
    as with sequence packing. **`<router>_perimage`** is the control: each image is entirely 16px or entirely 8px, with
    the same average budget. Both need scores that are comparable across images (loss, disagree, random), and the log
    reports tokens per image.
  - **`<router>_exact`** (default: `loss_exact`): the router's scores with the exact allocation instead of the greedy.
    A small dynamic program over the tree finds the layout that maximizes the summed scores. The greedy never splits a
    32px patch whose own gain is small to reach large gains among its children.
- **attribution** at `--attr_tokens`:
  - by t: stitching only inside each t window, with the 16px model elsewhere;
  - by patch size: the same layouts with the 8px regions, or the 32px regions, rendered by the 16px model (what
    splitting buys, what merging costs).
- **regions:** each region's split gain per t (the error the finer model removes, split into structure + detail).
  Every heuristic's rank correlation with it: loss, i.e. how much of the realized gain is predictable from z_t; dino and its gate and divergence terms; pixel; disagree; centre; random.,
  also in Inception feature space, plus the heuristics' correlations with each other.
- **recon** (`--recon`): reconstruction FID from t0, including the loss-oracle that sees the true image.
- **samples, seams and flicker**, per configuration, from 32 samples (same seed and labels everywhere):
  - `_samples_K{K}.png`: each image, then with its final layout drawn;
  - seam ratio: boundary vs inside-token contrast, relative to the same layouts on the 16px model's samples, where
    1 means no seams;
  - flicker: the share of the image re-tokenized from one layout to the next within a trajectory.

- **heatmap** (`--heatmap_only=True`, about 15–30 minutes at 5k images): FID of the one-step clean estimate from held-out
  images noised to each t, for uniform and each router at each K.
  - Each model runs once per t, and every layout is composited from their predictions.
  - The figure `_heatmap.png` shows uniform's share of the 16px → 8px gap closed, and each router minus uniform on one
    shared scale.
  - It measures the estimate at each t, not samples, so it guides where budget and placement matter in t. The `--t_schedules`
    runs then test that with real sampling.
- **t-varying budgets** (`--t_schedules=lin+192,lin-192,switch,switchrev --schedule_tokens=160`): K changes with t while
  the average tokens per network call (compute per sample) is matched to `--schedule_tokens`.
  - `lin±d` varies K linearly by d tokens per unit t: `+` puts more tokens near the clean end, `−` near the noise end.
  - `switch` uses the 16px model alone, then the 8px model alone from the t that matches the average; `switchrev` is
    the reverse. These involve no spatial routing at all.

  Both run for `--schedule_routers` (default `loss_exact_2level,uniform`), so {router, uniform} × {constant,
  t-varying} separates time allocation from spatial allocation.
- **layouts** (`--layouts_only=True` for only these, about a minute): every method's layout on the same held-out
  images noised to each t in `--layout_ts`, side by side, in `_layouts_t{t}_K{K}.png`. Each method's agreement is
  measured against the measured-gain oracle under the same allocation rules (exact or greedy, levels, pooling).

  Run with `--fid=False` to get only these, at about a minute per configuration, e.g. on top of a finished FID run.
  The regions part also reports how much measured error the greedy vs the exact allocation removes. It also reports partial splits, a
  no-training lower bound for binary (16×8) splits: refining only the best half of a 16px patch for +2 tokens, next to
  whole-patch splits for +3, at each K. It also gives how much of a patch's gain sits in its best half or quadrant.

Results are saved after every generation, and a rerun with the same `--out` skips finished ones. Cost is in units of
the 8px model generating the same images: a stitched configuration is about 1.3 for the models, plus 0.25 for the
routing estimate and about 1 for DINO when the dino router is used.

```bash
P="runs/in1k_128/jit-b16_p32_icl0/<run>/ckpt_<iter>.pt,runs/in1k_128/jit-b16_icl0/<run>/ckpt_<iter>.pt,runs/in1k_128/jit-b16_p8_icl0/<run>/ckpt_<iter>.pt"
python analyze_patch_sizes.py --ckpts=$P --data_dir=/data/in1k_128                                  # everything above except recon
python analyze_patch_sizes.py --ckpts=$P --data_dir=/data/in1k_128 --routers=loss,dino,random --tokens=64,100 --attr_routers=loss   # cheaper
```

`plot_patch_sizes.py --json_path=in1k_128_patch_sizes.json` plots the main results to `in1k_128_patch_sizes_main.png`, in
three panels:
- FID vs tokens per router: measured points, with quadratic fits in log2 K over K = 64–256;
- the drop-in view: FID change vs uniform at the same K;
- the compute view: tokens saved vs uniform at the same FID.

### Comparing tokenization families: `compare_tokenizations.py`

This compares how well each tokenization can describe held-out images at a matched token count K, with every token
given the same small capacity: a constant, linear or quadratic polynomial per colour channel. No diffusion model is
involved; it's a filter before any architecture work. Each family is allocated on the image itself, so it measures what
each family could represent, not a router.

| family | regions |
|---|---|
| `uniform` | content-blind coarse-first, centre-out layout |
| `quadtree8`, `quadtree4` | aligned squares, with 8px or 4px leaves |
| `merge` (B) | irregular regions grown bottom-up from 2×2 px units |
| `gauss` (C) | soft 2D Gaussian tokens with normalized blending, fit by Adam from the quadtree solution |

```bash
python compare_tokenizations.py --data_dir=/data/in1k_128     # -> in1k_128_tokenizations{.json,.png,_viz.png}
```

## Gaussian-placed tokens (GaussJiT)

`model_gauss.py` is a separate model, so the baseline is untouched. It runs JiT-B's backbone on K tokens that are
oriented, scaled patches (64 at 128px for `GaussJiT-B/16`).
- **Geometry:** each token is anchored on a grid of spacing `patch_size`. A small conv placer (about 0.1M parameters)
  predicts each token's centre offset, per-axis scale and rotation from the noisy input, in the same forward pass, trained
  only by the denoising loss.
  - Offsets are bounded to ±0.25 of the spacing and scales to ×1.33^±1 per axis.
  - Frames are axis-aligned by default. This keeps the basis separable, so encoding and decoding are two small matrix
    multiplies per token. `--gauss_max_rotation=0.785` allows ±45° rotation.

  On CUDA, the input and output layers run as fused Triton kernels (`gauss_kernels.py`). They're registered as custom ops,
  so `torch.compile` fuses everything around them.
  - Each token's kernel covers the whole layer: frame coordinates, masks, pixel reads, basis, Gaussian weights and
    accumulation into the image.
  - The backward passes return each token's 5 geometry gradients directly, so nothing per pixel is stored.
  - Each token visits only its own bounding box: its frame for the encoder, and for the decoder its Gaussian out to weight
    1e-3 (but at least as far as the pixels it may own, so coverage holds).
  - Axis-aligned frames use separable kernels (row, then column matrix multiplies). Rotated frames evaluate the 2D basis
    per pixel, which costs about n² per pixel instead.
  - `--gauss_fused=never` uses the chunked PyTorch path instead. It's slower and uses more memory.

  Measured on an L40S (128px, bf16, compiled, forward and backward, `--gauss_scale_range=3`):

  | model | batch 256, frames at init | batch 256, frames at max scale | batch 1024, max scale | peak memory at 1024 |
  |---|---|---|---|---|
  | JiT-B/16 | 94.8 ms | | 401 ms | 30.0 GB |
  | GaussJiT-B/16, axis-aligned (fused, the default) | 106 ms | 115 ms | 486 ms | 32.3 GB |
  | GaussJiT-B/16, axis-aligned, `--gauss_fused=never` | 168 ms | 168 ms | out of memory | |
  | GaussJiT-B/16, rotated | 137 ms | 274 ms | 1104 ms | 32.3 GB |

  The cost tracks the frames' size: "max scale" pins every frame at ×3, the worst case.

  `python gauss_kernels.py` checks the kernels against PyTorch references. Errors are about 1e-3 on GPU (TF32 matrix
  multiplies, like the rest of the bf16 model) and about 1e-7 under `TRITON_INTERPRET=1` on CPU.
  - The placer's last layer is zero-initialized, so training starts from a regular grid of overlapping square frames,
    1.5× the spacing.
- **Tokens are continuous functions,** on a cosine basis over each frame, so no pixel is ever resampled. There's no
  bilinear interpolation, and the noise level a token sees doesn't depend on its sub-pixel position.
  - **Encoder:** projects the actual pixels inside each frame onto the basis (reflected past the image borders), followed
    by JiT's bottleneck embedding. At the initial aligned frame this is the orthonormal DCT of the patch: lossless, and
    white noise gives unit-variance coefficients.
  - **Decoder:** each token's final layer predicts coefficients, evaluated exactly at each pixel's centre.
  - **Blending:** pixels blend nearby tokens with Gaussian weights normalized over tokens, a local softmax as in Slot
    Attention, so every pixel is covered whatever the geometry.
  - **Memory:** both run in chunks of tokens and are recomputed in the backward pass.
- **Positions:** sin-cos and RoPE at the continuous centres, a shape embedding, and the area attention bias.
- **Logging:** the training log shows the batch's geometry (mean offset, scale range, mean rotation), also logged to wandb
  under `geometry/`. Sample grids get a `_geometry.png` with every token's core outline, which tiles the image like JiT's
  16px patches at initialization.

```bash
python train_jit.py config/jit_in1k_128.py config/model_gaussjit_b16.py --optimizer_type=muon
```

Options: `--gauss_max_offset`, `--gauss_scale_range`, `--gauss_max_rotation`, `--gauss_extent`, `--gauss_basis`, `--gauss_sigma`,
`--gauss_placer_width/depth`, `--gauss_area_bias`, `--gauss_shape_embed`.

The presets use `bottleneck_dim=288`, which keeps JiT's 6× compression of a token's input: the default basis (24×24 per
channel) has 2.25× a 16px patch's values, and 128 × 2.25 = 288. If you change `--gauss_extent` or `--gauss_basis`, rescale it the same
way.

The input and output layers run in fp32 for precision (TF32 matrix multiplies in the kernels), inside the kernels on CUDA or
chunked and recomputed in the backward pass on the PyTorch path. Peak memory is in the table above.

## PixelDiT baseline

`--model=PixelDiT-B/16` (also `-L/16`, `-XL/16`) trains PixelDiT (Yu et al. 2025, arXiv 2511.20645), implemented in
`model_pixeldit.py` as a drop-in network for the same denoiser, losses, EMA, Muon, FID evals, `eval_jit.py` and
`eval_linear.py`.
- **Patch level:** N DiT blocks (RMSNorm, RoPE, adaLN-Zero) on 16px tokens.
- **Pixel level:** M PiT blocks on 16-dim per-pixel tokens of the noisy image. Each has pixel-wise adaLN from the
  semantic tokens, token compaction (patch pixels flattened to the hidden width for global attention, then expanded
  back) and a per-pixel FFN.
- **Size:** parameter counts match the paper's B / L / XL (185 / 570 / 798M at 256px vs 184 / 569 / 797M).
- **Objective:** the default is this repo's x-prediction, for an apples-to-apples comparison with the JiT runs.
  `--pixeldit_prediction=v` uses the paper's velocity output.
- **REPA:** the paper trains with DINOv2 REPA (block 8, weight 0.5); the `--repa_*` flags reproduce that.
- **Diagnostics:** the attention statistics and windowed-attention ablation are JiT-only and skipped. Per-noise-level
  losses, band errors and probes on the patch-level blocks work. JiT-only options (AttnRes, adaLN-single, skip, DC-AE,
  masking, ...) are rejected.
- **Cost:** PixelDiT-B has 2x the parameters of the best JiT setup (184M vs 92M) and +85% activation memory at 128px.

## adaLN-single

`--adaln_single=True` swaps JiT's per-block adaLN projections for PixArt-alpha's adaLN-single. One projection of
the conditioning vector is shared by all blocks, and each block adds its own learned 6 x hidden table. The final layer
uses a table plus the conditioning itself. JiT-B/16 goes from 131M to 91M parameters. Initialization follows PixArt:
the shared projection starts at zero, and blocks start as identities through zero-initialized attention / MLP output
projections, since the gates now start at the random tables rather than at zero. With Muon, the shared projection is
orthogonalized per component, like the per-block adaLN matrices.

## Gate initialization

`--gate_init=pixart` (default `zero`) changes how gated branches start as identities. With JiT's adaLN-Zero every gate
starts at exactly 0, so at first only the gates learn; the branch's own weights get no gradient. With `pixart`, as
adaLN-single's blocks already do, gates, shifts and scales start at small random values (normal / sqrt(hidden)) and
each branch's output weight starts at zero, so the output weight learns from the first step. It applies to per-block
adaLN (blocks, skip head blocks, final layer), the DC-AE branches (zero-initialized second conv), the skip gate
(zero-initialized skip projection) and the gated bottleneck. `--adaln_single` blocks are PixArt-style either way. Run
with per-block adaLN, it separates adaLN-single's two changes: shared modulation vs initialization.

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

The mid-training probes are kept cheap: 10k train / 5k val images and one AdamW run per cell, with no lr sweep. For an
accurate number, run `eval_linear.py` on the checkpoint. It uses the full train/val split the run trained on (found from
its `config.json`), the DINOv2 head sweep (SGD, 13 lrs, 12,500 iters), and k-NN per block. It reads the same tokens
(null label, input noised to `--jit_t`; pick t from the probe heatmap), and the heads cover
{the probe blocks concatenated, a depth-attention mixture of every block} x {mean pooling, attention pooling} x lr
(single-block heads only with `--jit_depth_attn=False`; k-NN still runs per probe block):
- depth attention (`--jit_depth_attn`): per token, a softmax over depth of a learned query against each block's
  RMS-normalized state (AttnRes's form), so the probe picks its own mix of blocks. The best heads' depth weights are printed.
- attention pooling (`--jit_pools=mean,attn`): a learned query per attention head over the tokens, no projections.
- both queries start at zero (uniform over depth, exactly mean pooling), so any gain over the mean-pooled heads is learned.
- every JiT head puts a BatchNorm without affine in front of its linear layer (MAE's linear-probe protocol,
  `--jit_feature_norm`). JiT's intermediate states aren't normalized like DINOv2's outputs, and plain SGD on their raw
  scale barely trains: without it the best head sat at the top of the lr grid, far below the quick probes.

Results land in `{run}/eval/results_t{t}.json`, with a feature-set x pooling table printed at the end. The heads keep
every block's tokens per batch (roughly 8 GB of activations at 128px, batch 1024); at 256px lower `--batch_size` or drop a
pooling.
```bash
python eval_linear.py --ckpt=runs/in100_128/jit-b16_icl0/muon/ckpt.pt --jit_t=1.0
python eval_linear.py --ckpt=runs/in100_128/jit-b16_icl0/muon/ckpt.pt --jit_t=0.75 --jit_blocks=4,6,8 --knn=False
```

`ckpt.pt` is overwritten at every save. With `--keep_ckpt_every_epochs=N`, a milestone `ckpt_{iter}.pt` (EMA1 weights
plus what's needed to rebuild the model, about 0.5 GB for JiT-B) is kept every N epochs. `eval_jit.py` (e.g. CFG
sweeps on matched checkpoints of two runs) and `eval_linear.py` both accept it. Setting it equal to
`eval_every_epochs` lines milestones up with the training-time FIDs. It doesn't affect the run directory name, so it
can be turned on when resuming.

Training-time FIDs use no CFG, 10k samples and a fixed seed, so the same noise and labels are used at every eval and across runs.
They're for comparing runs with each other. Paper-comparable numbers need `eval_jit.py` with 50k samples and CFG.

## Samplers

`--sampling_method` (in `train_jit.py` and `eval_jit.py`):
- `heun` (JiT's default, 50 steps) and `euler` are deterministic ODE samplers.
- `ancestral` is stochastic: DDIM's eta family on JiT's path. With `--sampling_eta` 1 (default) it's DDPM's ancestral sampler,
  fresh noise every step; 0 is deterministic DDIM, the same step as `euler` except above t = 1 - t_eps, where JiT's clamped
  velocity shortens euler's step.

In theory a stochastic sampler's error is bounded by the prediction's L2 error, while an ODE sampler's also depends on how
the prediction reacts to its input (diagnostic 8). Fresh noise doesn't undo a prediction that keeps too much of its input,
though: on a Gaussian toy, over-copying late in sampling inflates the sample std under both, slightly more under ancestral.
Heun makes two network calls per step, so Heun-50 (99 calls) costs the same as 99 euler or ancestral steps. Compare
samplers at matched calls. On a Gaussian toy with the exact denoiser, at 99 calls the sample std comes out 1.9% low for
Heun-50 and for ancestral eta 0, and 3.5% low for ancestral eta 1 (few-step ancestral sampling under-disperses: 6% low at
50 steps). Every method ends with JiT's final euler step.

`--time_schedule=train_quantile` puts the steps at quantiles of the training logit-normal (the diffusion_forcing repo's
`'logit'` grid for its Euler sampler). `--ema=0` in `eval_jit.py` samples the raw weights instead of EMA1, with results
written under an `_raw` tag.
```bash
python eval_jit.py --ckpt=runs/in100_128/<model>/<run> --data_dir=/data/in100_128 --final_num_images=10000 --sampling_method=ancestral --num_sampling_steps=99 --sampling_eta=1.0
python eval_jit.py --ckpt=runs/in100_128/<model>/<run> --data_dir=/data/in100_128 --final_num_images=10000 --sampling_method=euler --time_schedule=train_quantile
```

## Data-efficiency diagnostics

With `diag=True` (default; needs the `--split=val` data and `fid.py` run after it), every FID eval also logs to wandb
and `{out_dir}/diagnostics/diag_{iter}.json`:

| # | what | wandb keys | reading it |
| --- | --- | --- | --- |
| 1 | loss per noise level on fixed train / held-out images (raw weights) | `noise_loss/*`, `noise_xmse/*`, `noise_gap/*` | slow high-noise (small t) bins: semantics bottleneck; slow low-noise bins: detail; per-bin gap: where it memorizes |
| 1b | skip pathway runs: the per-noise-level held-out loss of the trunk-only prediction (what the trunk trains on, never sampled), same images and noise as 1 | `noise_loss_trunk/*`, `noise_xmse_trunk/*` | the sampled head's gain over its own trunk at each noise level |
| 2 | precision / recall (k=3, Inception) | `val/precision`, `val/recall` | low recall with fine precision: coverage / data-limited |
| 3 | raw vs EMA weights (off by default, `--diag_raw=True`: a second full FID run) | `val/*_raw` | EMA much better: noisy gradients (batch size, variance reduction, EMA horizon) |
| 4 | grad norms, update/weight ratio per block (every `log_interval`), adaLN gates, attention entropy | `grad_norm/*`, `update_ratio/*`, `gate_*/*`, `attn_entropy/*` | blocks that barely update or are gated off; collapsed or uniform attention |
| 4b | locality: mean attention distance of patch queries (relative to uniform attention) and attention mass on the in-context class tokens, per block; windowed-attention ablation: held-out loss increase when attention is confined to non-overlapping k x k patch tiles in every layer (`diag_window_tiles`, default 1,2,4) | `attn_dist/*`, `attn_prefix_mass/*`, `window_loss_increase/*`, heatmaps | an increase near 0 at high noise (small t) means the model isn't using context beyond the tile where global structure is decided |
| 5 | k-NN + linear probe of mean-pooled tokens (blocks at 1/3, 2/3 and the end of the model, t=1, 0.75, 0.5, null label) | `probe/*` | whether / when semantic features emerge, and whether FID gains follow them |
| 6 | relative error per frequency band and noise level; sample vs real spectrum | `freq_err/*`, `spectrum_log_ratio/*`, heatmaps | capacity spent on (or missing) high frequencies; vs copying the input (`z / t`) and a Wiener filter on the same images and noise (`freq_vs_copy/*`, `freq_vs_wiener/*` at the cleanest t, model-vs-copy panel in `detail.png`): above 1 at low noise = detail it was given is lost |
| 7 | memorization: nearest-training-image distance of samples vs held-out images | `mem/nn_ratio`, `mem/near_dup_rate` | ratio < 1 or near-dup rate well above 1%: samples copy training images |
| 8 | input sensitivity: Jacobian of the prediction w.r.t. its noisy input per noise level (Hutchinson estimate, one probe per image, `diag_sensitivity_images` held-out images, default 256), for the sampled prediction and, in skip pathway runs, the trunk-only one on the same inputs | `sensitivity/{copy,gain,tweedie}_t*`, `sensitivity_trunk/*`, `trends/sensitivity.png` | copy: 1 = passes the input through, 0 = ignores it. gain: also counts a change spreading to other pixels (texture made out of the noise). tweedie: 1 = what an MMSE denoiser with the same error would have, above 1 = follows its input more than that. The held-out loss can't see these, but a deterministic sampler's error depends on them | Band-resolved too (`copy_band` / `gain_band` / `tweedie_band` per noise level x band in the JSON): under an ODE sampler a band whose copy runs above its Tweedie-consistent level stops shedding its noise |
| 9 | trajectory error: held-out images noised to t0 (`diag_trajectory_t0`, default 0.3,0.5,0.7,0.85) and sampled from there with the FID's sampler and EMA weights, vs the one-step error at the same z, per band; final image spectrum vs the clean image's | `trajectory/excess_from_t*_band*`, `trajectory/spectrum_from_t*_band*` | the one-step error is what the loss sees; excess > 1 is what the sampler adds by feeding the model its own outputs, which is what samples and FID see. Spectrum > 0: excess power (grain) |

In the windowed ablation the in-context class tokens may only attend to each other, so they can't relay image content between
tiles; a test confirms that perturbing one tile leaves every other tile's output exactly unchanged. The model never saw masked
attention in training, so part of any increase is distribution shift. Compare across noise levels, training time and methods
rather than reading absolute values.

JiT's convention is t=1 clean and t=0 pure noise. The eval runs two extra sample batches (raw weights) plus about 40k
instrumented forward passes, so it costs roughly 2-3x a plain FID eval; `--diag_raw=False` or `--diag=False` cut that.

## Diagnostics on finished checkpoints

`diagnose_ckpt.py` runs the one-step loss / band error, input sensitivity (overall and per band) and trajectory passes on
existing checkpoints, no training, on the same held-out images and noise for every run, and prints the runs side by
side (also `{run}/diagnostics/ckpt_diag_{iter}_{weights}_{sampler}.json`). Skip pathway runs get the trunk-only prediction too.
EMA1 weights by default (what's sampled); milestone `ckpt_{iter}.pt` files compare runs at matched iterations.
```bash
python diagnose_ckpt.py --ckpts=runs/in100_128/jit-b16_icl0/<run1>,runs/in100_128/jit-b16_icl0/<run2> --data_dir=/data/in100_128
```

## Training-trend figures

`plot_trends.py` turns the per-eval diagnostics into trends over training (x = images seen, so runs with different batch
sizes line up by compute), written to `{run}/trends/`:

```bash
python plot_trends.py runs/in100_128/jit-b16/base
python plot_trends.py runs/in100_128/jit-b16/base runs/in100_128/jit-b16/mask0.5 runs/in100_128/jit-b16_bn256/base --out=trends_compare
```

| figure | question it answers |
| --- | --- |
| `convergence.png` | FID (log-log), train / held-out loss, precision / recall, memorization: is progress steady, is it starting to copy? |
| `noise_levels.png` | held-out loss per noise level vs the first eval, recent improvement per level, train/held-out gap: which noise levels still learn |
| `detail.png` | per-band error at low and high noise, sample vs real spectrum: is fine detail improving or stuck (e.g. at the patch-bottleneck floor)? |
| `semantics_locality.png` | probe accuracy per block, windowed-attention loss increase, attention distance: do representations and global context use grow? |
| `optimization.png` | loss, gradient norm, update/weight ratio per parameter group (from `train_log.jsonl`): are some blocks barely learning? |
| `sensitivity.png` | input sensitivity (copy, gain, Tweedie ratio) per noise level over training, and the trunk-only head's in skip pathway runs: does the prediction lean on its noisy input more than its error justifies? |
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
