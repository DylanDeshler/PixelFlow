# Experiments

Every training experiment so far and every one queued, in order. FIDs are FID-10k (FID-5k on in100), CFG 1.0, Heun 50
steps, EMA weights: the in-loop eval, comparable within a dataset. Lower is better. Runs live under `runs/{dataset}/{model}/{run}`.
Status: **done**, **running**, **queued**, **dropped**, **proposed** (designed, not queued). Compute policy: stack one
change at a time on the best setup so far (Phase 6); no grids.

Goal (see the README): make end-to-end pixel diffusion (JiT) beat tokenizer + latent-DiT pipelines on sample efficiency,
speed and FID, with methods that work on a new dataset from scratch. Policy since 2026-11-02: no grids; each new run
changes one thing from the current best, so its effect is readable on its own.

## Phase 1: in100_128 architecture search (JiT-B/16, Muon, batch 256, no in-context tokens)

FID at training iterations (k = thousand iterations).

| # | run | change | FID | verdict |
|---|---|---|---|---|
| 1.1 | `base` | JiT-B/16 as released (AdamW) | 208.4 @5k, 149.5 @20k, 145.2 @25k | reference |
| 1.2 | `muon_attnres-full` | Muon + full Attention Residuals | 130.0 @20k, 113.9 @40k | kept |
| 1.3 | `muon_attnres-full_adalnsingle` | + adaLN-single | 122.7 @20k, 115.7 @25k, 105.0 @40k | kept (survived an fp32-promotion fix) |
| 1.4 | `..._adalnsingle_layernorm` | LayerNorm instead of RMSNorm | 122.6 @20k | ≈ equal; RMSNorm kept |
| 1.5 | `muon_attnres-full_bn768` | 768-d bottleneck | 143.0 @15k | worse |
| 1.6 | `muon_attnres-full_gatedbn` | gated bottleneck (full-rank bypass) | 145.4 @15k | worse |
| 1.7 | `muon_attnres-full_skip` | skip pathway, no stop-grad | 135.3 @20k, 118.7 @40k | worse |
| 1.8 | `..._skip_skipat-2` | skip injected 2 blocks before the end | 134.5 @20k | worse |
| 1.9 | `..._skip_skipdrop0.5` | skip dropout 0.5 | 134.5 @20k | worse |
| 1.10 | `..._adalnsingle_skip_skipsg` | skip + stop-grad | 127.0 @20k | worse than 1.3 |
| 1.11 | `..._skip_skipsg_skipfl` | + separate final layer for the skip branch | 121.3 @20k, 114.0 @25k, 107.5 @35k, 72.5 @195k | **best in100 config** |
| 1.12 | `..._skipfl_skipres` | skip as a residual correction | 122.5 @20k, 118.5 @25k | worse (grain) |
| 1.13 | `..._skipsg_skipdec-pixel` | pixel skip decoder | 147.1 @10k | worse (grain, seams) despite 30% lower t=.95 loss |
| 1.14 | `..._skipsg_skipdec-subpatch` | subpatch skip decoder | 129.9 @20k, 122.3 @25k | worse |
| 1.15 | `..._skipdec-subpatch_no-skipdecmod` | same, no adaLN in the decoder | 133.6 @20k, 88.2 @155k | worse; best held-out low-noise loss of any run |
| 1.16 | `..._skipfl_dcaestem_dcaehead_dcaetrunkhead` | DC-AE conv stem and heads | 123.9 @20k, 118.0 @25k | worse |
| 1.17 | `..._dcaetrunkhead_gateinit-pixart` | same, PixArt gate init | 132.0 @20k, 89.6 @95k | worse |

Lessons:
- Paths that change how the trunk trains on input detail hurt. Reconstructing through the 128-d bottleneck is what
  teaches the trunk.
- Heads that read the noisy pixels harder lose on FID even with a better loss, so per-step held-out loss doesn't
  predict FID for heads.
- Hypothesis: those heads follow their input more, which an ODE sampler can't correct.

## Phase 2: in1k_128 baselines and patch size (JiT-B/16, Muon, batch 1024, 20 epochs)

FID at epochs. Unless noted, the baseline for everything after this is 2.1, plain Muon with no other flags.

| # | run | change | FID @5/10/15/20 ep | throughput / notes |
|---|---|---|---|---|
| 2.1 | `jit-b16_icl0/muon` | **baseline** p16 | 129.1 / 100.8 / 87.9 / 81.0 | 2065 img/s |
| 2.2 | `jit-b16_p8_icl0/muon` | 8px patches | 129.6 / 89.1 / 73.1 / 65.6 | 529 img/s (3.9× time), batch 256 × 4 accumulation |
| 2.3 | `jit-b16_p32_icl0/muon` | 32px patches | 142.2 / 110.4 / 100.0 / 97.4 | |
| 2.4 | `..._attnres-full_adalnsingle_skip_skipsg_skipfl` | in100's best config (1.11), 80 epochs | 113.1 / 91.2 / 81.8 / 75.1; 61.7 @40, 53.4 @80 | best in1k FID so far |

Analysis (no training): `analyze_patch_sizes.py` and `compare_tokenizations.py`, as the no-training estimate for
fixed-budget quadtree allocation (`research_proposal_fixed_budget_quadtree_patch_allocation.md`). Outputs are in
`in1k_128_patch_sizes*.json/png`.

## Phase 3: Gaussian-placed tokens (GaussJiT-B/16, in1k_128)

| # | run | change | FID at epochs | verdict |
|---|---|---|---|---|
| 3.1 | `gaussjit-b16/muon` | learned frames, scale range 1.33 | 131.9 @5 | frames saturate at the scale bound within ~1k steps |
| 3.2 | `gaussjit-b16/muon_grot0.785` | + rotation | 133.4 @5 | settled on a global ~14° tilt |
| 3.3 | `gaussjit-b16/muon_gscale2` | scale range 2 | 83.1 @20, 77.1 @25 | ≈ p16 at ~20% more compute; geometry only a function of t |
| 3.4 | `..._gscale2_gplacer-convnext` | ConvNeXt placer | 84.1 @20, 78.3 @25 | no better: placer capacity isn't the constraint |
| 3.5 | `gaussjit-b16_bn128/muon_gbasis16_gscale2` | capacity-matched (16² basis, bn 128), 55 epochs | 81.8 @20, 75.2 @25, 62.7 @55 | matches p16 on FID, probes better, −14% throughput; frames shrink globally toward 16px rather than adapting |

Toy CPU tests showed the code can learn anisotropic shapes. A jointly learned shared linear map on the coefficients
blocked shape learning (a co-adaptation trap). Open: whether a basis smaller than 16 forces content-adaptive placement.

## Phase 4: loss weighting, batch size, bottleneck (`queues/in1k_128_weighting_batch.sh`, from 2026-10-29)

Each run changes one thing from 2.1.

| # | run | change | FID @5/10/15/20 ep | verdict |
|---|---|---|---|---|
| 4.1 | gradsnr (`analyze_grad_snr.py`) | gradient SNR per t / frequency band, critical batch | — | critical batch grows ~20× over training (17 @5 ep → 250–600 @20 ep; p8 ≈110). Fine bands at high noise have clean gradients |
| 4.2 | `muon_lossf-wiener` | Wiener frequency weighting | 137.3 / 103.2 / 92.0 / 85.6 | **worse** at every eval (linear predictability undercounts what the network recovers) |
| 4.3 | `muon_losst-learned` | learned per-noise-level weight (EDM2, unnormalized) | 128.2 / 99.6 / 86.9 / 80.3 | **small gain**, 4/4 evals; best plain-Muon setup. Caveat: grad_clip active on most steps |
| 4.4 | `..._bs256` (linear lr) | batch 256, lr scaled linearly | 130.0 / 105.5 / 94.5 / 88.8 | worse |
| 4.5 | `..._blr0.0004_..._bs256` | batch 256, sqrt lr | 116.9 / 92.9 / 84.9 / 78.8 | better per image (fading: 12 → 2.2), ~30% slower per image |
| 4.6 | `jit-b16_bn768_icl0/muon` | 768-d bottleneck | 142.4 / 108.7 / 93.0 / 86.2 | worse; copy 2× (keep 128) |
| 4.7 | `jit-b16_bn768_icl0/muon_lossf-wiener` | bn768 + Wiener | 146.3 / 109.0 / 97.2 / 90.7 | worse; the penalties add up |
| 4.8 | `muon_lossf-wiener_losst-learned` | Wiener + learned weight | (result not collected) | done |
| 4.9 | calibrate (`calibrate.py`) | from-scratch noise and schedule calibration | warm-up stage: 108.1 @5 (P_mean −1.08 from data statistics, warm-up 1) vs 112.1 for 4.10 | stats / warm-up / SNR stages done (critical batch 17 at 5 ep). The P_mean sweep **crashed out of GPU memory again** (2026-11-02 10:59, another process held 33 GB); retry deferred |
| 4.10 | `muon_warmup1` (calibration reference) | 1-epoch lr warm-up (vs 5), 10 epochs | 112.1 @5, 94.4 @10 | **better** (vs 129.1 / 100.8) |
| 4.11 | frequency-table weighting (`queues/in1k_128_freq_table.sh`) | weights by what a trained model recovers | — | **dropped**: weighting is a weak lever, see 4.2 / 4.3. The table was built; the training run crashed out of GPU memory at startup and won't be redone |

## Phase 5: building on the learned weight (from 2026-11-02)

The current best plain-Muon setup is 5.1: `--warmup_epochs=1 --loss_t_weight=learned --loss_weight_normalize=False`, FID 76.1 @20 ep
(`runs/in1k_128/jit-b16_icl0/muon_losst-learned_no-lossnorm_warmup1`). Everything after it runs in Phase 6's stack.

| # | queue / step | change (vs) | status | question |
|---|---|---|---|---|
| 5.1 | `in1k_128_warmup.sh` · `warmup1_learned` | + 1-epoch warm-up (vs 4.3) | **done**: 109.7 / 91.9 / 82.5 / **76.1** @5/10/15/20; linear probe 5.10, recall 0.415 | **new best**. The gains more than add up: on top of warm-up 1, the learned weight gains 2.4 / 2.6 / 1.7 @5/10/15 (vs ~1 with the 5-epoch warm-up). kNN probe slightly lower (2.14 vs 2.32) |
| 5.2 | `in1k_128_warmup.sh` · `bn768_learned` | + bottleneck 768 (vs 4.6) | **dropped** (stopped by the GPU shutdown, not resumed) | would the learned weight rescue bn768? Can't change a decision: bn768 is 5.2 FID behind (4.6, 4.7), the learned weight is worth ~1–2.5, and the run isn't on the best setup (5-epoch warm-up) |
| 5.4 | `in1k_128_samplers.sh` (inference only) | Heun 50, Heun 50 + shifted grid, ancestral 100 steps at η 0.25 / 0.5 / 1, on 2.1, 4.6 and 4.3 | **done**: Heun 50 best on all three. Baseline 81.0 vs 86.2 (shift) / 85.3 / 85.0 / 85.3 (η 0.25 / 0.5 / 1); learned 80.4 vs 85.7 / 85.0 / 85.2 / 88.4; bn768 86.2 vs 86.9 / 86.8 / 87.1 / 89.9 | SiT's SDE gain doesn't transfer at CFG 1 / 20 ep / 128px: keep Heun 50. Higher η raises precision, lowers recall. bn768 loses least (0.6–0.9 vs 4–5), weak support for the copying hypothesis |
| 5.13 | calibrate retry | 4.9 with the GPU lock | deferred | from-scratch noise settings for a new dataset |

## Phase 6: stacking one change at a time (`queues/in1k_128_stack.sh`, from 2026-11-03)

One sequential queue. Every stage runs on the best setup so far (starting from 5.1, FID 76.1). If its FID-10k at 20 epochs
beats the current best, its change is adopted and the next stage builds on it. Decisions are replayed from the step
logs on every restart. The final best goes to `runs/queue_logs/stack_best.txt`.

| # | stage · step | change (on the best so far) | status | question / adoption |
|---|---|---|---|---|
| 6.1 | clip · `noclip_warmup1_learned` | `--grad_clip=0` | running: 109.0 @5 (best run 109.7); started in the earlier `in1k_128_clip.sh`, the stack reuses it | the learned weight clips every step in epochs 1–5 (51% after), the baselines 0% after epoch 1: is 5.1's gain the weight, or the clipping (a per-step gradient normalization)? Adopted if better |
| 6.2 | arch · `stack_arch` | `--attn_res=full --adaln_single=True` (the agreed pair) | queued | the in100 wins (145.2 → 115.7 at 25k). The in1k recipe with them trained at 1488 vs 2069 img/s: check the per-hour cost too |
| 6.3 | disp · `stack_disp_analysis` | `analyze_dispersion.py` on the current best (minutes) | queued | per block: how dispersed features already are, and the gradient share λ = 0.5 gives |
| 6.4 | disp · `stack_disp_b{3,6,9,all}_10ep` | Dispersive Loss (InfoNCE-L2, λ 0.5, τ 0.5) at block 3 / 6 / 9 / all, 10 epochs | queued | best depth; JiT's built-in encoder may push it later than the paper's depth // 4 |
| 6.5 | disp · `stack_disp_b{best}_20ep` | the screen's best, resumed to 20 epochs | queued | candidate for adoption |
| 6.6 | disp · `stack_disp_balance0.2_b{best single}` | EnCodec gradient balancer (share 0.2) instead of λ, at the best single block | queued | fixed gradient share vs fixed loss weight; the better of 6.5 / 6.6 is adopted if it beats the best |
| 6.7 | tokens · `stack_tokens_bench`, `stack_tokens_bench_mem0.5` | `bench_tokens.py` on the adopted architecture (with and without selective recompute) | queued | step time, img/s, peak memory, FLOPs; largest fine share with no accumulation |
| 6.8 | tokens · `stack_tokens` | `--token_sizes=16,8 --fine_frac=0.25` (loss-proportional allocation by t) | queued | how much of p8's gain (65.6 vs 81.0) at ~1.75× p16's FLOPs? |
| 6.9 | tokens · `stack_tokens_infer{0,0.1,0.5,1.0}` | 6.8's checkpoint at other sampling budgets (inference only) | queued | FID vs sampling compute from one model |
| 6.10 | tokens · `stack_tokens_matched_{N}ep` | the previous best continued to 6.8's GPU-hours | conditional (if 6.8 is adopted) | per-hour comparison (adoption is per image) |

## Proposed, not queued

- (Sampler: Heun 50 stays, 5.4.)
- If headline numbers move to CFG > 1: one ancestral η 0.25 check there (inference only); SDE gains in the literature
  appear mostly with guidance.
- 16/8 tokens with a budget other than 0.25, chosen from 6.7's memory limit and 6.8's FID-per-t-window plot.
- GaussJiT with a basis below 16, to remove the "shrink every frame" escape (3.5).
- Early small batch (from 4.5 and the growing critical batch in 4.1): a batch-size ramp. On hold, because the user
  prefers single-phase schedules.
