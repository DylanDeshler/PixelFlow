# Dispersive Loss in pixel-space JiT: why it failed, and what to try next

Scope: theory, an account of the three failed attempts, fixes, related auxiliary losses, prototypes (`disp_variants.py`,
CPU-tested) and a ranked experiment plan. No existing file was changed. Labels used below:
**[log]** = read from the queue logs or run diagnostics; **[CPU]** = computed here (scripts in
`~/.cache/cross_patch_scripts/`, or `python disp_variants.py`); **[paper]** = Wang & He 2025 (arXiv 2506.09027 v2) or
their official code (github.com/raywang4/DispLoss), checked during this analysis; **[hyp]** = hypothesis, not verified.

---

## 0. Summary

1. **Run 1 (plain `infonce_l2`, blocks 3 and 6) failed through the AttnRes routing loophole plus free scale growth.
   It is not just "feature inflation".** [log] The AttnRes mixing-weight plots at epoch 5 show the regularized block's
   output, i.e. the input of the next attention, took weight ~1.0 on the *patch-embedding source*, at every t: row
   `b4.attn` in the block-3 run, row `b7.attn` in the block-6 run. The baseline puts ~0 there. Most mixes up to that
   depth also read mainly the embedding, so blocks 1-3 (or 1-6) were largely bypassed. The embedding is the raw noisy
   patch through the 128-d bottleneck, so it is the most dispersed signal available per sample (8,192 noise dimensions per
   image). Its scale is free too: `x_embedder` is on AdamW with no weight decay, and every consumer is normalized. Pinning
   at log(1/B) required the block-3 output's batch-centered per-dimension std to grow from ~0.003 to ≥1.6 (≥500x). Block-4
   probes fell from 2.08/4.08 to 0.90/1.34 (kNN/linear %, t=0.5, 10 ep), and FID was 13 points worse.
2. **`infonce_l2n` spent nearly all of its range removing the batch-common component.** Normalizing each sample by its own
   RMS broke InfoNCE-L2's translation invariance. [log] In the best run the pooled block-8 outputs have a pairwise cosine
   of ~0.985 (pos@0.5 = 0.991), and the flattened block-3 outputs ~0.93. [CPU] Centering alone takes such features from
   -0.06 to -3.943 (optimum -3.949). Once centered, the loss can barely tell representations apart. Isotropic 768-d noise
   scores -3.939, a 128-d random code -3.889, and a class-hierarchical "semantic" code -3.92. The "-3.92 means the noise
   shortcut" reading is therefore **not supported**: -3.92 is what centered semantic features also score. Whether the runs
   took a noise route is open, and the checks are listed in §3.3.
3. **The real noise shortcut is structural.** Any single-view dispersion objective whose minimum is a spread-out batch is
   also minimized by per-sample noise. Noise enters pairwise distances only through the norms: for i ≠ j,
   E_e⟨f_i, f_j⟩ = ⟨μ_i, μ_j⟩ with μ = E_e f. That holds in latent SiT too. What is specific to JiT is that
   **x-prediction is built to discard noise**. A v/ε-predicting SiT must carry ε to its output anyway, so noise in its
   features is free. In JiT, noise in mid-level features is wasted capacity that later layers must remove. 84% of JiT's
   training samples have pixel SNR < 1 (t ~ logit-normal(-0.8, 0.8), median t = 0.31, SNR 0.2).
4. **Fixes, from cheapest to most principled:**
   - (a) Batch-standardize instead of per-sample normalize (`infonce_l2s`): translation- and scale-invariant, exactly the
     paper's loss on unit-variance features.
   - (b) Regularize a sublayer output, not an AttnRes mix (`SublayerCapture`), which closes the routing loophole.
   - (c) Compute distances only from cross-view products of two noise draws of the same image (`xview_infonce`). The
     noise-averaged features are then what gets dispersed: noise and t-only directions earn nothing in expectation, and
     their variance raises the loss. With `paired_views` it costs no extra compute (512 images x 2 noise/t draws per batch).
   - (d) The alignment half the paper omits: the noisier view predicts the cleaner view's features (`xview_distill`).
5. **Top 3 next experiments** (one change each, on the current best, 10 epochs against FID-10k 81.12):
   - **(E1)** `xview_infonce` on pooled block-8 outputs, paired views, gradient share 0.1 via the existing balancer.
   - **(E2)** `infonce_l2s` with the paper's settings (block 3, flattened, τ 0.5, λ 0.5), captured at block 3's MLP output
     rather than the block output.
   - **(E3)** `xview_distill` + `xview_infonce` at block 8 (asymmetric cross-noise contrastive).
   - Before any of them: a zero-training feature report on the baseline and the two resumed pooled runs (§7, step 0). It
     says whether there is anything left to disperse, and which route the pooled runs took.

---

## 1. Facts to anchor on

**Paper and official code [paper]:**
- InfoNCE-L2 with no positives on one block's flattened tokens: L = log mean_ij exp(-‖z_i - z_j‖²/(d τ)). No projection
  head, no normalization ("we do not apply normalization").
- Defaults are λ 0.5, τ 0.5, block depth//4. Robust over τ ∈ [0.25, 2] and λ ∈ [0.25, 1]: every setting beat the baseline
  (36.49 → 32.1-33.9, SiT-B/2, 80 ep). Placement barely matters (block 1: 33.64, block 3: 32.35, block 12: 33.06, all:
  32.05).
- **The loss is computed per device.** JAX: 16 devices x local batch 16. PyTorch: 8 x 32, with λ 0.25, τ 1 and the *last*
  block (`act[-1]`), FID 32.64. The applied feature is the DiT residual stream (`x = block(x, c)`).
- AdamW, lr 1e-4, β (0.9, 0.95), wd 0, no warm-up.
- Table 1: the *contrastive* variants with positives from independent noise were 14-20% **worse** than no regularizer;
  "restricted noise" was neutral to slightly better. Dispersive was best (InfoNCE-L2 -11%; covariance only -1.8%).
- Fig. 3: the regularized block's representation norm grows, and the paper links this to the gain.
- Tested only on latent DiT/SiT, MeanFlow and a CIFAR-10 UNet; no pixel-space or x-prediction models.

**This repo [log]:**
- Best run (attnres-full, adaLN-single, learned weight, Muon, no clipping): FID-10k 95.03 / 81.12 / 73.33 / 68.17 at
  5 / 10 / 15 / 20 epochs.
- Probes at t=0.5, 10 ep (kNN/linear %): b4 2.08/4.08, b8 2.98/5.68, b12 1.12/2.90.
- Learned weight e^-u(t) at 10-20 ep: 9.0 (t=0.05), 13.3, 19, 22.1 (t=0.3), 22.5 (t=0.5), 15.2 (t=0.7), 3.5 (t=0.9).
  Zero-initialized, so e^-u = 1 at step 0.
- Logged "loss" is the weighted diffusion term only (`loss_parts['skip']`). At equilibrium it is ≈ 1 + E_t log E[L|t], so
  a gap of Δ between runs is ≈ a relative change Δ in the diffusion loss.

| run (window means) | iters 400-800 | 1400-2000 | 2000-2300 | 5000-6000 | 11000-12500 |
|---|---|---|---|---|---|
| baseline | -1.298 | -1.799 | -1.844 | -1.955 | -1.994 |
| l2, b3, λ .5 | +0.337 | +0.148 | +0.105 | +0.030 | +0.012 |
| l2, b6, λ .5 | +0.352 | +0.155 | +0.119 | +0.045 | - |
| l2n flat, b3, λ .5 | +0.083 | +0.014 | +0.008 | - | - |
| l2n pooled, b8, λ .5 | +0.175 | +0.053 | +0.037 | - | - |
| l2n pooled, b8, λ .1 | +0.066 | +0.017 | +0.009 | - | - |

(Gaps vs baseline; positive = worse. From `train_log.jsonl`.) The l2 b3 run's training-loss gap shrank to 1.2% by 10 epochs
while its FID gap stayed at 13. Training loss is a poor proxy here; probes and FID are what count.

---

## 2. Theory

### 2.1 What the loss is, and its three regimes

The dispersive loss is the **uniformity** term of InfoNCE (Wang & Isola 2020) with a Gaussian potential:
L = log E_ij exp(-D_ij/τ). Its minimizer for normalized features is the uniform distribution on the sphere. The regression
is supposed to supply the "alignment" (which differences to keep). With D_ij = ‖z_i - z_j‖²/d there are three regimes,
set by the feature scale s relative to τ:

- **Collapsed (s² ≪ τ):** L ≈ -E_ij D_ij/τ = -2 tr Cov(z)/(d τ). This is pure variance maximization, with gradient
  ∝ (z_i - z̄). It inflates every centered direction in proportion to what is already there, with no preference for
  structure. Where downstream is scale-invariant this buys nothing, and the loss keeps asking for more. This is where the
  best run's blocks sit (RMS 0.010-0.016 at blocks 3-11, all positions = 1.000 at every τ in the l2 analysis).
- **Saturated (s² ≫ τ):** every exp(-D/τ) → 0, L → log(1/B), gradient ∝ e^{-D/τ} → 0 (pinned).
- **Working (D/τ ~ 1-8):** gradient concentrated on the closest pairs (soft hard-negative repulsion).

**Batch composition matters [CPU].** For isotropic unit-variance features at τ 0.5, the self term (i = j, D = 0) holds 78%
of the softmax mass in a group of 16, but 5% in a batch of 1024. The paper's per-device loss is therefore ~4.5x gentler
than one global-batch loss at the same λ. The split of repulsion between pair types is the same (same-class pairs get
~8x their frequency at τ 0.5 and ~50x at τ 0.25, in a class-hierarchy toy). Smaller τ means more pressure against
semantic neighbours, i.e. an anti-clustering bias.

### 2.2 Why it can help latent DiT/SiT (what is known vs. inferred)

- **Anti-collapse.** Regression doesn't penalize low-rank or collapsed intermediate features. Low SNR makes it worse: the
  MMSE target tends to the data mean, so features at high noise carry little inter-sample variation. A uniformity prior
  keeps inter-sample differences alive in the hidden space. Representation quality and generation quality are linked
  (REPA: aligning with DINOv2 accelerates SiT a lot; in this repo probes predict FID with ρ ≈ 0.9).
- **Regularization.** The paper's own framing: gains grow with model size.
- **Norm growth (paper Fig. 3) [hyp].** In a plain residual stream, a larger block-3 norm makes early features persist
  against later additions. With AttnRes this mechanism is largely redundant: any sublayer can re-read early sources
  directly.
- **The SiT regime [hyp].** The SiT residual stream at block 3 is O(1) per dimension (the patch embedding of unit-variance
  latents plus position embeddings), with B=16 per device. That puts the loss mostly in the saturated-to-working regime
  from the start: a weak barrier against pairs getting close, not a strong push. AdamW at 1e-4 with no weight decay makes
  any scale drift slow. Nothing in the paper says the loss stayed off its floor (no loss curves are reported).

What a good representation means for a denoiser: features that carry the posterior of x given z_t (global/semantic at high
noise, local detail at low noise). Per the law of total variance, Var f = Var_x E[f|x] (content) + E_x Var(f|x)
(noise-driven). **A single-view dispersive loss rewards both terms. Only the first is useful for an x-predicting denoiser.**

### 2.3 What changes in pixel-space JiT

| factor | SiT-B/2 (paper) | JiT-B/16, 128px (here) | consequence |
|---|---|---|---|
| prediction target | v (= ε - x up to scale) | x (v-loss on x-prediction) | v/ε nets must carry ε to the output: noise in features is free. JiT's design is to discard noise, so noise in features is pure waste. |
| t distribution | uniform t | logit-normal(-0.8, 0.8); median t 0.31 | pixel power SNR (t/(1-t))²: 0.06 at the 23rd percentile, 0.2 at the median, 1 at the 84th, 5.4 at the 98th. Most training inputs are noise-dominated. |
| noise dims entering | 4,096 per image (32x32x4) | 49,152 raw; 8,192 after the 128-d bottleneck | enough to make any batch mutually orthogonal in both cases (B=16 needs 16 dims). Availability is not the difference; cost is. |
| flattened feature dims | 256 x 768 = 196,608 | 64 x 768 = 49,152 | both ≫ B |
| loss batch | 16 or 32 per device | 1024 global | ~4.5x stronger repulsion at the same λ (§2.1) |
| scale of regularized tensor | O(1) residual stream | AttnRes mix, RMS ~0.01, all consumers normalized | collapsed regime; scale is a free (diffusion-null) direction |
| optimizer | AdamW 1e-4, wd 0 | Muon (blocks) + AdamW (embeddings, queries, gains), wd 0 | normalized updates move along the diffusion-null radial direction at full speed |
| loss weighting | plain | learned e^-u ~9-23, from 1 at init | λ is effectively ~10-20x stronger at step 0 than after warm-up |
| residual | additive stream | softmax mix over all earlier sources (incl. the embedding) | a loss on one block's output can be met by re-routing (observed, §3.1) |

**Noise share by t [CPU, approximate].** Assumptions: per-pixel content std 0.5, noise std (1-t)·0.5; 90% of patch content
energy inside the 128-d bottleneck subspace vs 128/768 of the (isotropic) noise; 30% of patch content variance survives
image-level averaging; noise independent across patches.

| t (training quantile) | 0.1 (4%) | 0.2 (23%) | 0.31 (50%) | 0.5 (84%) | 0.7 (98%) |
|---|---|---|---|---|---|
| noise share, token embedding | 0.94 | 0.75 | 0.48 | 0.16 | 0.03 |
| noise share, mean-pooled embedding | 0.44 | 0.13 | 0.05 | 0.01 | 0.002 |

So at the token level about half of what enters the network at a typical t is noise. After mean pooling, noise matters
only for the noisiest ~quarter of samples, unless the network amplifies it. Pooling raises the bar, but it doesn't close the
shortcut.

**The single-view shortcut, stated precisely.** For one noisy view, D_ij = ‖f_i‖² + ‖f_j‖² - 2⟨f_i, f_j⟩. The
cross-products have noise-free expectations. Noise enters only through the norms (the diagonal of the Gram matrix), where
it inflates all distances uniformly, i.e. *isotropically*. Isotropy is exactly what InfoNCE-L2, infonce_cos, hinge and the
per-dimension-standardized covariance loss reward.

[CPU] Batch-standardized features with content confined to 8 directions score -2.70. Adding per-sample noise of 1x / 3x the
content std takes the loss to -3.46 / -3.76: noise strictly helps. At a noise-dominated representation the gradient with
respect to the content scale is +0.001, i.e. no pull toward content. The loss is blind to whether a sample's variance is
signal.

**The learned weight changes λ's meaning.** The diffusion gradient per sample is ≈ ∇L / E[L|t] (relative loss), so it is
scale-free in the loss value. The dispersive term enters in absolute units. Measured at 20 epochs, λ 0.5 is ~10% of the
gradient into block 8 for pooled l2n, ~15-24% for flattened l2n at blocks 3-9, and ~0.1% for plain l2 at RMS 0.01. At
step 0, e^-u = 1, so the same λ had roughly 10-20x more relative weight [hyp: assumes gradient ratios similar at init]. A
10% share then becomes ~60%. **The dispersive term dominated the first few hundred steps**, which is when every run reached
its floor (step ~600). The existing balancer (`--disp_balance`) fixes the share against the diffusion branch's running norm
and removes this ramp.

**Normalized architecture + normalized optimizers.** If the diffusion loss is invariant to the scale of a tensor, its
gradient has no radial component (Euler: ⟨∇L, w⟩ = 0). A scale-dependent auxiliary term is then the *only* radial force,
and Adam/Muon turn even a tiny consistent gradient into steps of normal size. [CPU, toy] With a fresh-noise "diffusion"
gradient plus a fixed rank-1 auxiliary component at 0.1% (1%) Frobenius share, the accumulated update put 0.8% (2.7%) along
the auxiliary direction under SGD-momentum, 0.7% (2.6%) under Adam and 1.8% (8.7%) under Muon. Consistency amplifies small
terms ~8x for every optimizer; Muon adds ~2-3x more. Muon is a secondary factor. The decisive one is the diffusion-null
direction.

**AttnRes.** The hooked "block output" is `res.mix(2k)`: a softmax mix over the embedding and all earlier sublayer outputs,
read only by block k+1's attention. A loss on it gets two cheap handles: the mixing weights (pseudo-queries, on AdamW) and
the scale of a source with free scale (the embedding). Neither has to change what the blocks compute.

---

## 3. The failure modes

### 3.1 Run 1: plain `infonce_l2`, λ 0.5, τ 0.5, block 3 / block 6

**Evidence [log]:**
- At step 0, disp = -0.0065 (collapsed regime). By step 600 it was -6.87 and stayed at -6.93 for 10 epochs (floor -6.93).
- Gradient-norm transient: 104 (b3) / 120 (b6) at step 600, vs 44 in the baseline.
- Diffusion loss +15% at 1.4-2k steps, +3% at 5-6k, +1.2% at 10 epochs.
- FID-10k 105.97 / 94.48 (b3, 5 / 10 ep) and 111.06 (b6, 5 ep), vs 95.03 / 81.12.
- Probes (b3 run, 10 ep, t=0.5): block 4 0.90/1.34 vs 2.08/4.08; block 8 2.50/5.02 vs 2.98/5.68.
- b6 run, 5 ep: block 4 0.60/0.60 (near chance) vs 1.60/3.42.
- **AttnRes mixing weights at 5 ep:**
  - Block-3 run: row `b4.attn` (block 3's output) has weight ~1.0 on `emb` at t = 0.047, 0.58 and 0.95. Rows `b1.mlp` to
    `b4.attn` are mostly `emb`. Columns b1.attn-b3.mlp are dark, so few sublayers read those outputs.
  - Block-6 run: row `b7.attn` ~1.0 on `emb`, and rows up to b6.mlp read `emb` at 0.4-0.9.
  - Both runs also send ~0.5 into `b8.mlp`.
  - The baseline has ~0 on `emb` beyond row b1.mlp.

**Mechanism:**
1. In the collapsed regime the gradient is pure variance maximization.
2. The patch embedding is the most per-sample-dispersed source (raw noisy patch, 128-d per token; noise share 0.48 at
   median t), and its scale is free: AdamW, no weight decay, normalized consumers.
3. So the mix selected it and it grew until every pair was far apart.

Pinning needs (B-1)·mean exp(-D/τ) < e^{0.05} - 1, i.e. D/τ > ~10, i.e. a batch-centered per-dimension variance > 2.5 (std
> 1.6). The starting centered std was ~0.003: total RMS 0.011 with a ~0.93 common share. That is a ≥500x growth along a
direction the diffusion loss can't see. Blocks whose outputs got routed around stopped contributing, which explains the
collapse of the block-4 probes. The FID cost scales with how many blocks were bypassed: b6 was worse than b3.

**Verdict.** The failure came from three properties of our setup together: scale freedom, a raw-input source reachable by
re-routing, and a noise-rich input. It is not a test of the idea.

### 3.2 Run 2: `infonce_l2n` on flattened block-3 outputs

[log] Step 0: -0.539 (shared cosine ~0.87 [CPU fit]). By ~800 steps it reached -3.945 to -3.947, slightly *below* the
"all orthogonal" value of -3.9449, which only centered (mutually anti-correlated) features can reach. No eval was run
(the guard stopped it at 1.9 ep), so there are no probes or mixing plots. Diffusion loss +1.4% at 1.4-2k steps.

Reading:
- Most of the 3.4-nat drop is explained by removing the batch-common component. For flattened tokens this is mostly the
  shared, position-dependent pattern: sin-cos position embedding, biases, adaLN table shifts.
- After that the term has almost no gradient.
- Whether the residual spread came from noise, content, or the `emb` route is **unverified**.

### 3.3 Run 3: pooled `infonce_l2n` at block 8, λ 0.5 / 0.1 (now resumed)

[log] Step 0: -0.062 (cosine ~0.985, the same as the end-of-training baseline). Step 300: -1.98. Step 600: -3.84. Step
1800: -3.92. Diffusion loss +5.3% (λ 0.5) and +1.7% (λ 0.1) at 1.4-2k steps; λ 0.1 within ~1% by 2.1k steps.

[CPU] At B = 1024, τ 0.5, the loss **cannot discriminate** between the candidate end states:

| representation (unit-RMS per sample, pooled) | infonce_l2n |
|---|---|
| baseline-like: common component 98.5% of energy + isotropic rest | -0.060 |
| the same, batch-centered | -3.943 |
| isotropic noise, 768-d / 128-d / 32-d | -3.939 / -3.889 / -3.713 |
| class hierarchy (cos 0.2 in superclass, 0.5 in class) | -3.923 |
| label embedding only (10% dropped share one null vector) | -3.497 |
| optimum (all pairs orthogonal) | -3.949 |

The trajectory is explained by common-mode removal plus "some ≥100-dimensional spread". The value -3.92 is consistent with
semantic features, noise, or a mix. The guard message "found the noise shortcut" overstated what the value shows. A pure
label route can't reach -3.92 (the null-label samples cap it near -3.5), so non-label variation must be spread too.

**Likely outcome [hyp].** Over the remaining epochs the term contributes a small gradient (|g|·|z| about 1/4 of its initial
value [CPU]) pushing the existing variance toward isotropy. My expectation for 10-epoch FID-10k:
- λ 0.1: within ~±3 of 81.12.
- λ 0.5: 1-6 worse.

A clear win would be surprising, because the objective carries almost no information beyond "remove the common mode".

**What to look for:**
1. `disp` stays ≈ -3.92 to -3.93 with no trend. Expected; uninformative.
2. **AttnRes mixing plot, row `b9.attn`** (block 8's output): weight on `emb` well above baseline (~0) means the routing
   shortcut. Rising weight on a single early source means the same.
3. **Linear vs kNN probes at block 8.** The kNN probe uses cosine on *uncentered* features, so removing the common mode can
   raise kNN with no real representational change. The linear probe standardizes features and is immune. Trust linear;
   treat a kNN-only gain as an artifact.
4. Probes at blocks 4 and 12, against the baseline numbers in §1. A drop at block 4 suggests re-routing.
5. After the run, `feature_report` (§6) at block 8:
   - `common` should drop from ~0.985 toward 0.
   - **`noise` share vs baseline:** a large rise means the noise route was taken.
   - `label` share vs baseline: a rise means the label route.
   - `erank` vs `erank_x`: a large gap means the extra spread is noise.
6. Diffusion-loss gap at 5-6k / 11-12.5k iters vs baseline (-1.955 / -1.994).

---

## 4. Fixes

Each fix lists the mechanism, the expected effect, the risks and the cost. F1-F4 address the failures directly. F5-F11 are
the other levers requested.

**F1. Batch standardization (`infonce_l2s`).**
- **Mechanism:** subtract the batch mean, divide by one batch RMS, gradient through both; then the paper's InfoNCE-L2.
  - Translation-invariant, like the paper's loss. Scale-invariant, so its gradient has no radial component [CPU: cosine
    with z < 1e-4].
  - τ keeps its unit-variance meaning (typical pair D ≈ 2).
  - Equals repa `infonce_l2` exactly on zero-mean, unit-RMS features [CPU].
- **Expected effect:** no inflation, no common-mode chase, the loss in the working regime from step 0 (on the tiny JiT at
  init: -2.83 vs optimum -3.39).
- **Risks:** the noise shortcut and the AttnRes routing remain; it should be paired with F2.
- **Cost:** a B x B Gram matrix. Even flattened (1024 x 49,152) that is ~0.1 TFLOP per step, negligible against ~34 TFLOP
  for the model.

**F2. Regularize a sublayer, not a mix (`SublayerCapture`).**
- **Mechanism:** a forward hook on `blocks[k].mlp` (ungated SwiGLU output). That is what block k computes and what every
  later mix can read. Mixing weights and the embedding scale can't satisfy the loss directly. The MLP's own input mix can
  still route toward the embedding, so the loophole is narrowed, not closed.
- **Expected effect:** the regularizer has to change computation.
- **Risks:**
  - The MLP output layer starts at zero, so scale-invariant losses see 0/0. A 1e-8 floor on the mean square bounds this.
  - Under `torch.compile` the hook sits inside the compiled `mlp_branch`; verify once against eager.
  - Doesn't work with the balancer's output replacement.
- **Cost:** none.

**F3. Cross-view distances (`xview_infonce`, `paired_views`).**
- **Mechanism:** the batch is the first 512 images twice. `flow_loss` draws independent noise *and* independent t per row,
  so rows i and i+512 are two views of one image. With K = A Bᵀ over centered views,
  D_ij = K_ii + K_jj - K_ij - K_ji, an unbiased estimate of ‖μ_i - μ_j‖², where μ averages over noise and t.
  - Noise never enters D in expectation, and its variance raises the loss (convexity of exp).
  - A direction that depends only on t cancels, because t is independent across the views and across images.
  - Label directions still count, since both views share the label.
  - Scale reference `cross` (default; the noise-free variance, floored at 0.1 x total) or `total` (noise and t variance
    included, i.e. an added invariance pressure).
- **[CPU checks]:**
  - Adding noise at 1x / 3x the content std moves the loss from -2.70 to -2.67 / -0.38 (single-view: -3.46 / -3.76).
  - At a noise-dominated representation the gradient toward content is -2.8 (single-view: +0.001).
  - A 64-dimensional random-Fourier function of t lowers the single-view loss by >0.3 but leaves `xview` unchanged within
    0.05.
- **Expected effect:** the only way down is to make the *content* of the pooled features more spread out. Content is
  dispersed less than f is (μ collapses at low t), so the loss has headroom and keeps a gradient. It shouldn't pin.
- **Risks:**
  - Halves distinct images per step. The diffusion still sees 1024 noise/t draws, but "10 epochs" becomes 5 unique-image
    epochs x 2 draws. Repeated-augmentation literature suggests this is near-neutral; unverified for diffusion.
  - Estimator noise per pair ≈ (noise/content variance)/√d: 0.04 at a 1:1 ratio for d=768, negligible pooled.
  - The `emb` route still gives low-level content (mean colour/texture) a noise-free reward. Watch the mixing plot, or
    combine with F2.
  - Not compatible with two-token-size models as is: allocation splits pairs across paths.
- **Cost:** same compute.

**F4. Centering.** Built into F1 and F3. On its own (centered `infonce_l2n`) it removes the common-mode chase. The
common mode may be functional (massive-activation / attention-sink-like directions, biases); a translation-invariant loss
leaves it alone instead of fighting it.

**F5. t-dependence.**
- *Gating* (`snr_weight`, sigmoid in logit t; t0 = 0.5 keeps ~16% of a batch). Disperses only where the image is visible.
  This shrinks the noise shortcut for single-view losses but doesn't remove it: the network can still amplify residual
  noise. It also drops 80+% of samples from the term.
- *Similar-t strata* (pairs only within logit-t bins). Removes t as a dispersal axis. In this model t enters through a
  low-frequency sinusoidal embedding (frequencies ≤ 1 rad per unit t), so t alone can fill only a few smooth directions:
  a minor axis compared with noise.
- F3 with independent-t views handles both, in expectation, at no extra cost. So gating is a second knob for whichever
  variant wins, not a first run.

**F6. Label (class) axis.**
- 90% of samples are labelled, and label embeddings start near-orthogonal. A pooled loss can be met partly by writing
  y_emb into features via adaLN shifts. This survives pooling and is shared by both views in F3.
- The probes use the null label, so this route is invisible to them.
- Options: compute the term only on label-dropped samples (~102 per batch; few); or accept it, since class information in
  block 8 may help conditional FID.
- Measure first: `label` share in `feature_report`.

**F7. Projection head + stop-gradient.** For dispersion, a head is counter-productive: it can expand features into a
dispersed space without changing the trunk (a cheaper shortcut). Heads belong on *alignment* terms (BYOL/SimSiam
predictors), as in F9.

**F8. Disperse the x-prediction or an EMA-teacher feature.**
- *Dispersing x̂ directly* is ill-posed: it competes with the regression and amounts to Contrastive FM (§5).
- *Re-encoding x̂, or using EMA-teacher features at clean t:* these have no input noise to exploit, but cost an extra
  forward. The `ema` copy exists, but running it is +30-60% compute.
- F3 gets the noise-free property without a second model.

**F9. Temperature.** With batch-standardized features τ is relative to the feature scale by construction (adaptive τ =
F1). Caveat: concentration of measure means high-dimensional noise makes all distances equal, which every
distance-uniformity objective rewards, whatever τ. Use τ 0.5-1 for first runs (paper: robust over [0.25, 2]). Smaller τ
focuses repulsion on semantic neighbours (same-class pairs get ~50x their frequency at τ 0.25).

**F10. Batch statistics (VICReg variance/covariance; `xview_rank`).**
- **Why `covariance` isn't safe as is:** the repo's `covariance` kind standardizes each dimension by its batch std. Per-dim
  independent noise inflates every std and drives all correlations to 0, so it is also noise-gamed.
- **The fix:** compute the covariance across views, C = Aᵀ B / n (noise-free in expectation), and penalize its
  concentration: L = log(d‖C_s‖²/(tr C)²), 0 for isotropic, log d for rank one.
- **[CPU]:** with noise 2x the content std, the cross-view loss stays at 2.23 vs 2.21 clean. The single-view version is
  fooled down to 0.29.
- **Character:** dimension-wise, so it has no per-sample shortcut. It asks for more directions, not more content. The
  paper's covariance variant was its weakest (-1.8%), so it ranks below F3.

**F11. Placement.**
- The probe peak is block 8 (kNN 3.42 / linear 6.56 at t=0.5, 20 ep). In the paper placement barely mattered, so put the
  term where the features to improve live. Pooled block 8 is what the probes read.
- The paper default (block 3, flattened) is the faithfulness check (E2).
- Avoid block 12: it specializes in pixels.
- Several blocks at once ("all" was the paper's best by 0.3 FID) multiplies the routing risk.

**F12. Weighting.**
- Prefer the balancer share (`--disp_balance`) over a fixed λ. It is invariant to the learned-weight ramp and to the
  loss's scale. Start at 0.1: about what λ 0.5 pooled l2n gave at block 8, and the paper's working range in share terms.
- With fixed λ, remember that λ 0.5 here is ~20x stronger at step 0 than at equilibrium.

**F13. Single objective, no phases.** Every variant here is one constant-weight term from step 0. F3's floor-free design
and F1's scale invariance remove the reasons a warm-up or schedule was tempting: pinning and the early dominance (with the
balancer).

---

## 5. Other losses to "disperse-ify"

| loss | relation to dispersion | fit for JiT | verdict |
|---|---|---|---|
| **Asymmetric cross-noise self-distillation** (`xview_distill`; cf. SRA, Jiang et al. 2025, which aligns a student block at higher noise with EMA-teacher features at lower noise) | the alignment half that dispersion lacks; together = InfoNCE with "same image, cleaner t" positives | uses `paired_views`; the pair's cleaner view is the stop-gradient target; DistillHead (identity + zero-init MLP) | **promising**. It is a denoising objective in feature space, not invariance: only the noisy side is pulled. The paper's symmetric independent-noise positives were 20% worse. Can collapse on its own, so pair with F3. |
| **Contrastive Flow Matching** (Stoica et al. 2025) | dispersion of predictions: push x̂_i away from other images' x_j | trivial: a loss on x_pred vs a permuted batch (`contrastive_x_loss`) | **wildcard**. [CPU-verified algebra] It moves the optimum to x̂* = m + (E[x\|z] - m)/(1-λ), i.e. it extrapolates the posterior mean away from the data mean, a built-in guidance against an unconditional "mean" model. Likely to lower cfg-1.0 FID for that reason, not through representations. May not stack with CFG. |
| **VICReg variance/covariance** | dimension-wise dispersion | `xview_rank` (noise-free version) | secondary (F10) |
| **EMA-teacher feature prediction (data2vec/BYOL)** | alignment; the teacher prevents collapse | needs a teacher forward (+30-60%) | only if `xview_distill` shows signal and an EMA target is worth the cost |
| **Self-REPA** (align block 4 to the EMA model's block-8 features at clean t) | REPA without an external encoder | as above; PixelREPA notes plain REPA hurts JiT long-term, and its fix is an adapter (MTA, already in repa.py) | lower priority |
| **Uniformity on x̂ re-encoded at t=1** | noise-free dispersion of what the model generates | second forward | costly; F3 is the cheap version |
| **Token masking** (masking.py) | not dispersion; forces context use | exists | orthogonal; could combine later |

Compatibility:
- All of these are single-run and need no external model.
- `xview_*` need pairs inside one forward. They don't fit the two-token-size path unless pairs are kept within a path
  (`path_aux` calls the term per path).

---

## 6. Prototypes (`disp_variants.py`)

Standalone. Nothing imports it. `python disp_variants.py` runs 26 CPU checks (all pass, ~4 s):

| function | claims tested |
|---|---|
| `infonce_l2s(hidden, tau, pool, weights, groups)` | scale invariance; invariance to a batch-common shift; no radial gradient; equals repa `infonce_l2` on standardized features; weights = masking; groups (floor log(1/16)); vs `infonce_l2n` on a cos-0.98 batch: -0.063 vs -3.777 |
| `xview_infonce(h_a, h_b, tau, pool, norm, weights, min_share)` + `paired_views`, `split_views` | never rewards per-sample noise (both norms); equals l2s on identical views; scale/shift invariance; pulls toward content at a noise-dominated representation (>50x l2s); t-only directions not rewarded with independent t |
| `xview_rank(h_a, h_b, pool)` | isotropic < low-rank; invariances; ignores noise where the single-view version is fooled |
| `xview_distill(h_a, h_b, t_a, t_b, head, pool, min_gap)`, `DistillHead` | gradient only into the noisier view; 0 for identical views; blind to batch-common offsets |
| `contrastive_x_loss(x_pred, x, t)` | stationary point (x - λ x_neg)/(1-λ) |
| `snr_weight(t, t0, width)` | values |
| `SublayerCapture(net, depth)` | captures [B, T, C] from a tiny JiT; gradient reaches the patch embedding |
| `feature_report(f1, f2, f_null)`, `block_features(net, x, y, t, blocks, noise_scale, null_label)` | runs on a tiny JiT; at init, block 2, t=0.5: common 0.94, noise share 0.48, erank 10.6 vs noise-free 7.9 |

### Plug-in snippets for train_jit.py (text only; not applied)

```python
# imports
from disp_variants import infonce_l2s, xview_infonce, xview_distill, DistillHead, paired_views, split_views, SublayerCapture

# config
disp_capture = 'block' # 'block' (repa.DispCapture, the block's output = an AttnRes mix) or 'mlp' (disp_variants.SublayerCapture)

# setup, replacing the DispCapture line (the balancer needs 'block')
disp_captures = [SublayerCapture(denoiser.net, b) if disp_capture == 'mlp' else DispCapture(denoiser.net, b, disp_balancer)
                 for b in disp_idx]
distill_head = DistillHead(denoiser.net.hidden_size).to(device) if disp_kind == 'xview_distill' else None
# (add distill_head's parameters to the AdamW group, and to the checkpoint)

# micro-step: paired batch for the cross-view kinds (flow_loss then draws independent noise and t per row)
Xd, Yd = paired_views(X, Y) if disp_kind.startswith('xview') else (X, Y)
loss_fm, t_used, y_used = flow_loss(denoiser, Xd, Yd, masker, token_mask, mask_weighting, return_aux=True, parts=loss_parts,
                                    weighter=loss_weighter, backward=path_backward, path_aux=...)

def disp_term(h):
    if disp_kind == 'infonce_l2s':
        return infonce_l2s(h, disp_temp, pool=disp_pool)
    if disp_kind == 'xview':
        return xview_infonce(*split_views(h), disp_temp, pool=disp_pool)
    if disp_kind == 'xview_distill': # alignment + uniformity
        a, b = split_views(h)
        ta, tb = split_views(t_used)
        return xview_distill(a, b, ta, tb, distill_head, pool=disp_pool) + xview_infonce(a, b, disp_temp, pool=disp_pool)
    return dispersive_loss(h, disp_temp, disp_kind, pool=disp_pool)
loss_disp = sum(disp_term(c.out) for c in disp_captures) / len(disp_captures)
loss = loss + disp_weight * loss_disp
```

Paired batches always take the `flow_loss` branch here (the learned weighter is on), so `t_used` exists. With
`masker` or `token_sizes`, don't use the xview kinds.

### Zero-training diagnostic (GPU box, scratch script, minutes)

```python
import numpy as np, torch
from jit_common import load_denoiser
from data_jit import MemmapImages, to_model_input
from disp_variants import block_features, feature_report
den, meta = load_denoiser('runs/in1k_128/jit-b16_icl0/<run>/ckpt.pt', 0, 'cuda')
net = den.net.eval()
val = MemmapImages('/data/in1k_128/val')
xs, ys = val.get(np.linspace(0, len(val) - 1, 1024).round().astype(np.int64))
x, y = to_model_input(xs, flip=False, device='cuda'), torch.from_numpy(ys).cuda()
for tv in (0.15, 0.3, 0.5, 0.75):
    feats = block_features(net, x, y, torch.full((1024,), tv, device='cuda'), range(1, 13), den.noise_scale, den.num_classes,
                           ctx=torch.autocast('cuda', torch.bfloat16))
    for b, f in feats.items():
        print(tv, b, {k: round(v, 3) for k, v in feature_report(*f).items()})
```

---

## 7. Experiment plan (ranked; one change per run on the current best)

Reference numbers for screening:
- FID-10k 95.03 (5 ep) / 81.12 (10 ep).
- Probes at t=0.5, 5 ep: b4 1.60/3.42, b8 2.44/4.78. At 10 ep: b4 2.08/4.08, b8 2.98/5.68.
- Diffusion-loss window means 1400-2000 = -1.799 and 5000-6000 = -1.955.
- FID-10k seed noise is unmeasured; treat < 1.5 as noise.

**Step 0 (free, do first).**
- Run `feature_report` on the baseline checkpoint, blocks 1-12, t ∈ {0.15, 0.3, 0.5, 0.75}.
- When the resumed pooled-l2n runs finish (or at their epoch-5 checkpoint), run it at blocks 4, 8, 12 and read their
  mixing plots (row `b9.attn`).

Decisions:
- If baseline `l2s` at a block is within ~0.1 of `optimum` while `xview` is far from it (≥ 0.5), single-view dispersion has
  nothing useful left there and F3 has headroom. That supports E1 over E2.
- Pooled-l2n runs:
  - A rise in `noise` share, or in the `emb` weight → shortcut confirmed.
  - Neither → the term was common-mode removal, and its FID effect is what centering does.
- `label` share large → consider F6.

**E1: `xview_infonce`, pooled block 8 output, paired views, `--disp_balance=0.1`, τ 0.5, norm `cross`.**
- Why first: it is the only form here where a falling loss implies more *content* spread. It can't pin; its optimum isn't
  reachable by noise, t or the common mode.
- Watch:
  - The xview loss should fall slowly over epochs and stay well above its optimum (no crash to -3.9 in 600 steps).
  - `b9.attn` weight on `emb`.
  - Linear probe at block 8.
- Continue to 20 ep if FID-10k at 10 ep ≤ 79.6 (≥1.5 better) with the block-8 linear probe ≥ baseline.
- Kill at 5 ep if FID > 98, or the linear probe at block 4 or 8 drops by > 0.5.
- If it wins: one paired-views-only run (no loss) attributes the gain. Paired views alone are a plausible confound.
  Skip it if the user prefers no controls; then attribute with probes and `feature_report`.

**E2: faithful replication. `infonce_l2s`, block 3, flattened, λ 0.5, τ 0.5, `disp_capture=mlp`.**
- Tests "the paper's loss, made scale- and translation-safe, at the paper's placement". The routing loophole is narrowed by
  the sublayer capture. The noise shortcut remains, as in SiT.
- Watch: the loss should sit in the working regime (between -3.0 and -3.9 at B=1024), not at the optimum; `emb` weights in
  rows b3.mlp / b4.attn; probe at block 4.
- Continue on the same FID rule.
- Expected [hyp]: this is the variant most at risk of the noise route, because block 3 sees token-level noise shares of
  ~0.5 at median t. A loss at -3.94 within 1k steps with a block-4 probe drop means it took that route. That would point
  to E1-style designs and settle the "setup vs idea" question for the single-view form.

**E3: `xview_distill` + `xview_infonce` (sum), pooled block 8, paired views, balancer share 0.1.**
- Adds the alignment half: noisier view predicts cleaner view. Closest to what made REPA/SRA work, without an external
  model.
- Run it after E1, since it builds on E1's term. If E1 is neutral, still run E3: the alignment may be the useful part.
- Watch: distill loss (1 - cos) should fall but not reach ~0 within 1 epoch (that would be collapse; check `erank_x`).
  Linear probes at blocks 8 and 4.

**Wildcard E4: `contrastive_x_loss`, λ 0.05 (ΔFM in x-units).**
- Cheap and orthogonal. Expect a cfg-1.0 FID gain from its guidance-like target shift, not better representations.
- Worth knowing because the repo evaluates at cfg 1.0. Check later whether it stacks with CFG before adopting it.

**Lower priority:**
- `xview_rank` at block 8, share 0.1. Run if E1 helps and you want to know whether dimension-wise spread suffices.
- `snr_weight` gating on the winner.
- `norm='total'` on the winner (adds invariance pressure).

**Don't run:**
- More `infonce_l2n` / `l2` variants, λ grids, or τ sweeps of the single-view forms.
- Multi-block placement before a single-block win.

---

## 8. Unverified points and caveats

- No checkpoint was available here, so nothing was measured on trained JiT features beyond the logged analyses. The
  noise-share table uses stated assumptions. `feature_report` (step 0) replaces it with measurements.
- The routing finding rests on the epoch-5 mixing plots of the two l2 runs. The l2n runs have no plots yet.
- The early dominance of the dispersive term (learned weight at 1 at init) is inferred from the end-of-training gradient
  shares, not measured at step 0.
- The Muon toy is a cartoon (random + low-rank gradients, n=256). It bounds Muon's extra amplification at ~2-3x in that
  setting, nothing more.
- SRA and Contrastive Flow Matching are cited from memory: their mechanisms, not their numbers.
- The paired-views batch (half the distinct images per step) has not been tested for its own effect on FID.
- `SublayerCapture` under `torch.compile` and with the balancer is untested on GPU.
- The pooled-l2n FID predictions in §3.3 are judgment calls, not derivations.
