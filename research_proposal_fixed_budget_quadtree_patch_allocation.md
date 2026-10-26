# Research Proposal: Fixed-Budget Quadtree Patch Allocation for Pixel-Native Diffusion Transformers

## 1. Executive Summary & Problem Formulation

Diffusion Transformers (DiTs) have established state-of-the-art generative performance, yet their computational foundation relies on a structurally inefficient assumption: **spatially uniform tokenization**. By carving inputs into rigid $p \times p$ grids, transformers assign identical computational weight to uniform low-frequency backdrops (e.g., clear skies, flat walls) and high-entropy, detail-dense regions (e.g., facial features, fine text, object boundaries).

For **pixel-native models like JiT (Just image Transformer)**, uniform tokenization introduces a steep scaling penalty. Generating images directly from pixels forces an unfavorable trade-off: either accept quadratic self-attention FLOPs ($O(N^2)$) with fine patches, or lose texture detail with coarse patches.

This proposal details an architecture and experimental methodology for **Fixed-Budget Quadtree Tokenization in Pixel-Native DiTs** on ImageNet-1k at $128 \times 128$. Rather than permitting variable token counts, which complicate batching and need extra loss regularization, we enforce an exact, static sequence length $K$ for every batch item. A semantic heuristic (frozen DINOv2 on a frozen baseline's clean-image estimate $\hat{x}_0$) redistributes the fixed budget to where structure is densest.

**Primary question:** at a fixed token count $K$, does semantically routed multi-scale tokenization give a better **FID** than uniform tokenization with the same $K$ and the same backbone? Every comparison is made on FID of sampled images (with IS, precision / recall and sample spectra as supporting metrics), at matched training images. Loss alone is not evidence: the method changes the input representation, so held-out losses are not comparable across tokenizations, and anything that cannot be sampled cannot be compared.

**Scope of this proof of concept:** the external DINOv2 router and the frozen baseline that supplies $\hat{x}_0$ are accepted here. If the idea works, later refinements will remove the external representation and the extra routing passes (Section 7).

## 2. Theoretical Framework: The Fixed-Budget Geometry

### 2.1 The Quadtree $+3$ Invariant

In a 2D quadtree any split replaces one parent node with four children:

$$
\Delta \text{tokens} = 4 - 1 = +3
$$

From a uniform base grid of $N_0$ macro-patches (Level 0), exactly $m$ splits give

$$
K = N_0 + 3m \iff m = \frac{K - N_0}{3}
$$

so for $(K - N_0) \equiv 0 \pmod 3$ **the sequence length is identical for every image in a batch**, whatever the tree topology.

### 2.2 Instantiation for $128 \times 128$ ImageNet-1k

**Hierarchy (first study): three levels, $p \in \{32, 16, 8\}$.**

* **Level 0 (coarse):** $32 \times 32$ patches, $N_0 = 4 \times 4 = 16$ tokens.
* **Level 1:** $16 \times 16$ children (the JiT-B/16 baseline's patch size).
* **Level 2 (fine):** $8 \times 8$ leaves (the dense ceiling's patch size).

A fourth level ($p = 4$) is deferred: it is only worth its cost (DINOv2 at $448$ px, Section 4.1) if the dense $p = 8$ ceiling shows clear headroom over $p = 16$, and at $K = 64$ it could cover at most about 5% of the image anyway.

**Budgets:** $K \in \{64, 100, 160\}$, i.e. $m \in \{16, 28, 48\}$ splits ($K = 256$ with this hierarchy is exactly uniform $p = 8$).

### 2.3 What a Fixed Budget Can Buy (Area Accounting)

Token areas must tile the canvas: $\sum_l n_l a_l = 128^2$ with $a_l \in \{1024, 256, 64\}$ px$^2$ and $\sum_l n_l = K$. The uniform $p = 16$ layout already spends all 16 splits of $K = 64$, so **every finer token must be paid for by coarser tokens elsewhere**:

| $K$ | max area at $p=8$ if the rest is $p=32$ | max area at $p=8$ if the rest is $p=16$ |
| ----- | ----- | ----- |
| 64 | 20% | 0% (only the uniform layout) |
| 100 | 35% | 19% |
| 160 | 60% | 50% |

So at $K = 64$ the question is purely one of *reallocation*: does sending part of the image to $p = 32$ to buy $p = 8$ detail elsewhere beat uniform $p = 16$? Larger $K$ tests whether adaptive placement buys more FID per token than uniform tokenization, bracketed by uniform $p = 16$ (64 tokens) and dense $p = 8$ (256 tokens).

## 3. Architectural Design

**Backbone:** the best in100/in1k configuration (JiT-B, full Attention Residuals, adaLN-single, Muon) **without the skip pathway**. A multi-scale skip would be a second variable-layout path to get right, and the skip's contribution shrinks over long training. The uniform baselines use exactly this backbone.

### 3.1 Position and Scale Embedding

A quadtree token must convey **where it is** (its centre), **how much image it covers** (its extent) and **how it relates to tokens of other sizes**. JiT's grid encodings (fixed absolute sin-cos plus 2D RoPE) assume integer positions and equal sizes, so each part is redefined. Each token has a physical centre $(c_x, c_y)$ in units of the finest grid ($8$ px, centres in $[0, 16]$), a width $w \in \{1, 2, 4\}$ in the same units, and a level $d \in \{0, 1, 2\}$.

**Default (four parts):**

1. **Relative position: 2D RoPE on the physical centres.** Rotation angles are $\omega c$ with JiT's frequencies, laid out for the $16 \times 16$ finest grid. Co-located tokens get the same phase at every level, so $q \cdot k$ depends only on true displacement, whatever the token sizes.
2. **Absolute position: footprint-integrated sin-cos.** JiT's separable 2D sin-cos, averaged over each token's box footprint. Per axis and frequency, in closed form:

   $$
   \frac{1}{w}\int_{c - w/2}^{c + w/2} \sin(\omega x)\,dx = \sin(\omega c)\,\mathrm{sinc}\!\left(\frac{\omega w}{2}\right), \qquad \mathrm{sinc}(u) = \frac{\sin u}{u}
   $$

   and likewise for $\cos$. This is Mip-NeRF's integrated positional encoding for box regions (Barron et al. 2021), equivalent to average-pooling a fine sin-cos grid over the footprint (the earlier Option C) but exact.
   * **Extent:** frequencies finer than the token are attenuated, so the embedding encodes extent as well as centre, with no parameters.
   * **Linearity:** because integration is linear, a parent's embedding is exactly the mean of its four children's.
3. **Scale: a learned level embedding $E_d$** (3 vectors, small init). The per-level patch embedders already give each level distinct statistics; this makes scale explicit and cheap.
4. **Proportional attention: $+\log s_j$ on the attention logits of key $j$,** where $s_j \in \{1, 4, 16\}$ is the token's area in finest-grid patches (ToMe's correction for merged tokens, Bolya et al. 2023).
   * **The problem it fixes:** a softmax weights every token equally, so without it a $32$ px token counts as $1/16$ of the same area tiled with $8$ px tokens.
   * **The effect:** with it, attention mass follows area.

**Split consistency.** When a region of uniform content is split:
* the children's absolute embeddings average exactly to the parent's;
* their total attention mass equals the parent's ($4 e^{x} = e^{x + \log 4}$);
* their relative phases agree with the parent's.

The representation therefore changes only where a split adds information. This matters because layouts shift between sampling steps when the router re-runs.

**Uniform baselines use the same scheme.** With a single level it reduces to sin-cos at grid centres with a constant area bias (which the softmax ignores), so the positional scheme is not a confound in the comparisons.

**Implementation note.** The area bias is a float attention mask, which moves `scaled_dot_product_attention` from the flash kernel to the memory-efficient one; negligible at $K \le 160$.

**Rejected:**

* **RoPE on level-scaled coordinates ($c \cdot 2^d$):** the same location gets different phases at different levels, which breaks relative position across scales.
* **RoPE with per-token frequencies scaled by size:** RoPE is relative only when both tokens use the same frequencies. With different frequencies, $q \cdot k$ depends on absolute position.

**Ablations (selected by FID and seams at scale boundaries):**

* **Option A, integrated (damped) RoPE:** the valid scale-aware RoPE. Keep the frequencies and scale each token's rotary pair by $\mathrm{sinc}(\omega w / 2)$ for its own width, on both $q$ and $k$. $q \cdot k$ becomes the relative rotation times each token's damping: still relative in position, while fine positional interactions fade for coarse tokens.
* **Option B:** sin-cos at the centre (not integrated) instead of the footprint-integrated embedding. This gives coarse tokens high-frequency components for a point they do not occupy, and no extent.
* **Without the area bias.**
* **Without $E_d$.**

### 3.2 Multi-Scale Patch Projection and Reconstruction Heads

* **Input:** one bottleneck patch embedding per level (JiT's $128$-d bottleneck; $p = 8$ patches are $192$ values, so still mildly compressed). For batching, every candidate patch at every level is embedded ($16 + 64 + 256$ small linear projections, negligible cost), then each image gathers its own $K$ tokens through an index buffer. Shapes are static, so one compiled graph serves all layouts.
* **Output:** one final layer per level (adaLN-single modulation). Each token's patch is scattered onto the canvas through the same index buffer. If seams appear at scale boundaries, overlap-add blending (already implemented for uniform patches as `head_overlap`) is the fallback.
* **Known risk, measured rather than assumed away:**
  * A $p = 32$ token's output is a $768 \to 3072$ linear map, so it can only produce a $\le 768$-dimensional subspace of each patch.
  * Regions left coarse lose texture. The routing heuristic deliberately avoids splitting stochastic textures (Section 4.1), so those regions are exactly the ones that stay coarse.
  * Per-level area statistics and sample spectra are logged to see whether this costs FID.

## 4. Semantic Routing Strategy: The Angular Divergence Heuristic

Routing uses a frozen DINOv2 evaluated on a clean-image estimate $\hat{x}_0$, with **Semantic Angular Divergence and PCA-1 Foreground Gating**.

### 4.1 DINOv2 Settings

* **Model:** `dinov2_vitb14_reg`. The register variants have cleaner patch-feature maps, which both the gate and the divergence rely on.
* **Inference:** $\hat{x}_0$ is upsampled from $128$ to $224$ px (bicubic) with ImageNet normalization, DINOv2's standard inference resolution. Each $14$ px DINO patch then covers exactly $8 \times 8$ px of the original image: the $16 \times 16$ feature grid matches the finest level of the hierarchy.
* **Child features:** each child quadrant's feature is the mean of the DINO patch features inside it.
  * Children of $p = 16$ patches are single DINO cells.
  * Children of $p = 32$ patches average $2 \times 2$ cells.
* **If the $p = 4$ level is added later:** DINOv2 runs at $448$ px ($32 \times 32$ grid), about $4\text{–}5\times$ the DINO compute.

### 4.2 Micro Semantic Divergence ($\mathcal{D}_{\cos}$)

For a candidate parent $P$ with child-quadrant features $\{\mathbf{f}_1, \dots, \mathbf{f}_4\}$:

$$
\mathcal{D}_{\cos}(P) = 1 - \frac{1}{6} \sum_{i=1}^{3} \sum_{j=i+1}^{4} \frac{\mathbf{f}_i \cdot \mathbf{f}_j}{\Vert{}\mathbf{f}_i\Vert{}_2 \Vert{}\mathbf{f}_j\Vert{}_2}
$$

This suppresses splits on stochastic textures (ocean, grass), whose features are collinear, and favors low-contrast semantic boundaries, whose features diverge.

### 4.3 Macro Semantic Gate ($g(P)$)

The dominant eigenvector $\mathbf{v}_1$ of the centered patch-feature matrix of each image is found by power iteration. The foreground gate for patch $P$ is $g(P) = \sigma\left(\frac{\mathbf{f}_P \cdot \mathbf{v}_1}{\tau}\right) \in [0, 1]$.

* **Sign:** $\mathbf{v}_1$ has an arbitrary sign, which is fixed per image so that the gate is larger on the foreground (e.g. by the border-vs-centre convention).
* **$\tau$:** chosen once on held-out images by inspecting gate maps, then kept fixed.

### 4.4 Combined Split Decision (Hierarchical Greedy)

* **Score:** $S_{\text{split}}(P) = g(P) \cdot \mathcal{D}_{\cos}(P)$.
* **Comparability across levels:** raw $\mathcal{D}_{\cos}$ is not comparable across levels (coarse quadrants average more), so scores are standardized within each level before comparison.
* **Allocation:** a priority queue over the current leaves. Start from the 16 Level-0 patches and repeatedly split the highest-scoring splittable leaf, adding its children to the queue, until $m$ splits are made. A child can therefore only split after its parent, and every layout is a valid quadtree with exactly $K$ tokens.
* **Layout output:** a per-image index buffer (token level and grid position), consumed by the tokenizer and the scatter.

### 4.5 Where $\hat{x}_0$ Comes From (Train and Sample Identically)

The layout is computed by the same procedure in training and sampling, so the network trains on the layouts it will see at sampling time:

1. **Frozen router baseline:** a fixed milestone checkpoint (EMA weights) of the in1k_128 JiT-B/16 best-configuration run predicts $\hat{x}_0^{\text{base}}$ from the same noisy input $x_t$ and $t$.
2. **Allocation:** DINOv2 on $\hat{x}_0^{\text{base}}$, then the greedy allocation of Section 4.4.
3. **Adaptive pass:** the adaptive model tokenizes the true noisy input $x_t$ with this layout and runs the backbone on exactly $K$ tokens.

In sampling, the router runs on the current $x_t$ at each routing step. The router baseline predicts with the null (unconditional) label, so the conditional and unconditional passes of a classifier-free-guidance batch, which share $x_t$ and $t$, get one layout.

**Noise dependence.** At high noise $\hat{x}_0^{\text{base}}$ is a blur and carries little routable structure. At those noise levels fine tokens also add little: in the in100 best run the one-step relative error at $t = 0.3$ is $0.6\text{–}0.86$ in the upper frequency bands. Two policies are compared:

* **Route at every $t$** (default).
* **Uniform below a threshold:** the uniform $p = 16$ layout for $t < t^*$ ($t^* \approx 0.5$), routed above it.

**Routing cadence (sampling only).** Ablations: re-route at every step vs every $N$ steps. A layout frozen from the first step (pure noise) is not a candidate: $\hat{x}_0$ there is close to the dataset mean.

## 5. Experimental Methodology & Evaluation Plan

### 5.1 Model Configurations

All models train on **ImageNet-1k ($128 \times 128$)** with class-conditional $x$-prediction, the same backbone, optimizer and schedule (batch 1024, as in the in1k_128 run), compared at matched training images.

| Model | Tokenization | $K$ | Purpose |
| ----- | ----- | ----- | ----- |
| **1. JiT-Dense** | uniform $p = 8$ | 256 | ceiling, and the headroom check |
| **2. JiT-Baseline** | uniform $p = 16$ | 64 | floor |
| **3. JiT-Adaptive (DINO)** | quadtree, DINOv2 on $\hat{x}_0^{\text{base}}$ | 64, 100, 160 | the method |
| **4. JiT-Random** | quadtree, uniformly random valid layouts | 64, 100, 160 | does the routing matter? |
| **5. JiT-PixelRoute** (optional) | quadtree, local gradient energy of $\hat{x}_0^{\text{base}}$ | 64 | a non-semantic heuristic: is semantic routing better than routing by visual detail? |

Every model can be sampled, so every model gets an FID. (A layout derived from the clean image is not included: it cannot be computed at sampling time, so it yields no FID.)

### 5.2 Core Comparisons (FID)

1. **Headroom:** Dense $p = 8$ vs Baseline $p = 16$. If the dense ceiling does not clearly beat the baseline at matched images, adaptive allocation inside $K \le 160$ has little to gain, and the study stops here.
2. **Reallocation at equal tokens:** Adaptive vs Baseline at $K = 64$.
3. **Routing value:** Adaptive vs Random at each $K$; optionally vs PixelRoute at $K = 64$.
4. **Budget scaling:** Adaptive at $K \in \{64, 100, 160\}$ against the $K = 64$ and $K = 256$ uniform anchors: FID vs tokens (and vs FLOPs).
5. **Position and scale embeddings:** the default (Section 3.1) vs integrated RoPE (Option A), centre sin-cos (Option B), and the default without the area bias or without $E_d$ (FID, boundary artifacts).
6. **Noise policy and cadence:** route-everywhere vs uniform-below-$t^*$; re-routing every step vs every $N$ steps.

### 5.3 Metrics and Reporting

* **Primary:** FID-10k (CFG 1, Heun 50 steps) at matched training images. The final checkpoints also get a CFG sweep with FID-50k at the best scale.
* **Supporting:** IS, precision / recall, sample power spectrum by frequency band, and a seam check at scale boundaries.
* **Routing statistics:** area per level as a function of $t$, and layout stability across sampling steps.
* **Compute:** training and sampling FLOPs and wall-clock *including* the router baseline pass and DINOv2. A "fixed budget" claim covers the backbone's token count; the routing overhead is reported next to it. Rough per-step cost at batch 1024:
  * Router baseline forward: about $+0.1$ s.
  * DINOv2-B at 224 px: about $+0.15\text{–}0.3$ s.
  * Together that is roughly $+50\%$ on a $\sim 0.7$ s step.

## 6. Staged Execution

1. **Headroom:** Dense $p = 8$ (`config/model_jit_b8.py`, i.e. `--patch_size=8`) vs Baseline $p = 16$, short runs at matched images. Go / no-go.
2. **Machinery:** multi-scale tokenizer, per-level heads and position embeddings, trained with **random** layouts at $K = 64$ against the uniform baseline. Does mixed-scale tokenization train cleanly, and which embedding avoids seams?
3. **Routing:** DINO-routed adaptive vs random (and optionally PixelRoute) at $K = 64$, both noise policies.
4. **Scaling and sampling ablations:** $K = 100, 160$; routing cadence.
5. **Optional:** add the $p = 4$ level (DINOv2 at $448$ px) if stage 1 shows headroom that the 3-level hierarchy cannot reach.

## 7. Learned Routing from Internal Representations (the End Goal)

The DINO router (Section 4) is a proof of concept: it depends on an external network and a frozen baseline's estimate. The end goal is a router that reads **the model's own internal representations** and is **learned during training**. Sampling would then need no extra network: the router reads the previous sampling step's states, which are already computed.

### 7.1 Design

The hierarchical greedy (Section 4.4) stays as the decoder, so every layout is still a valid quadtree with exactly $K$ tokens; only the split scores are learned. Per training step:

1. **Representation pass (no gradient).** The **EMA weights** run on the same $(z_t, t)$ as the training pass, with the neutral coarse-first layout (a small fraction of passes use random layouts), keeping every block's token states.
   * **Same $t$:** this follows self-conditioning (Chen et al. 2022), which trains with two passes on the same input and samples with the previous step's output. The offset between same-$t$ states in training and previous-step states in sampling is one step and, by that precedent, harmless.
   * **EMA weights:** sampling uses the EMA weights, so the router is trained on states from exactly the network that produces them at sampling time, and its inputs drift slowly.
2. **Candidate features.** Every 8 px cell takes the state of the token that covered it in the representation pass, and every candidate patch averages its cells. Every candidate at every level then has a feature, even where the representation pass tokenized the region differently.
   * A patch's hidden state is assumed to carry enough information to decide whether it should split.
   * The child-quadrant divergence of Section 4.2 was only a heuristic for the DINO phase.
3. **Router head.** A small head with **depth attention** over all blocks' states, in AttnRes's form and as in `eval_linear.py`'s depth-attention probes, plus the candidate's level, position and $t$.
   * **Why depth attention:** semantics peak around block 8 of 12, while where detail is missing may be better read from the late, pixel-near blocks, so the head chooses.
   * **Output:** the predicted error of the candidate's region at each patch size, $\hat{e}_l(P)$.
   * **Split score:** $\hat{e}_l(P) - \hat{e}_{l+1}(P)$, the predicted error reduction from tokenizing $P$ one level finer.
4. **Exploration.** A fraction of training samples use random layouts, so every region is regularly observed at every patch size. These are the counterfactuals the regression needs.
5. **Training pass.** The routed (or random) layout is trained with the usual denoising loss; nothing about the model's objective changes.
6. **Router regression** (a contextual bandit learned by plain regression: low variance, no reinforcement learning).
   * **Targets:** the training pass's per-region loss, detached.
     * A candidate covered by one token at level $l$ gives a sample of $e_l(P)$.
     * A candidate covered by its four children gives a sample of $e_{l+1}(P)$.
     * The head is fit only on the levels observed.
   * **Inputs:** the router reads detached states. Its gradients reach only the head, never the trunk, the same principle as `skip_stop_grad`.
7. **Sampling.** Step $i$ routes from step $i - 1$'s states (EMA weights, routed layout); the first step uses the neutral layout. With CFG, both halves share the conditional half's layout.

Cost: one EMA forward per training step (about $+30\text{–}35\%$), comparable to the current frozen-baseline pass but without DINOv2. Sampling costs nothing extra.

### 7.2 Training Difficulties and How the Design Handles Them

| difficulty | handling |
| ----- | ----- |
| Routing is discrete, so no gradient reaches the router from the denoising loss | Split-value regression (7.1, steps 3, 4 and 6). REINFORCE (single objective, high variance) and relaxations (Gumbel / perturbed top-$k$; soft coarse/fine mixing, which processes all 336 candidates) are fallbacks only. |
| Training has no previous sampling step | A same-$t$ representation pass (7.1, step 1), following self-conditioning. |
| The representation pass needs a layout before any routing exists | The neutral layout, with occasional random ones, so the head also sees states from mixed layouts, as at sampling. |
| Unsplit patches have no child features | Score from the patch's own (pooled) hidden state (7.1, step 2). |
| Router and denoiser could co-adapt into a locked-in layout | The exploration fraction keeps every region observed at every patch size, and the regression learns from those observations. |
| Router gradients could distort the trunk's representations | Detached router inputs; only the head trains on the regression. |
| Per-step loss isn't FID: a loss-driven router may chase copy-friendly texture | Validate on FID and the sample diagnostics (seams, detail per patch size) against the DINO and pixel routers. If needed, weight the regression target toward structure (low-frequency or mid-noise error). |
| Layouts can churn between sampling steps | EMA representations (7.1, step 1); split-consistent position embeddings and the area bias (Section 3.1); `--quad_route_every` if needed. |
| Routing inside the forward pass (early blocks on coarse tokens decide splits for the rest) | **Rejected:** semantic representations only emerge about two thirds of the way through the network, much too late to route from early blocks. |

The regression trains only a detached head: it never touches the denoiser's weights or objective, in the same way as the separate skip head.

### 7.3 Steps

1. **Remove the frozen baseline.** Route with DINO on the adaptive model's own estimate (its EMA weights, same $t$, neutral layout) instead of a separate baseline's.
2. **Learned router** (7.1): depth-attention regression head on the model's own states, with exploration. Compare against the DINO, pixel and random routers at equal $K$ and matched images, by FID.
3. **Compute:** routing every $N$ sampling steps, and a cheaper representation pass in training: full depth (the semantics are late) but at a coarse layout (e.g. $K = 16$, about a quarter of the cost), measured against FID.

## 8. Implementation

`model_quadtree.py` (`--model=QuadJiT-B/32-8`, `config/model_quadjit_b32_8.py`) implements Sections 2–4. Run commands for every configuration in Section 5.1 are in the README section "Fixed-budget quadtree tokenization (QuadJiT)". The uniform floor ($K = 64$) and the dense ceiling ($K = 256$) run on the same code path with `--quad_router=uniform`, so the comparisons share position embeddings, heads and backbone.


## 9. Noise-Aware Tokenization (Next Steps)

Sections 2–6 hold the tokenization fixed across noise levels: the router reads an estimate that gets sharper as $t \to 1$, but the budget, the decision rule and the patch embedding don't depend on $t$. Three further axes can depend on $t$ while the total compute stays fixed:

* **Where** the fine tokens go (Section 9.1).
* **How many** tokens each noise level gets (Section 9.2).
* **What** each token's bottleneck keeps (Section 9.3).

**Evidence so far.** On in1k_128, dense $p = 8$ (256 tokens) was level with $p = 16$ at 5 epochs (FID-10k 129.6 vs 129.1) and substantially ahead at 15. The headroom exists, but it appears only once detail starts to matter. The per-noise-level breakdown of that gain (held-out loss and per-band error by $t$, from the two runs' diagnostics, with no new training) is the first input to Sections 9.1 and 9.2.

### 9.1 Noise-Conditioned Patch-Size Decisions

What a split buys depends on $t$: splitting a patch adds detail at the frequencies its children resolve, and that detail is only recoverable where the signal at those frequencies exceeds the noise. Two variants:

* **Recoverability prior (no learned parameters).** Weight each candidate split by the Wiener gain of the band its children add:

  $$
  g_{l+1}(t) = \frac{S_{l+1}}{S_{l+1} + N(t)}, \qquad N(t) \propto \left(\frac{(1-t)\,\sigma}{t}\right)^2
  $$

  Here $S_{l+1}$ is the training images' mean power in the band resolved by level-$(l+1)$ patches and $N(t)$ the noise power in $z_t / t$. Allocation stays semantic (DINO), but the depth it may reach follows what is recoverable: coarse at high noise, shifting to $8$ px as the noise falls.
* **Scorer blend.** $w(t)\,S_{\text{DINO}} + (1 - w(t))\,S_{\text{pixel}}$: structure-driven when noisy, detail-driven when clean. The clean-image router visualization already shows the two scorers placing tokens differently.

Both are checked by eye first (noise-conditioned layouts in `viz_quadtree_router.py`'s noisy figure), then run as ablations of the DINO router at $K = 64$. Class conditioning of the decision is not pursued: the router already reads the image content, and the unconditional half of a CFG pair has no class.

### 9.2 Noise-Dependent Token Count at Fixed Total Compute

**Idea.** At high noise, fine tokens mostly carry noise, so move tokens from noisy steps to clean ones and keep the total compute fixed.

**Budget.** The cost per sample is $c(K) \propto 24 D^2 K + 4 K^2 D$ per block; at $D = 768$ the linear term dominates, so matching tokens matches FLOPs to within about 5%. A schedule $K(t)$ is held to the fixed-$K$ budget $\bar K$ in one of two ways:

* **training:** $\mathbb{E}_{t \sim p_{\text{train}}}[c(K(t))] = c(\bar K)$;
* **sampling:** $\frac{1}{N} \sum_i c(K(t_i)) = c(\bar K)$ over the sampling grid.

**The two budgets conflict.** $p_{\text{train}}$ (logit-normal, median $t \approx 0.31$) puts much more mass at high noise than a uniform 50-step grid, so one schedule cannot match both in general.

* **Example:** switch at $t = 0.5$ (84% of training mass below it), with $K = 40$ below and $K = 190$ above (both $16 + 3m$).
* **Training:** the expected budget is $0.84 \cdot 40 + 0.16 \cdot 190 = 64$ tokens, matching $\bar K = 64$.
* **Sampling:** on the uniform grid the average is about $115$ tokens per step, about $1.8\times$ the fixed-$K$ model.
* **Choice:** hold the training budget fixed and report sampling cost next to FID. Alternatively, sample on the `train_quantile` grid, where the budgets coincide, but that grid skips the clean end of sampling.

**Implementation (static shapes).**
* **Training:** split each batch into fixed-size $t$-buckets. Each bucket's size is its share of $p_{\text{train}}$, and its $t$ values are drawn from $p_{\text{train}}$ truncated to the bucket's range, so the overall $t$ distribution is unchanged. Run one forward pass per bucket at that bucket's $K$, then take one optimizer step.
* **Sampling:** each step uses $K(t_i)$.
* **Composes with any router:** the router allocates within each bucket's budget.

**Comparisons.**
* **Against:** fixed $K = 64$ at matched images, and dense $p = 8$ at matched compute.
* **Ablations:** switch point, number of buckets, and the $K$ ratio.
* **Expected signature if it works:** lower low-noise error and better sample detail at no extra training compute.

**Related work:**
* Pyramidal Flow Matching (Jin et al. 2024): noisy stages at lower resolution.
* FlexiDiT (Anagnostidis et al. 2025): larger patches for early denoising steps, one model.
* DyDiT: timestep-dependent width.

### 9.3 Noise-Conditioned Bottleneck

**What decides what passes.** The patch embedding is $h = W_1 x$ ($p^2 \cdot 3 \to 128$), then $W_2 h$. Whatever lies outside the row space of $W_1$ is discarded, identically at every $t$.

* **The fixed subspace is a compromise.** At high noise its high-frequency directions carry noise; at low noise a low-frequency-weighted projection drops detail.
* **FiLM after the bottleneck can't fix this.** It only rescales the same 128 features, so it cannot change what passes, and it largely duplicates the first block's adaLN.
* **The goal** is a $t$-dependent subspace **at fixed width**. This distinguishes it from the gated bottleneck, which widened the input path under conditioning and hurt FID and trunk probes.

**Options, in order of priority:**
1. **Mixture of projections:** $W_1(t) = W_0 + \sum_{k=1}^{4} \alpha_k(t)\, \Delta W_k$, with $\Delta W_k$ zero-initialized (the model starts exactly as today) and $\alpha(t)$ from a small MLP on the timestep embedding.
   * **What it can do:** blends of different matrices span different subspaces, so the kept information can rotate with $t$.
   * **Cost:** $K$ narrow projections mixed per sample, negligible next to the backbone.
2. **Fixed Wiener pre-filter (no parameters):** filter each patch in its frequency domain before projecting, $x \mapsto \mathrm{DCT}^{-1}\!\big(g(t) \odot \mathrm{DCT}(x)\big)$ with $g = S / (S + N(t))$, so the bottleneck compresses only what is recoverable. If it matches option 1, recoverability is the $t$-dependence that matters.
3. **Ordered (nested) bottleneck (no parameters):** channel $i$ is active only when $i < r(t)$, with $r$ growing as the noise falls (e.g. $32 \to 128$).
   * **What it forces:** the first channels must carry what every noise level needs (structure), later channels detail.
   * **What it tests:** it tightens the bottleneck at high noise, the most direct test of the finding that the bottleneck is what builds trunk representations.
4. **Alternatives:**
   * learned frequency-domain FiLM before the projection;
   * noise-band channel windows over a wider projection, as Denoising Task Routing (Park et al., ICLR 2024) does for a DiT's channels.

**Implementation and measurement.** Each option is a default-off flag on the patch embedding, parity-tested against the baseline. It applies to plain JiT (best config) and to each QuadJiT level's embedder. Measured by FID at matched images, per-noise held-out loss, and block-8 probes.

### 9.4 Parked: Bottleneck Width as a Ratio of the Patch Dimension

**The confound.** The $128$-d bottleneck compresses patches by different amounts at different sizes:

* $8$ px: $192$ values, $1.5\times$ compression;
* $16$ px: $768$ values, $6\times$;
* $32$ px: $3072$ values, $24\times$.

This confounds the dense $p = 8$ vs $p = 16$ comparison and makes QuadJiT's levels unequal in how much each token can carry.

**The fix.** Tie the width to a fixed compression ratio, for example $6\times$, giving $32 / 128 / 512$ at $8 / 16 / 32$ px. The runs are slow, so this waits for compute.

### 9.5 Order

1. **Finish the core study** (Sections 5–6).
2. **Per-noise breakdown of dense $p = 8$ vs $p = 16$** from existing diagnostics: where the gain lives sets the Section 9.2 schedule.
3. **Noise-conditioned bottleneck on plain JiT** (Section 9.3): independent of the quadtree, and the cleanest question.
4. **Recoverability prior or scorer blend in the router** (Section 9.1): by eye, then as an ablation.
5. **Noise-dependent token count at fixed training budget** (Section 9.2).
6. **Ratio-tied bottleneck** (Section 9.4).
