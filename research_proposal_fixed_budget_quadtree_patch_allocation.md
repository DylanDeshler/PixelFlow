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

## 7. Later Refinements (if the idea works)

* **Remove the external representation:** route from the adaptive model's own previous-step prediction $\hat{x}_0^{(t - \Delta t)}$, which is free at sampling time and removes the router baseline pass, with a non-DINO score.
  * **Score:** the split score must come from quantities that exist for unsplit patches, such as the pixel statistics of that patch in $\hat{x}_0$. Hidden states only exist for the current layout's tokens, so the divergence between a parent's (nonexistent) children cannot be read from them.
  * **Training stand-in:** training has no previous step, so it needs a substitute, e.g. a no-gradient pass at a slightly noisier $t$ on the same image.
* **Compute efficiency:** cheaper routers, routing every $N$ steps, and amortizing the allocation across the CFG pair and batch.

## 8. Implementation

`model_quadtree.py` (`--model=QuadJiT-B/8`, `config/model_quadjit_b8.py`) implements Sections 2–4. Run commands for every configuration in Section 5.1 are in the README section "Fixed-budget quadtree tokenization (QuadJiT)". The uniform floor ($K = 64$) and the dense ceiling ($K = 256$) run on the same code path with `--quad_router=uniform`, so the comparisons share position embeddings, heads and backbone.

