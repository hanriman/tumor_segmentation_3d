# 3D VISReg JEPA: Architecture & End-to-End Pipeline

*Authoritative architectural specification and end-to-end dataflow for the `thesis_3d` codebase.*

---

## 1. Executive Summary & Paradigm Overview

3D Joint-Embedding Predictive Architectures (JEPA) extend self-supervised representation learning to full multi-modal volumetric brain MRI scans (**BraTS 2024 GLI**). Unlike Masked Autoencoders (MAE), which reconstruct raw voxels and waste capacity modeling high-frequency Rician scanner noise and RF coil inhomogeneities, JEPA predicts abstract representations in latent space:

$$\hat{y}_{\text{tgt}} \approx y_{\text{tgt}}$$

### Methodological Synthesis (Thesis Innovation)
Prior non-contrastive self-supervised methods like LeJEPA (Balestriero & LeCun, 2025) and VISReg (Wu, Balestriero, & Levine, 2026) were evaluated primarily on 2D natural images. This thesis synthesizes **3D spatial masked target prediction (I-JEPA)** with **direct encoder-space VISReg geometric optimal transport regularization** (`SingleEncoderJEPA3D`), eliminating both EMA teacher networks and negative pairs in volumetric medical vision.

```
[Raw 4-Modality MRI: 240³] 
         │
         ▼  (Crop to brain bbox & trilinear resample)
[Canonical Volume: 4 × 128³] 
         │
         ▼  (PatchEmbed3D: Canonical 8³ voxels / Lightweight 16³ voxels)
[Canonical: 4,096 Tokens (16³ grid)  |  Lightweight: 512 Tokens (8³ grid)] 
         │
    ┌────┴──────────────────────────────────────────┐
    ▼                                               ▼
[Visible Context Tokens]                    [Anisotropic Target Slots]
(Canonical: N=1,536 / 37.5%)                (Canonical: N=776 / 18.9% across 6 cuboids)
(Lightweight: N=192 / 37.5%)                (Lightweight: N=97 / 18.9% across 6 cuboids)
    │                                               │
    ▼ (8-Layer Context Encoder: D=384)              ▼ (Shared Target Pass: torch.no_grad())
[Context Latents: D=384]                    [LayerNorm Ground-Truth Embeddings: D=384]
    │                                               │
    ├──► Tissue-Filtered Context Tokens ────────────┼──► VISReg / SigReg Regularization Loss
    │    (Z_tissue ∈ ℝ^{N_tissue × 384})            │    (Direct Encoder-Space Anti-Collapse)
    │                                               │
    ▼ (4-Layer Predictor: D=192)                    │
[Predicted Embeddings: D=384] ──────────────────────┴──► Smooth L1 Prediction Loss
```

---

## 2. Volumetric Preprocessing & Tokenization Contract

| Pipeline Step | Input Shape | Output Shape | Specification & Rationale |
| :--- | :--- | :--- | :--- |
| **Volume Ingestion** | $4 \times 240 \times 240 \times 155$ | $4 \times D_z \times D_y \times D_x$ | Native multi-modal MRI channels: T1, T1c, T2, FLAIR ($1.0\,\text{mm}^3$ spacing). Non-zero brain bounding box extracted. |
| **Resampling** | $4 \times D_z \times D_y \times D_x$ | $4 \times 128 \times 128 \times 128$ | Trilinear interpolation into canonical $128^3$ isotropic volume. Bounding box stretched to cube. |
| **Parenchyma Norm** | $4 \times 128^3$ | $4 \times 128^3$ | Z-score normalization computed strictly over non-zero parenchyma voxels ($\mu_{\text{nz}}, \sigma_{\text{nz}}$). |
| **Modality Dropout** | $4 \times 128^3$ | $4 \times 128^3$ | Zero-imputed channel dropout ($p=0.25$, fallback $\ge 1$ active, `rescale=False`) matching zero-imputed missing-modality evaluation. |
| **Canonical Patch Tokenization** | $4 \times 128^3$ | $4,096 \times 384$ | Non-overlapping $8 \times 8 \times 8$ 3D convolution (`kernel=8, stride=8`), projecting 512 voxels per patch into $D=384$. Grid is $16 \times 16 \times 16 = 4,096$ tokens. Used for primary benchmark. |
| **Lightweight Patch Tokenization** | $4 \times 128^3$ | $512 \times 384$ | Non-overlapping $16 \times 16 \times 16$ 3D convolution (`kernel=16, stride=16`), projecting 4,096 voxels per patch into $D=384$. Grid is $8 \times 8 \times 8 = 512$ tokens. Supported for fast local smoke tests. |
| **3D Positional Bias** | $N_{\text{tokens}} \times 384$ | $N_{\text{tokens}} \times 384$ | Additive frozen sinusoidal 3D coordinates: $[\text{PE}(z) \,\|\, \text{PE}(y) \,\|\, \text{PE}(x)]$, imparting explicit spatial geometry from step 0. |

---

## 3. Masking & Sampling Geometry

1. **Context Masking ($37.5\%$ visible)**:
   - Sampled via 3D Connected Breadth-First Search (BFS) starting from a random tissue-containing anchor patch.
   - Visible tokens are gathered into a compact sequence: $Z_{\text{ctx}} \in \mathbb{R}^{B \times 1536 \times 384}$ (canonical $8^3$ patch) or $\mathbb{R}^{B \times 192 \times 384}$ (lightweight $16^3$ patch).
   - Evaluating only visible tokens reduces self-attention compute by **$85.9\%$** compared to dense masking.

2. **Target Masking (Anisotropic Aspect-Ratio Jitter)**:
   - Implements 6 anisotropic cuboid target slots:
     $$\mathcal{S}_{8\text{-grid}} = \big[(3,3,3),\, (2,3,3),\, (3,2,3),\, (3,3,2),\, (2,2,2),\, (2,2,2)\big]$$
     $$\mathcal{S}_{16\text{-grid}} = \big[(6,6,6),\, (4,6,6),\, (6,4,6),\, (6,6,4),\, (4,4,4),\, (4,4,4)\big]$$
   - Target token volumes:
     - On canonical $16^3$ grid ($8^3$ patch size): $216 + 144 + 144 + 144 + 64 + 64 = \mathbf{776\text{ tokens total}}$ ($18.9\%$ of grid).
     - On lightweight $8^3$ grid ($16^3$ patch size): $27 + 18 + 18 + 18 + 8 + 8 = \mathbf{97\text{ tokens total}}$ ($18.9\%$ of grid).
   - **Proportional Tissue Filtering**: Candidates are accepted if $\text{tissue\_count} \ge \max(2, \lfloor 0.25 \times \text{vol} \rfloor)$. Prevents rejection deadlock on small slots while ensuring targets never sample empty background air.
   - **Multi-Aspect Fallback Bounds**: Fallback cuboid sizes are dynamically constrained within $[2, 3]^3$ (or $[4, 6]^3$ on $16^3$ grid), preserving 3D aspect diversity even under dense anatomical constraints.

3. **Physical Millimeter Invariance Across Resolutions**:
   - Both $8^3$ and $16^3$ patch tokenizations cover identical physical anatomical coverage: $48\text{ mm}$ macro cube, $32 \times 48 \times 48\text{ mm}$ planar slabs, and $32\text{ mm}$ regional meso cubes.

---

## 4. Architectural Specifications

### 4.1 Backbone Parameters (`SingleEncoderJEPA3D`)

Both `visreg_jepa` and `sigreg_jepa` use the unified [`SingleEncoderJEPA3D`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/src/brats_jepa_3d/models/single_encoder_jepa_3d.py) architecture. By default (`use_projector=True`), an auxiliary 2-layer Projector MLP ($384 \to 1024 \to 128$) maps tissue-filtered context latents before geometric optimal transport regularization, isolating the core ViT representations from excessive isotropic constraints:

| Component | Depth / Layers | Hidden Dim ($D$) | Attention Heads | MLP Dim | Parameter Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Context / Target Encoder (`ViTEncoder3D`)** | 8 | 384 | 6 | 1,536 | ~14.2M |
| **JEPA Predictor (`JEPAPredictor3D`)** | 4 | 192 | 6 | 768 | ~2.1M |
| **Auxiliary Projector MLP (`projector`)** | 2 | 384 → 1024 → 128 | — | — | ~0.5M |
| **Total Pre-Training (`SingleEncoderJEPA3D`)** | — | — | — | — | **~16.8M** |

### 4.2 Downstream Decoders & Probing Taxonomy

When pre-training completes, the JEPA predictor is **discarded**. The pre-trained Context Encoder is transferred to downstream segmentation via [`build_segmentation_model`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/src/brats_jepa_3d/models/builder.py):

1. **Multi-Scale 3D FPN (`multiscale`) — Pure Latent Probe**:
   - Lateral skip connections tapped from encoder layers $L_2, L_4, L_6, L_8$.
   - Scale-adaptive hierarchical decoding:
     - On canonical $16^3$ token grids (patch size $8^3$): Stage 1 operates at native token resolution ($16^3 \to 16^3$), followed by 3 progressive $2\times$ upsampling stages ($16^3 \to 32^3 \to 64^3 \to 128^3$), preserving output dimensionality.
     - On lightweight $8^3$ token grids (patch size $16^3$): 4 progressive $2\times$ transposed 3D convolution stages with `GroupNorm` ($8^3 \to 16^3 \to 32^3 \to 64^3 \to 128^3$).
   - Total model size: **~29.43M** parameters.
   - **Probing Role**: In frozen evaluation (`freeze_encoder=True`), FPN operates solely on extracted ViT latents without raw-image bypass, serving as the **pure latent probe** ($72.31\%$ Dice).

2. **Hybrid 3D UNETR (`unetr_hybrid`) — Frozen Hybrid Probe**:
   - Dual-stream architecture: pre-trained ViT lateral representations fused with a lightweight native convolutional stem (`stem_128` at $128^3$, `stem_64` at $64^3$) operating directly on the raw 4-channel input volume.
   - High-frequency edge maps from the stem are concatenated into decoder stages 3 and 4 ($64^3$ and $128^3$).
   - Dynamically supports both $8^3$ and $16^3$ token grids, maintaining channel and spatial compatibility.
   - Total model size: **~29.52M** parameters.
   - **Probing Role**: In frozen evaluation (`freeze_encoder=True`), UNETR combines frozen transformer latents with trainable high-resolution convolutional stems, serving as a **hybrid probe** ($75.13\%$ Dice) that demonstrates architectural complementarity.

---

## 5. Mathematical Loss Formulation

### 5.1 Total Pre-Training Objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \lambda_{\text{reg}}(t) \cdot \mathcal{L}_{\text{reg}}(\mathbf{Z}_{\text{tissue}})$$

### 5.2 Prediction Loss ($\mathcal{L}_{\text{pred}}$)
Smooth $L_1$ loss between predicted embeddings $\hat{Z}_{\text{tgt}}$ and LayerNorm-normalized stop-gradient target encoder representations $\text{LN}(\text{sg}(Z_{\text{tgt}}))$:

$$\mathcal{L}_{\text{pred}} = \frac{1}{|\mathcal{S}_{\text{tgt}}|} \sum_{i \in \mathcal{S}_{\text{tgt}}} \text{SmoothL}_1\big(\hat{z}_i, \, \text{LN}(\text{sg}(z_i))\big)$$

### 5.3 VISReg Regularization ($\mathcal{L}_{\text{VISReg}}$)
Three decoupled moment penalties over tissue-filtered encoder representations $\mathbf{z} \in \mathbb{R}^{N_{\text{tissue}} \times D_{\text{enc}}}$ ($D_{\text{enc}}=384$, Wu, Balestriero, & Levine, 2026, Algorithm 1):

$$\mathcal{L}_{\text{VISReg}} = \lambda_{\text{c}} \frac{\|\mu\|_2^2}{D_{\text{enc}}} + \lambda_{\text{sc}} \frac{\sum_{d=1}^{D_{\text{enc}}} (1 - \sigma_d)^2}{D_{\text{enc}}} + \lambda_{\text{sh}} \mathcal{W}_2^2\big(\mathbf{z}_{\text{std}}, \mathcal{N}(0, \mathbf{I})\big)$$

1. **Center Loss ($\lambda_{\text{c}} = 1.0$)**: Penalizes drift of the global representation mean $\mu$ away from the origin.
2. **Scale Loss ($\lambda_{\text{sc}} = 1.0$)**: Penalizes variance collapse along each encoder dimension ($1 - \sigma_d$).
3. **Shape Loss (Sliced-Wasserstein Distance, $\lambda_{\text{sh}} = 4.0$, $M = 1024$ slices)**: Standardizes tokens with detached moments $\mathbf{z}_{\text{std}} = (\mathbf{z} - \text{sg}(\mu)) / \text{sg}(\sigma)$, projects onto $M=1024$ random unit directions on $\mathbb{S}^{D_{\text{enc}}-1}$, and computes closed-form 1D Wasserstein-2 distance against standard normal quantiles in $O(N \log N)$ via sorting.

### 5.4 Regularizer Warmup & Annealing Schedule ($\lambda_{\text{reg}}(t)$)
$$\lambda_{\text{reg}}(t) = \begin{cases} 1.0 & t \le T_{\text{warmup}} \\ \lambda_{\text{min}} + \frac{1}{2}(1.0 - \lambda_{\text{min}})\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}} \pi\right)\right) & t > T_{\text{warmup}} \end{cases}$$

In canonical pre-training ($T_{\text{max}} = 200, T_{\text{warmup}} = 30, \lambda_{\text{min}} = 0.10$):
- **Warmup Phase ($t \le 30$)**: $\lambda_{\text{reg}} = 1.0$ enforces maximum geometric entropy and variance stabilization early, preventing informational collapse during initial ViT patch projection.
- **Annealing Phase ($t > 30$)**: $\lambda_{\text{reg}}(t)$ decays smoothly from $1.0$ down to a floor of $\lambda_{\text{min}} = 0.10$. This relieves the model from being over-constrained into an artificial isotropic standard Gaussian sphere, allowing target prediction loss ($\mathcal{L}_{\text{pred}}$) to dominate and form dense semantic anatomical clusters while retaining sufficient anti-collapse pressure throughout later epochs.

### 5.5 Learning Rate Schedule ($\eta(t)$)
$$\eta(t) = \begin{cases} \eta_{\text{base}} \times \frac{t}{T_{\text{lr\_warmup}}} & t \le T_{\text{lr\_warmup}} \\ \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{base}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t - T_{\text{lr\_warmup}}}{T_{\text{max}} - T_{\text{lr\_warmup}}} \pi\right)\right) & t > T_{\text{lr\_warmup}} \end{cases}$$

With $\eta_{\text{base}} = 5 \times 10^{-4}$, $\eta_{\text{min}} = 1 \times 10^{-5}$, and $T_{\text{lr\_warmup}} = 10$ epochs for $T_{\text{max}} = 200$. A linear warmup over the first 10 epochs prevents early self-attention gradient divergence, followed by cosine annealing decay down to $\eta_{\text{min}}$ over the remaining 190 epochs.

