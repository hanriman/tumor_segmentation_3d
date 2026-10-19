# 3D VisReg JEPA: Architecture & End-to-End Pipeline

*Authoritative architectural specification and end-to-end dataflow for the `thesis_3d` codebase.*

---

## 1. Executive Summary & Paradigm Overview

3D Joint-Embedding Predictive Architectures (JEPA) extend self-supervised representation learning to full multi-modal volumetric brain MRI scans (**BraTS 2024 GLI**). Unlike Masked Autoencoders (MAE), which reconstruct raw voxels and waste capacity modeling high-frequency Rician scanner noise and RF coil inhomogeneities, JEPA predicts abstract representations in latent space:

$$\hat{y}_{\text{tgt}} \approx y_{\text{tgt}}$$

To prevent representation collapse without relying on momentum teacher networks (EMA) or negative pairs, this pipeline employs **VisReg** (Variance-Invariance-Shape Regularization) and **SigReg** (Statistical Independence via Epps-Pulley Gaussianity tests).

```
[Raw 4-Modality MRI: 240³] 
         │
         ▼  (Crop to brain bbox & trilinear resample)
[Canonical Volume: 4 × 128³] 
         │
         ▼  (PatchEmbed3D: 16³ voxels per patch)
[512 Latent Tokens: D=384] 
         │
    ┌────┴──────────────────────────────┐
    ▼                                   ▼
[Context Tokens: N=192]        [Anisotropic Target Slots: N=97]
    │                                   │
    ▼ (8-Layer Context Encoder)         ▼ (Target Encoder Pass)
[Context Representations: D=384]    [Ground-Truth Embeddings]
    │                                   │
    ▼ (4-Layer Predictor: D=192)        │
[Predicted Embeddings: D=384] ──────────┼──► Smooth L1 Prediction Loss
                                        │
                                        └──► VisReg Regularization Loss
                                             (Center + Scale + Sliced-Wasserstein)
```

---

## 2. Volumetric Preprocessing & Tokenization Contract

| Pipeline Step | Input Shape | Output Shape | Specification & Rationale |
| :--- | :--- | :--- | :--- |
| **Volume Ingestion** | $4 \times 240 \times 240 \times 155$ | $4 \times D_z \times D_y \times D_x$ | Native multi-modal MRI channels: T1, T1c, T2, FLAIR ($1.0\,\text{mm}^3$ spacing). Non-zero brain bounding box extracted. |
| **Resampling** | $4 \times D_z \times D_y \times D_x$ | $4 \times 128 \times 128 \times 128$ | Trilinear interpolation into canonical $128^3$ isotropic volume. Bounding box stretched to cube. |
| **Parenchyma Norm** | $4 \times 128^3$ | $4 \times 128^3$ | Z-score normalization computed strictly over non-zero parenchyma voxels ($\mu_{\text{nz}}, \sigma_{\text{nz}}$). |
| **Modality Dropout** | $4 \times 128^3$ | $4 \times 128^3$ | Inverted dropout ($p=0.25$, fallback $\ge 1$ active) rescaled by $C / \text{keep}$ to preserve train/test magnitude. |
| **Patch Tokenization** | $4 \times 128^3$ | $512 \times 384$ | Non-overlapping $16 \times 16 \times 16$ 3D convolution (`kernel=16, stride=16`), projecting 4,096 voxels per patch into $D=384$. Grid is $8 \times 8 \times 8 = 512$ tokens. |
| **3D Positional Bias** | $512 \times 384$ | $512 \times 384$ | Additive frozen sinusoidal 3D coordinates: $[\text{PE}(z) \,\|\, \text{PE}(y) \,\|\, \text{PE}(x)]$, imparting explicit spatial geometry from step 0. |

---

## 3. Masking & Sampling Geometry

1. **Context Masking ($N_{\text{ctx}} = 192$ tokens, $37.5\%$ visible)**:
   - Sampled via 3D Connected Breadth-First Search (BFS) starting from a random tissue-containing anchor patch.
   - Visible tokens are gathered into a compact sequence: $Z_{\text{ctx}} \in \mathbb{R}^{B \times 192 \times 384}$.
   - Evaluating only visible tokens reduces self-attention compute by **$85.9\%$** ($192^2 / 512^2$) compared to dense masking.

2. **Target Masking (Anisotropic Aspect-Ratio Jitter)**:
   - Implements 6 anisotropic cuboid target slots:
     $$\mathcal{S} = \big[(3,3,3),\, (2,3,3),\, (3,2,3),\, (3,3,2),\, (2,2,2),\, (2,2,2)\big]$$
   - Target token volumes on $8^3$ grid ($16^3$ patch size): $27 + 18 + 18 + 18 + 8 + 8 = \mathbf{97\text{ tokens total}}$ ($18.9\%$ of grid).
   - **Proportional Tissue Filtering**: Candidates are accepted if $\text{tissue\_count} \ge \max(4, \lfloor 0.40 \times \text{vol} \rfloor)$. Prevents rejection deadlock on small slots while ensuring targets never sample empty background air.
   - **Multi-Aspect Fallback Bounds**: Fallback cuboid sizes are dynamically constrained within $[2, 3]^3$, preserving 3D aspect diversity even under dense anatomical constraints.

3. **Resolution & Patch Size Scaling ($16^3$ vs. $8^3$ Patches)**:
   - **Standard Resolution ($16^3$ patch, $8^3 = 512$ grid):** $N_{\text{ctx}} = 192$ ($37.5\%$), $N_{\text{tgt}} = 97$ ($18.9\%$).
   - **High Resolution ($8^3$ patch, $16^3 = 4,096$ grid):** Targets scale $2\times$ per axis to maintain identical physical millimeter volume:
     $$\mathcal{S}_{16\text{-grid}} = \big[(6,6,6),\, (4,6,6),\, (6,4,6),\, (6,6,4),\, (4,4,4),\, (4,4,4)\big]$$
     Yielding $N_{\text{tgt}} = 776$ tokens ($18.9\%$) and $N_{\text{ctx}} = 1,536$ tokens ($37.5\%$).
   - **Physical Millimeter Invariance:** Both configurations cover identical anatomical volumes ($48\text{ mm}$ macro cube, $32 \times 48 \times 48\text{ mm}$ planar slabs, and $32\text{ mm}$ regional meso cubes).


---

## 4. Architectural Specifications

### 4.1 Backbone Parameters (Matched to BraTS Dataset Scale)

The architecture is calibrated for 1,144–1,621 volumetric scans to prevent overfitting while fitting comfortably within $\le 8.1$ GB VRAM:

| Component | Depth / Layers | Hidden Dim ($D$) | Attention Heads | MLP Dim | Parameter Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Context Encoder** | 8 | 384 | 6 | 1,536 | ~14.2M |
| **JEPA Predictor** | 4 | 192 | 6 | 768 | ~2.1M |
| **Projector MLP** | 2 | $384 \to 1024 \to 128$ | — | — | ~0.5M |
| **Total Pre-Training** | — | — | — | — | **~16.8M** |

### 4.2 Downstream Decoders

When pre-training completes, the JEPA predictor and projector are **discarded**. The pre-trained Context Encoder is transferred to downstream segmentation:

1. **Multi-Scale 3D FPN (`vit_fpn`)**:
   - Lateral skip connections tapped from encoder layers $L_2, L_4, L_6, L_8$.
   - Scale-adaptive hierarchical decoding:
     - On canonical $8^3$ grids (patch size $16^3$): 4 progressive $2\times$ transposed 3D convolution stages with `GroupNorm` ($8^3 \to 16^3 \to 32^3 \to 64^3 \to 128^3$).
     - On high-resolution $16^3$ grids (patch size $8^3$): Stage 1 operates at native token resolution ($16^3 \to 16^3$), followed by 3 progressive $2\times$ upsampling stages ($16^3 \to 32^3 \to 64^3 \to 128^3$), preserving output dimensionality.
    - Total model size: **~29.43M** parameters.

2. **Hybrid 3D UNETR (`vit_unetr`)**:
   - Replaces pure transposed convolutions with a dual-stream architecture: pre-trained ViT lateral representations fused with a lightweight native convolutional stem (`stem_128` at $128^3$, `stem_64` at $64^3$).
   - High-frequency edge maps from the stem are concatenated into decoder stages 3 and 4 ($64^3$ and $128^3$).
   - Dynamically supports both $8^3$ and $16^3$ token grids, maintaining channel and spatial compatibility.
   - Total model size: **~29.52M** parameters.
   - Delivers superior performance at low data fractions (e.g. $+7.08\text{pp}$ over from-scratch UNETR at $5\%$ data) by resolving the coarse token boundary bottleneck.

---

## 5. Mathematical Loss Formulation

### 5.1 Total Pre-Training Objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \lambda_{\text{reg}}(t) \cdot \mathcal{L}_{\text{VisReg}}$$

### 5.2 Prediction Loss ($\mathcal{L}_{\text{pred}}$)
Smooth $L_1$ loss between predicted embeddings $\hat{Z}_{\text{tgt}}$ and target encoder representations $Z_{\text{tgt}}$:

$$\mathcal{L}_{\text{pred}} = \frac{1}{|\mathcal{S}_{\text{tgt}}|} \sum_{i \in \mathcal{S}_{\text{tgt}}} \text{SmoothL}_1\big(\hat{z}_i, \, z_i\big)$$

### 5.3 VisReg Regularization ($\mathcal{L}_{\text{VisReg}}$)
Three decoupled moment penalties over projected representations $u = \text{Projector}(z) \in \mathbb{R}^{N \times D_{\text{proj}}}$ ($D_{\text{proj}}=128$):

$$\mathcal{L}_{\text{VisReg}} = \lambda_{\text{c}} \frac{\|\mu\|_2^2}{D_{\text{proj}}} + \lambda_{\text{sc}} \frac{\sum_{d=1}^{D_{\text{proj}}} (1 - \sigma_d)^2}{D_{\text{proj}}} + \lambda_{\text{sh}} \mathcal{W}_2^2\big(\mathbf{u}, \mathcal{N}(0, \mathbf{I})\big)$$

1. **Center Loss**: Penalizes drift of the global representation mean away from the origin.
2. **Scale Loss**: Penalizes variance collapse along any individual latent dimension ($1 - \sigma_d$).
3. **Shape Loss (Sliced-Wasserstein Distance)**: Projects tokens onto $M=256$ random 1D hyperplanes, computes closed-form 1D Wasserstein distance against standard normal quantiles in $O(N \log N)$ via sorting, with $\mu$ and $\sigma$ detached so gradients strictly organize feature geometry.

### 5.4 Regularizer Schedule Annealing ($\lambda_{\text{reg}}(t)$)
$$\lambda_{\text{reg}}(t) = \begin{cases} 1.0 & t \le T_{\text{warmup}} \\ 0.10 + 0.90 \times \frac{1}{2} \left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}} \pi\right)\right) & t > T_{\text{warmup}} \end{cases}$$

In canonical 200-epoch pre-training on Google Colab A100 ($T_{\text{max}} = 200$), $T_{\text{warmup}} = 30$ epochs (defaulting to $15$ epochs for 100-epoch runs). Clamping $\lambda_{\text{reg}} = 1.0$ during the initial 30 epochs creates strong anti-collapse forces while latent representations are established; subsequent cosine decay down to $0.10$ across epochs 31–200 allows the predictor to refine subtle anatomical predictions without regularizer over-penalization.

### 5.5 Learning Rate Schedule ($\eta(t)$)
$$\eta(t) = \begin{cases} \eta_{\text{base}} \times \frac{t}{T_{\text{lr\_warmup}}} & t \le T_{\text{lr\_warmup}} \\ \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{base}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t - T_{\text{lr\_warmup}}}{T_{\text{max}} - T_{\text{lr\_warmup}}} \pi\right)\right) & t > T_{\text{lr\_warmup}} \end{cases}$$

With $\eta_{\text{base}} = 5 \times 10^{-4}$, $\eta_{\text{min}} = 1 \times 10^{-5}$, and $T_{\text{lr\_warmup}} = 20$ epochs for $T_{\text{max}} = 200$ ($T_{\text{lr\_warmup}} = 10$ for 100-epoch runs). A linear warmup over the first 20 epochs prevents early self-attention gradient divergence, followed by cosine annealing decay down to $\eta_{\text{min}}$ over the remaining 180 epochs.

