# Forensic Codebase Audit & Tier 2 Pre-Training Upgrades

*Authoritative documentation of the Phase 1 forensic audit remediations and Phase 2 Tier 2 algorithmic pre-training upgrades.*

---

## 1. Overview & Context

To ensure mathematical rigor, benchmark reproducibility, and peer-review defensibility ([`AGENTS.md`](../AGENTS.md)), the codebase underwent a forensic audit followed by a series of pre-training algorithmic upgrades, consolidated on the `main` branch.

```
       Phase 1: Forensic Codebase Audit
  ┌──────────────────────────────────────────────┐
  │ 1. 0% Patient Leakage (Grouped base_id)      │
  │ 2. Composite Stratified Subsampling          │
  │ 3. Deep Supervision Symmetric Parity         │
  │ 4. Simpson's Paradox Fix & Uninit Head Guard │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
       Phase 2: Tier 2 Pre-Training Upgrades
  ┌──────────────────────────────────────────────┐
  │ A. Anisotropic 3D Target Jitter Masking      │
  │ B. Dense Volumetric Token Regularization     │
  │ C. Dual Cosine Schedulers (LR & Reg Scale)   │
  └──────────────────────────────────────────────┘
```

---

## 2. Phase 1: Forensic Audit Remediations

### 2.1 Remediation 1: Zero Longitudinal Patient Leakage
- **Vulnerability Identified**: BraTS patient identifiers include multi-time-point longitudinal scans (e.g., `BraTS-GLI-00001-000` and `BraTS-GLI-00001-001`). Naive randomized or simple stratified splitting allowed scans from the same patient to appear simultaneously in `train` and `test` splits, causing data leakage.
- **Implementation**: [`scripts/prepare_data_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/scripts/prepare_data_3d.py) extracts base patient IDs using regex `r"^(.*)-\d+$"`. Stratified splitting is performed strictly at the patient level on unique `base_id`s. Scans from a given patient are assigned atomically to exactly one split.
- **Verification Guarantee**: 0% overlap between splits ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$). Verified in [`tests/test_data_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_data_3d.py).

### 2.2 Remediation 2: Composite Stratification in Low-Data Subsampling
- **Vulnerability Identified**: When subsampling labels down to $1\%$ (13 volumes) or $5\%$ (63 volumes), naive random subsampling could randomly select only non-enhancing or small lesions, skewing evaluation results across baseline comparisons.
- **Implementation**: [`src/brats_jepa_3d/data/dataset.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/src/brats_jepa_3d/data/dataset.py) constructs composite stratification keys: $\text{volume\_quartile} \times \text{has\_enhancing\_tumor}$.
- **Verification Guarantee**: Low-data subsets retain identical distributional proportions of micro-lesions, macro-lesions, and enhancing tumor subtypes across all models.

### 2.3 Remediation 3: Deep Supervision Parity Across All Baselines
- **Vulnerability Identified**: In low-data evaluation ([`scripts/evaluate_low_data_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/scripts/evaluate_low_data_3d.py)), state-dict key-peeking (`any(k.startswith("decoder.ds"))`) was used to detect deep supervision. Because self-supervised JEPA checkpoints do not contain downstream decoder heads, deep supervision was erroneously disabled for JEPA while remaining enabled for nnU-Net, violating benchmark parity.
- **Implementation**: Replaced key-peeking with explicit CLI flags (`--deep_supervision` / `--no_deep_supervision`) that configure all models identically.
- **Verification Guarantee**: 3D VisReg JEPA, 3D nnU-Net, and 3D Residual UNet all receive identical multi-scale gradient supervision during downstream fine-tuning.

### 2.4 Remediation 4: Simpson's Paradox Elimination & Loud Guards
- **Vulnerability Identified**: 
  1. Averaging batch-level metrics when evaluating small or missing regions (e.g., enhancing tumor ET in non-enhancing scans) created unweighted distortion (Simpson's paradox).
  2. Evaluating an encoder checkpoint without downstream head initialization yielded soft warnings instead of halting execution.
- **Implementation**: 
  1. [`scripts/evaluate_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/scripts/evaluate_3d.py) records per-sample metric accumulation conditioned on target tumor presence, reporting true macro-averages and sample standard deviations.
  2. Added a loud `RuntimeError` guard preventing uninitialized decoders from producing invalid evaluation metrics.

---

## 3. Phase 2: Tier 2 Pre-Training Upgrades

### 3.1 Upgrade A: Anisotropic 3D Target Jitter Masking

#### The Mathematical Problem
Conventional 3D JEPA designs typically use isotropic cubic target blocks ($3 \times 3 \times 3 = 27$ tokens). In 3D MRI, gliomas display irregular, anisotropic morphology (infiltrating along planar white matter tracts and sulcal boundaries). Fixed isotropic targets fail to force the predictor to learn directional spatial correlations.

#### Algorithmic Formulation
The target masking strategy specifies 6 multi-aspect slots with varying aspect ratios and token volumes:
$$\mathcal{S}_{\text{slots}} = \big[(3,3,3),\, (2,3,3),\, (3,2,3),\, (3,3,2),\, (2,2,2),\, (2,2,2)\big]$$

$$\text{Total Target Tokens (8³ Grid)} = 27 + 18 + 18 + 18 + 8 + 8 = \mathbf{97\text{ tokens}} \quad (18.9\% \text{ of 512-token grid})$$

When running on high-resolution $16^3$ token grids (e.g., $8^3$ voxel patch size on NVIDIA A100), the cuboids automatically scale $2\times$ per axis to preserve the exact same physical millimeter anatomical coverage:
$$\mathcal{S}_{16\text{-grid}} = \big[(6,6,6),\, (4,6,6),\, (6,4,6),\, (6,6,4),\, (4,4,4),\, (4,4,4)\big]$$
$$\text{Total Target Tokens (16³ Grid)} = 216 + 144 + 144 + 144 + 64 + 64 = \mathbf{776\text{ tokens}} \quad (18.9\% \text{ of 4,096-token grid})$$

1. **Volume & Physical Millimeter Preservation**: Maintains dense supervisory signal ($\approx 100$ tokens on $8^3$ grid, $776$ tokens on $16^3$ grid) corresponding to an identical $48\text{ mm}$ isotropic cube, $32 \times 48 \times 48\text{ mm}$ planar slabs, and $32\text{ mm}$ regional meso blocks.
2. **Proportional Tissue Filtering**:
   $$\text{min\_tissue} = \max\big(4, \, \lfloor 0.40 \times (c_z \times c_y \times c_x) \rfloor\big)$$
   This proportional threshold prevents rejection deadlock on small 8-token and 18-token cuboids while guaranteeing that no target samples empty background air.
3. **Multi-Aspect Fallback Bounds**: When sampling falls back due to high tissue occupancy constraints, dimensions are clamped strictly within $[2, 3]^3$ (or $[4, 6]^3$ for $16^3$ grid), preserving anisotropic diversity.

---

### 3.2 Upgrade B: Dense Volumetric Token Regularization

#### The Statistical Formulation
VisReg and SigReg regularizers compute empirical covariance and Sliced-Wasserstein distances over token representations. Unlike 2D image models where token counts per batch are modest, full 3D volumetric token grids generate dense token populations in every mini-batch:
- **Token Density**: At $B=8$ or $B=16$ with patch size $16^3$ ($512$ tokens per volume) or $8^3$ ($4,096$ tokens per volume), each mini-batch provides $N = B \times N_{\text{tokens}} = \mathbf{4,096\text{ to }65,536\text{ tokens}}$.
- **Statistical Fidelity**: Computing regularizers directly across active mini-batch tokens avoids historical cross-batch queue staleness (where tokens from past optimization steps no longer reflect current encoder weights) while delivering high statistical power for Sliced-Wasserstein projections ($K=64$) and empirical covariance estimation.
- **Autograd Purity**: 100% of tokens in the regularizer receive active, non-stale gradients backpropagated directly into the context encoder backbone.

```
Current Batch Tokens [B * N_tokens, D] (requires_grad=True)
                 │
                 ├──► VisReg Loss Computation (Center, Scale, SWD Shape)
                 │    or SigReg Loss Computation (Epps-Pulley Test)
                 ▼
         Backpropagate directly to Context Encoder
```

#### Architectural Safeguards:
1. **Autograd Isolation**: Center and scale statistics ($\mu, \sigma$) in VisReg are detached when normalizing representations for the Sliced-Wasserstein shape loss, preventing gradient interference.
2. **Gradient Scaling**: SigReg applies an explicit $N$-multiplier scaling factor to match the test statistic gradient scale across arbitrary batch sizes.
3. **Decoupled Evaluation**: Representation collapse metrics (Effective Rank, Centered Cosine Sim, Feature Variance) are computed in `eval()` mode strictly on holdout validation volumes.

---

### 3.3 Upgrade C: Cosine Regularizer Annealing Schedule

#### The Optimization Problem
At constant regularizer strength ($\lambda_{\text{reg}} = 1.0$), statistical anti-collapse forces remain dominant throughout training, competing with the prediction loss and creating an optimization plateau around Epoch 12.

#### Algorithmic Formulation
Implemented in [`scripts/train_jepa_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/scripts/train_jepa_3d.py):
$$\lambda_{\text{reg}}(t) = \begin{cases} 1.0, & t \le T_{\text{warmup}} \\ 0.10 + 0.90 \times \frac{1}{2}\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}} \pi\right)\right), & t > T_{\text{warmup}} \end{cases}$$

- **Warmup Phase ($t \le T_{\text{warmup}}$ epochs, e.g., 30 epochs for $T_{\text{max}}=200$, 15 epochs for $T_{\text{max}}=100$)**: Full strength ($\lambda_{\text{reg}} = 1.0$) rapidly pushes representations away from the trivial constant collapse basin.
- **Annealing Phase ($t > T_{\text{warmup}}$ epochs)**: Smooth cosine decay down to $\lambda_{\text{min}} = 0.10$, allowing the predictor to refine fine-grained anatomical representations without excessive regularizer penalty.

---

### 3.4 Upgrade D: Dual Schedule Synergy (Learning Rate Warmup & Cosine Decay)

#### The Optimization Dynamics
Pre-training 3D Vision Transformers on volumetric MRI with optimal transport regularizers presents dual optimization challenges: early attention gradient instability and late-stage regularizer interference. To resolve both, the training pipeline pairs two synchronized schedules:

1. **Learning Rate Schedule with Linear Warmup**:
   $$\eta(t) = \begin{cases} \eta_{\text{base}} \times \frac{t}{T_{\text{lr\_warmup}}}, & t \le T_{\text{lr\_warmup}} \\ \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{base}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t - T_{\text{lr\_warmup}}}{T_{\text{max}} - T_{\text{lr\_warmup}}} \pi\right)\right), & t > T_{\text{lr\_warmup}} \end{cases}$$
   With $\eta_{\text{base}} = 5 \times 10^{-4}$, $\eta_{\text{min}} = 1 \times 10^{-5}$, and $T_{\text{lr\_warmup}} = 20$ epochs for canonical 200-epoch pre-training ($T_{\text{lr\_warmup}} = 10$ epochs for 100-epoch runs).

2. **Synergistic Phasing Across Training Epochs (200-Epoch Canonical Schedule)**:
   - **Phase 1 (Epochs 1–20: Gradient Stabilization & Dispersal)**: $\eta(t)$ ramps up linearly while $\lambda_{\text{reg}} = 1.0$. The low early learning rate prevents self-attention divergence, while maximal regularizer forces actively repel token clusters outward across the latent hypersphere.
   - **Phase 2 (Epochs 21–30: Latent Geometry Stabilization)**: $\eta(t)$ begins its cosine decay while $\lambda_{\text{reg}}$ remains clamped at $1.0$, allowing the backbone to consolidate the high-dimensional latent manifold before regularizer relaxation.
   - **Phase 3 (Epochs 31–200: Asymptotic Predictor Convergence)**: Both $\eta(t)$ and $\lambda_{\text{reg}}(t)$ anneal via smooth cosine decay toward their floors ($\eta_{\text{min}} = 1 \times 10^{-5}$, $\lambda_{\text{min}} = 0.10$). As parameter updates become subtle, the predictor refines complex anatomical boundaries without fighting strong regularizer gradient repulsion.

---

## 4. Test Suite Verification & Validation Summary

The upgrades are covered by the consolidated 6-suite regression harness:

| Suite | File | Test Count | Scope |
| :--- | :--- | :---: | :--- |
| **Data Suite** | [`test_data_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_data_3d.py) | 28 | 0% leakage, anisotropic jitter token counts, proportional tissue rejection, fallback bounds. |
| **Evaluation Suite** | [`test_evaluation_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_evaluation_3d.py) | 39 | Deep supervision parity, Simpson's paradox elimination, uninit decoder guard. |
| **Integration Suite** | [`test_integration_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_integration_3d.py) | 10 | End-to-end smoke tests for training and low-data evaluation pipelines. |
| **Losses Suite** | [`test_losses_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_losses_3d.py) | 32 | VisReg SWD shape loss, SigReg Epps-Pulley test, Dice/BCE/Tversky, deep supervision, cosine annealing. |
| **Metrics Suite** | [`test_metrics_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_metrics_3d.py) | 19 | 3D HD95, Dice, IoU calculation, multi-region zero handling. |
| **Models Suite** | [`test_models_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_models_3d.py) | 32 | ViT encoder, predictor, FPN, UNETR hybrid, DynUNet, Residual UNet architectures. |
| **Total** | — | **159** | **100% Pass Rate** |
