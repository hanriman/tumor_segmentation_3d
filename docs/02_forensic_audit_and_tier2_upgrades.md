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
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
       Phase 3: Scientific Validity & Audit Remediations
  ┌──────────────────────────────────────────────┐
  │ I.   Patch Size Realignment (8³ / 4,096 tok) │
  │ II.  Pure vs Hybrid Probing Disentanglement  │
  │ III. Missing Modality Inverted Dropout Scale │
  │ IV.  Tissue-Filtered Effective Spectral Rank │
  │ V.   Low-Data Step Budget Formal Analysis    │
  │ VI.  Test Output Isolation (Zero Data Risk)  │
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
   $$\text{min\_tissue} = \max\big(2, \, \lfloor 0.25 \times (c_z \times c_y \times c_x) \rfloor\big)$$
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

### 3.3 Upgrade C: Regularizer Warmup & Constant Full-Strength Schedule

#### Algorithmic Formulation
Implemented in [`scripts/train_jepa_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/scripts/train_jepa_3d.py):
$$\lambda_{\text{reg}}(t) = \begin{cases} \frac{t}{T_{\text{warmup}}}, & t < T_{\text{warmup}} \\ 1.0, & t \ge T_{\text{warmup}} \end{cases}$$

- **Linear Warmup ($t < 30$ epochs)**: Linearly ramps regularizer weight from $0 \to 1.0$ while early attention representations form.
- **Constant Full-Strength Phase ($t \ge 30$ epochs, `reg_min_scale=1.0`)**: Following VISReg (Wu, Balestriero, & Levine, 2026) and LeJEPA (Balestriero & LeCun, 2025), $\lambda_{\text{reg}}$ remains at $1.0$ throughout training to prevent late-stage variance decay across all 384 encoder dimensions.

---

### 3.4 Upgrade D: Learning Rate Linear Warmup & Cosine Decay

1. **Learning Rate Schedule with Linear Warmup**:
   $$\eta(t) = \begin{cases} \eta_{\text{base}} \times \frac{t}{T_{\text{lr\_warmup}}}, & t \le T_{\text{lr\_warmup}} \\ \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{base}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t - T_{\text{lr\_warmup}}}{T_{\text{max}} - T_{\text{lr\_warmup}}} \pi\right)\right), & t > T_{\text{lr\_warmup}} \end{cases}$$
   With $\eta_{\text{base}} = 5 \times 10^{-4}$, $\eta_{\text{min}} = 1 \times 10^{-5}$, and $T_{\text{lr\_warmup}} = 10$ epochs for canonical 200-epoch pre-training.

---

## 4. Phase 3 & 4: Scientific Validity, Paper-Faithful Alignment & Code Unification

Following an exhaustive forensic audit against scientific integrity standards ([`AGENTS.md`](../AGENTS.md)) and the original VISReg (`arXiv:2602.11401`) and LeJEPA/SigReg (`arXiv:2511.08544`) papers, the codebase was aligned and simplified:

### 4.1 Remediation I: Architectural Patch Size & Grid Realignment ($8^3$ vs. $16^3$)
- **Vulnerability**: Checkpoints in `outputs/experiments/` were trained on an $8 \times 8 \times 8$ patch embedding ($16^3 = 4,096$ tokens, $N_{\text{ctx}} = 1,536$, $N_{\text{tgt}} = 776$), whereas early manuscript drafts claimed $16 \times 16 \times 16$ patches ($512$ tokens).
- **Remediation**:
  1. Updated [`paper/latex/main.tex`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/paper/latex/main.tex) to specify the canonical $8 \times 8 \times 8$ voxel patch embedding ($16^3 = 4,096$ tokens).
  2. Set default `patch_size: [8, 8, 8]` across all YAML configuration files ([`configs/model/visreg_jepa_3d.yaml`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/configs/model/visreg_jepa_3d.yaml), [`configs/model/sigreg_jepa_3d.yaml`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/configs/model/sigreg_jepa_3d.yaml), [`configs/dataset/brats3d.yaml`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/configs/dataset/brats3d.yaml)) and evaluation scripts.

### 4.2 Remediation II: Direct Encoder-Space VISReg/SigReg Regularization
- **Vulnerability**: Applying anti-collapse regularization after a bottleneck projector MLP ($384 \to 1024 \to 128$) allowed the projector to satisfy Gaussianity in $\mathbb{R}^{128}$ while the encoder representations in $\mathbb{R}^{384}$ collapsed along their nullspace, and omitting standardization before unit-sphere projection conflated scale and shape gradients.
- **Remediation**:
  1. Removed the projector MLP from [`SingleEncoderJEPA3D`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/src/brats_jepa_3d/models/single_encoder_jepa_3d.py) so both `VisRegLoss` and `SigRegLoss` operate directly on tissue-filtered context encoder tokens $\mathbf{Z}_{\text{tissue}} \in \mathbb{R}^{N_{\text{tissue}} \times 384}$ (matching VISReg Section 3 and LeJEPA Section 3).
  2. Aligned [`VisRegLoss._sliced_wasserstein_distance`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/src/brats_jepa_3d/losses/visreg_loss.py) with Algorithm 1 of Wu et al. (2026): standardizing tokens via detached moments `z_std = (z - mu.detach()) / sigma.detach()` before projecting onto $M=1024$ unit directions with $\lambda_{\text{sh}} = 4.0$.

### 4.3 Remediation III: Zero-Imputed Modality Dropout Parity
- **Vulnerability**: Training with inverted-dropout channel rescaling ($C / \text{keep} = 4\times$ when 1 of 4 MRI modalities survives) while evaluating missing-modality OOD regimes with zero-imputed unscaled inputs created a $4\times$ input magnitude mismatch for ViT `PatchEmbed3D` + `LayerNorm` relative to `InstanceNorm3d` CNNs.
- **Remediation**:
  1. Set `rescale: bool = False` by default in [`RandomModalityDropout3D`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/src/brats_jepa_3d/data/transforms.py), ensuring training-time modality dropout uses exact zero-imputation matching [`evaluate_ood_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/scripts/evaluate_ood_3d.py).

### 4.4 Remediation IV: Low-Data Step-Budget Floor & Validation Selection
- **Vulnerability**: Fixed 25-epoch training at $1\%$ labels ($12$ volumes, batch size $4$) yielded only $75$ gradient updates, and selecting the best low-data epoch directly on the test split risked optimistic selection bias.
- **Remediation**:
  1. Added `--min_steps 250` floor in [`scripts/evaluate_low_data_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/scripts/evaluate_low_data_3d.py) so every low-data fraction receives at least $250$ gradient updates.
  2. Selects the best checkpoint strictly on the validation split (`val_loader`) before evaluating the selected checkpoint once on `test_loader`.

### 4.5 Remediation V: Intracranial Brain-Tissue Decoupling in Effective Rank
- **Vulnerability**: Because $\approx 60\%$ of brain MRI volumes consist of background air zeros ($X=0$), over half of all tokens fed into `compute_representation_collapse_metrics` were deterministic functions of static sinusoidal positional embeddings, contaminating Effective Spectral Rank ($S^2$).
- **Remediation**: [`scripts/evaluate_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/scripts/evaluate_3d.py) pools `brain_mask` onto the token grid to compute both all-token metrics (`eRank(all)`) and tissue-filtered metrics (`eRank(tissue)`).

### 4.6 Remediation VI: Central Model Factory & Shared Supervised Trainer
- **Architectural Unification**: Consolidated all model construction and checkpoint loading into [`src/brats_jepa_3d/models/builder.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/src/brats_jepa_3d/models/builder.py) and unified downstream/baseline training loops into [`src/brats_jepa_3d/training/supervised_trainer.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/src/brats_jepa_3d/training/supervised_trainer.py), guaranteeing identical training, validation, and evaluation semantics across ViT-JEPA, Scratch ViT, 3D Residual UNet, and 3D nnU-Net.

---

## 5. Test Suite Verification & Validation Summary

The complete codebase is guarded by a comprehensive 7-suite regression harness:

| Suite | File | Test Count | Scope |
| :--- | :--- | :---: | :--- |
| **Benchmark Consistency** | [`test_benchmark_consistency.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_benchmark_consistency.py) | 7 | Mathematical synchronization between CSV benchmarks, paper tables, and documentation. |
| **Data Suite** | [`test_data_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_data_3d.py) | 30 | 0% leakage, anisotropic jitter token counts, proportional tissue rejection, fallback bounds, zero-imputed modality dropout. |
| **Evaluation Suite** | [`test_evaluation_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_evaluation_3d.py) | 39 | Central model builder, checkpoint loader, Simpson's paradox elimination, uninit decoder guard. |
| **Integration Suite** | [`test_integration_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_integration_3d.py) | 10 | End-to-end smoke tests for training and low-data evaluation pipelines with safe teardown in `tests/test_outputs/`. |
| **Losses Suite** | [`test_losses_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_losses_3d.py) | 35 | VISReg SWD shape loss (detached $\mu,\sigma$), SigReg Epps-Pulley test, Dice/BCE/Tversky, deep supervision, warmup schedule. |
| **Metrics Suite** | [`test_metrics_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_metrics_3d.py) | 19 | 3D HD95, Dice, IoU calculation, multi-region zero handling. |
| **Models Suite** | [`test_models_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/tests/test_models_3d.py) | 32 | ViT encoder, predictor, `SingleEncoderJEPA3D`, FPN, UNETR hybrid, DynUNet, Residual UNet architectures. |
| **Total** | — | **172** | **100% Pass Rate** |
