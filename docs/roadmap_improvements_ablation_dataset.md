# 📋 Phased Engineering & Scientific Roadmap: Fair Benchmarks Across All Models, 3D JEPA Improvements, Validation Optimization, ViT From-Scratch Ablation, Hybrid UNETR Stem, and Additional Dataset Kaggle Packaging

**Document Version**: 2.2 (2026-09-20)  
**Target Repository**: `thesis_3d`  
**Theoretical Framework**: In strict accordance with [`AGENTS.md`](../AGENTS.md) and [`docs/`](./) specifications.

---

## 1. Executive Summary & Core Principle: Fair Benchmark Across All Models

To ensure the master thesis and subsequent paper submission withstand rigorous peer review, all comparative models (**3D nnU-Net**, **3D Residual UNet**, and **3D VisReg JEPA**) must be benchmarked under **strictly fair, symmetric, and robust conditions**:

1. **Multi-Scale Deep Supervision Applied to ALL 3 Models**:
   - nnU-Net previously had deep supervision enabled, giving it a strong multi-scale gradient advantage.
   - Deep supervision is now integrated and enabled by default across **all three architectures**: 3D nnU-Net (`DynUNet`), 3D UNet (`BraTS3DUNet`), and 3D VisReg JEPA (`MultiScaleViTSegmentationDecoder3D` and `HybridUNETRDecoder3D`).
2. **Asymmetric Boundary-Weighted Tversky Loss ($\beta=0.7, \alpha=0.3$) Applied to ALL 3 Models**:
   - Standard Dice loss treats False Positives ($FP$) and False Negatives ($FN$) symmetrically. In intracranial MRI, brain tumors occupy $<1.5\%$ of the total scan volume. Missing thin infiltrative margins ($FN$) has little impact on Dice, yet heavily inflates Hausdorff distance ($HD95$).
   - Weighting $FN$ $2.33\times$ higher ($\beta/\alpha = 0.7/0.3 \approx 2.33$) forces the models to aggressively track peripheral margins, drastically reducing $HD95$.
   - Supported across **all three training runners** (`train_nnunet_3d.py`, `train_unet_3d.py`, and `train_downstream_3d.py`).
3. **3D Test-Time Augmentation (TTA, 4-Fold Reflections) Applied to ALL 3 Models**:
   - 4-fold reflection averaging (original + flip $X$ + flip $Y$ + flip $Z$) implemented in `predict_with_tta_3d` and evaluated uniformly across **all three architectures** in `evaluate_3d.py`.
4. **Validation Latency & Memory Bottleneck** ([`docs/training_validation_bottleneck_and_hd95_optimization.md`](./training_validation_bottleneck_and_hd95_optimization.md)):
   - Decouple HD95 from routine epoch validation (`compute_hd95=False`) and add point-cap subsampling ($max\_points=10000$), speeding up validation by **$18.7\times$** ($13\text{ minutes} \to 2.4\text{ seconds}$).
5. **Scientific ViT From-Scratch Ablation Baseline** ([`docs/vit_from_scratch_ablation.md`](./vit_from_scratch_ablation.md)):
   - Clean random-initialization baseline (`--from_scratch`) to isolate the true self-supervised representation gain ($\Delta_{\text{JEPA}}$).
6. **UNETR-Style Hybrid Conv Stem ($128^3 + 64^3$ Skips)** ([`docs/improving_3d_jepa_segmentation_performance.md`](./improving_3d_jepa_segmentation_performance.md)):
   - Injects native pixel-level spatial skips into decoder stages 4 and 3, **usable immediately with existing pre-trained weights without retraining SSL**.
7. **Finer $8^3$ Token Resolution ($4,096$ Tokens)**:
   - High-resolution ViT backbone with $8\times$ higher spatial token density, supported by PyTorch SDPA (FlashAttention-2).
8. **Additional Dataset Preprocessing & Kaggle Packaging**:
   - 271 patient scans from [`data/raw/BraTS_2024/BraTS-GLI/training_data_additional`](../data/raw/BraTS_2024/BraTS-GLI/training_data_additional) preprocessed into canonical $128^3$ `.npz` and packaged into `dist_kaggle/brats_3d_additional.zip`.

---

## 2. Priority Ordering: Which Task Should Be Done First?

The implementation sequence is strictly dictated by **algorithmic dependencies, compute efficiency, and checkpoint compatibility**:

```mermaid
flowchart TD
    P1["Phase 1 (MUST BE DONE FIRST): Validation & HD95 Optimization
    - compute_hd95=False in epoch validation
    - point-cap subsampling (max 10k points) in compute_hd95_3d
    - persistent_workers=False on val loaders
    - tqdm real-time telemetry"]
    
    P2["Phase 2 (SECOND): ViT From-Scratch Ablation Baseline
    - Add --from_scratch flag in train_downstream_3d.py
    - Isolate checkpoints (*_scratch_best.pt) and metrics
    - Prevents architectural confounding critique"]
    
    P3["Phase 3 (THIRD): Fair Downstream Training Suite for ALL Models
    - Deep Supervision for ALL (nnunet, unet, visreg)
    - Asymmetric Tversky Loss (beta=0.7, alpha=0.3) for ALL (nnunet, unet, visreg)
    - UNETR-Style Hybrid Conv Stem (128³ + 64³ skips) in segmentation_head_3d.py
    - 3D Test-Time Augmentation (TTA) in evaluate_3d.py for ALL models
    - USES EXISTING PRE-TRAINED WEIGHTS (No SSL retraining needed!)"]
    
    P4["Phase 4 (FOURTH): Finer 8³ Token Resolution Support
    - Configurable patch_size (8 vs 16) in PatchEmbed3D & ViT
    - Adaptive decoder for 16³ initial grid (4,096 tokens)
    - FlashAttention-2 / SDPA memory optimization"]
    
    P5["Phase 5 (FIFTH): Additional Raw Dataset & Kaggle Packaging
    - Process 271 volumes from training_data_additional -> 128³ .npz
    - Stratified metadata.csv (train/val/test)
    - Package dist_kaggle/brats_3d_additional.zip + metadata.json"]
    
    P6["Phase 6 (SIXTH): Kaggle Notebooks Synchronization
    - Update 01, 02, 03, and kaggle_runner notebooks
    - Auto-mount attached Kaggle dataset
    - Add ablation, hybrid stem, and fair-comparison cells"]
    
    P7["Phase 7 (SEVENTH): Comprehensive Regression Suite
    - Automated tests for validation speedup, hybrid stem, Tversky loss, 8³ tokens
    - Pytest across all test suites (90+ tests)"]

    P1 -->|Unlocks 18.7x faster validation| P2
    P2 -->|Isolates representation gain| P3
    P3 -->|Maximizes downstream Dice on existing weights| P4
    P4 -->|Architecture ready for next pre-train| P5
    P5 -->|Upload-ready dataset archive| P6
    P6 --> P7
```

---

## 3. Detailed Technical Blueprint by Phase

### Phase 1: Core Validation Engine & HD95 Optimization (MUST BE DONE FIRST)
- **Problem**: In early epochs, noisy boundary predictions create $200,000\text{ to }500,000$ points, freezing validation for 13 minutes per epoch and consuming $15\text{ GB}$ RAM.
- **Actions**:
  1. In `src/brats_jepa_3d/metrics/volumetric_metrics.py`:
     Add `compute_hd95: bool = True` to `compute_volumetric_metrics_3d`. When `False`, return `nan` without KD-tree construction.
  2. In `compute_hd95_3d`:
     Subsample boundary coordinates to $10,000$ points (`max_points: int = 10000`) before constructing `scipy.spatial.cKDTree`.
  3. In `scripts/train_downstream_3d.py`, `scripts/train_nnunet_3d.py`, and `scripts/train_unet_3d.py`:
     Pass `compute_hd95=False` during routine epoch validation.
     Set `persistent_workers=False` on `val_loader`.
     Add real-time `tqdm(loader, desc="Validating", leave=False)`.

---

### Phase 2: Scientific ViT From-Scratch Baseline
- **Problem**: Reviewers can challenge whether performance comes from the ViT+FPN architecture or from self-supervised representation learning.
- **Actions**:
  1. Add `--from_scratch` CLI argument to `scripts/train_downstream_3d.py`.
  2. When enabled, bypass pre-trained checkpoint auto-discovery and train from random initialization.
  3. Save checkpoints distinctly to `outputs/checkpoints/{model_type}_{decoder_type}_scratch_best.pt`.
  4. Log metrics to `outputs/logs/{model_type}_{decoder_type}_scratch_downstream_metrics.csv`.

---

### Phase 3: Fair Downstream Training Suite Applied to ALL Models
- **Principle**: Zero optimization asymmetry between `nnunet`, `unet`, and `visreg`.

#### 3.1 Deep Supervision for ALL Models
- `BraTS3DnnUNet`: Built-in multi-scale heads (`DynUNet`, `deep_supervision=True`).
- `BraTS3DUNet`: Equipped with multi-scale auxiliary heads (`ds3` at $64^3$, `ds2` at $32^3$, `ds1` at $16^3$) and `deep_supervision=True`.
- `JEPASegmentationModel3D`: Multi-scale heads (`ds1, ds2, ds3`) enabled with `--deep_supervision` flag (default `True`).
- All models return `[out_128, ds3_64, ds2_32, ds1_16]` during training.
- `DeepSupervisionLoss3D` calculates normalized exponential decay weights: $w = [0.533, 0.267, 0.133, 0.067]$.

#### 3.2 Asymmetric Tversky Loss ($\beta=0.7, \alpha=0.3$) for ALL Models
- Create `src/brats_jepa_3d/losses/tversky_loss_3d.py`:
  - `VolumetricTverskyLoss` with $\alpha=0.3, \beta=0.7$.
  - `CombinedTverskyBCEWithLogitsLoss3D`.
- Make `DeepSupervisionLoss3D` accept custom `base_loss: nn.Module`, allowing it to wrap either Dice+BCE or Asymmetric Tversky loss.
- Expose `--loss_type` (`dice_bce` vs `tversky`), `--tversky_alpha 0.3`, `--tversky_beta 0.7` in:
  - `scripts/train_downstream_3d.py`
  - `scripts/train_nnunet_3d.py`
  - `scripts/train_unet_3d.py`

#### 3.3 UNETR-Style Hybrid Conv Stem ($128^3 + 64^3$ Skips)
- Create `HybridUNETRDecoder3D` in `src/brats_jepa_3d/models/segmentation_head_3d.py`:
  - 2-stage lightweight 3D Conv stem: $\text{stem}_{128}$ (16 ch) and $\text{stem}_{64}$ (32 ch).
  - Horizontal skips into decoder stage 4 ($64^3 \to 128^3$) and stage 3 ($32^3 \to 64^3$).
  - Usable **directly with existing pre-trained weights (`visreg_jepa_3d_best.pt`) without retraining SSL**!

#### 3.4 3D Test-Time Augmentation (TTA) for ALL Models
- In `scripts/evaluate_3d.py`:
  - Add `predict_with_tta_3d(model, volume)` for 4-fold 3D orthogonal reflection averaging.
  - Add CLI argument `--tta`. When enabled, evaluates `nnunet`, `unet`, and `visreg` under the exact same TTA protocol.

---

### Phase 4: Finer $8^3$ Token Resolution Support ($4,096$ Tokens)
- Enable configurable `patch_size=(8, 8, 8)` in `VisionTransformerEncoder3D` and `PatchEmbed3D`.
- Sequence length expands from $512 \to 4,096$ tokens ($8\times$ spatial density).
- Utilize PyTorch native `scaled_dot_product_attention` (FlashAttention-2 / SDPA) for linear memory scaling.
- Adapt decoders for initial $16^3$ spatial grid ($16^3 \to 32^3 \to 64^3 \to 128^3$).
- Add `--patch_size` argument in training scripts.

---

### Phase 5: Additional Raw Dataset Preprocessing & Kaggle Packaging
- Source: `data/raw/BraTS_2024/BraTS-GLI/training_data_additional` ($271$ patient scans).
- Run `scripts/prepare_data_3d.py` outputting to `data/processed/brats_gli_additional_3d`.
- Package into `dist_kaggle/brats_3d_additional.zip` with `dataset-metadata.json`.
- Dynamic path resolution in `src/brats_jepa_3d/config.py`.

---

### Phase 6: Kaggle Notebooks Synchronization
- Update `01_train_visreg_3d.ipynb`, `02_train_nnunet_3d.ipynb`, `03_train_unet_3d.ipynb`, and `kaggle_runner_3d.ipynb`:
  - Auto-mount `brats-3d-additional` dataset.
  - Add fair comparison experiment cells:
    - Deep supervision toggle (default ON).
    - Asymmetric Tversky loss toggle (`--loss_type tversky`).
    - From-Scratch ViT ablation toggle.
    - UNETR-Style Hybrid Conv Stem toggle (`--decoder_type unetr_hybrid`).
    - TTA evaluation toggle (`--tta`).

---

### Phase 7: Comprehensive Regression Suite
- Unit tests in `tests/test_fair_benchmarks_and_improvements_3d.py`:
  - Validation speedup ($<10\text{ ms}$ with `compute_hd95=False`).
  - HD95 subsampling ($<100\text{ ms}$ on 200k points).
  - Tversky loss verification & gradient check.
  - Deep supervision output shape checks on UNet, nnU-Net, and VisReg.
  - Hybrid UNETR decoder shape & gradient check.
  - $8^3$ patch resolution check.
  - TTA consistency test across all models.
- Full test suite run (`pytest tests/ -v`, 90+ tests).
