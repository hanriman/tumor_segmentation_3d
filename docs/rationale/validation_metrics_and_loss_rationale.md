# 🔬 Scientific & Clinical Rationale: Validation Metrics vs. Surrogate Loss & Checkpoint Selection

**Document Version**: 1.0 (2026-09-19)  
**Target Repository**: `thesis_3d` (`scripts/train_nnunet_3d.py`, `scripts/train_unet_3d.py`, `scripts/train_downstream_3d.py`)  
**Primary Standards**: MICCAI, IEEE Transactions on Medical Imaging (TMI), BraTS Challenge Guidelines, Fabian Isensee et al. (Nature Methods 2021)  
**Theoretical Framework**: In accordance with [`AGENTS.md`](../AGENTS.md) guidelines for empirical evidence, metric validity, and clinical evaluation integrity.

---

## Table of Contents
1. [Executive Summary & Core Principle](#1-executive-summary--core-principle)
2. [Anatomy of Training Logs & Progress Indicators](#2-anatomy-of-training-logs--progress-indicators)
3. [Why Clinical Metrics (Dice & HD95) Supersede Surrogate Validation Loss](#3-why-clinical-metrics-dice--hd95-supersede-surrogate-validation-loss)
4. [Deep Supervision Asymmetry in 3D nnU-Net & Multi-Scale ViT-FPN](#4-deep-supervision-asymmetry-in-3d-nnu-net--multi-scale-vit-fpn)
5. [Model Checkpoint Selection Dynamics (`best_val_dice`)](#5-model-checkpoint-selection-dynamics-best_val_dice)
6. [Empirical Convergence & Diagnostics Profile](#6-empirical-convergence--diagnostics-profile)
7. [Thesis Defense & Peer Review Reference Handbook](#7-thesis-defense--peer-review-reference-handbook)

---

## 1. Executive Summary & Core Principle

During the training of 3D medical segmentation models in this repository ([`scripts/train_nnunet_3d.py`](../scripts/train_nnunet_3d.py), [`scripts/train_unet_3d.py`](../scripts/train_unet_3d.py), and [`scripts/train_downstream_3d.py`](../scripts/train_downstream_3d.py)), the console output displays:

```text
[2026-09-18 20:37:17] [INFO] [train_nnunet_3d]: Epoch 1/30 - LR: 0.000084 | Train Loss: 0.9522 | Val Dice: 0.6770 | Val HD95: 26.24 mm
[2026-09-18 20:37:17] [INFO] [train_nnunet_3d]: New best model saved: .../nnunet_3d_best.pt (Val Dice: 0.6770)
```

A common question is: **"Where is the validation loss?"**

**Direct Answer**:  
There is no `Val Loss` reported because the validation loop intentionally evaluates **direct clinical segmentation metrics** rather than the surrogate training loss function. Specifically:
1. **Primary Metric**: **`Val Dice`** (Volumetric Dice Similarity Coefficient) — governs model checkpointing (`nnunet_3d_best.pt`).
2. **Boundary Metric**: **`Val HD95`** (95th-percentile Hausdorff Distance in millimeters) — monitors sub-voxel boundary error.
3. **Region Overlap**: **`Val IoU`** (Jaccard Index) — logged for volumetric overlap verification.

This design adheres strictly to the benchmark protocol established by the **MICCAI Brain Tumor Segmentation (BraTS) Challenge** and **Fabian Isensee’s nnU-Net framework** (Nature Methods, 2021).

---

## 2. Anatomy of Training Logs & Progress Indicators

Each epoch produces two distinct logging outputs. Understanding the mathematical origin of each number eliminates ambiguity:

```text
Epoch 1/30: 100%|████████████████| 473/473 [05:53<00:00,  1.34it/s, loss=0.3884]
                                                                    ▲
                                                  [1] Final Training Batch Loss

[2026-09-18 20:37:17] [INFO] [train_nnunet_3d]: Epoch 1/30 - LR: 0.000084 | Train Loss: 0.9522 | Val Dice: 0.6770 | Val HD95: 26.24 mm
                                                              ▲                   ▲                   ▲                   ▲
                                                      [2] Learning Rate   [3] Epoch Train Loss  [4] Val Dice      [5] Val HD95
```

### [1] Progress Bar Postfix (`loss=0.3884`)
* **Source**: `pbar.set_postfix({"loss": f"{loss.item():.4f}"})`
* **Definition**: The **instantaneous loss of the single final mini-batch** (batch 473 of 473).
* **Scope**: It fluctuates from batch to batch due to stochastic gradient noise and variations in tumor volume across mini-batches. It does *not* reflect the whole epoch.

### [2] Learning Rate (`LR: 0.000084`)
* **Source**: `scheduler.get_last_lr()[0]`
* **Definition**: The scheduled learning rate at the current epoch. In our pipeline, this follows a 5-epoch linear warmup from $0.1 \times \text{LR}$ up to $\text{base\_lr} = 3 \times 10^{-4}$, followed by cosine annealing decay down to $\eta_{\min} = 10^{-6}$.

### [3] Epoch Train Loss (`Train Loss: 0.9522`)
* **Source**: `avg_train_loss = train_loss / max(1, num_batches)`
* **Definition**: The **mean loss averaged across all 473 training mini-batches** in that epoch:
  $$\overline{\mathcal{L}}_{\text{train}} = \frac{1}{K} \sum_{k=1}^K \mathcal{L}_{\text{batch}}^{(k)}$$
  Where $\mathcal{L}_{\text{batch}}$ is computed by [`DeepSupervisionLoss3D`](../src/brats_jepa_3d/losses/deep_supervision.py), integrating multi-scale Soft Dice and Binary Cross-Entropy.

### [4] Validation Dice (`Val Dice: 0.6770`)
* **Source**: `val_metrics['val_dice'] = float(np.mean(all_dices))`
* **Definition**: The mean volumetric Dice Similarity Coefficient (DSC) computed across all holdout validation patients at native resolution ($128^3$).

### [5] Validation HD95 (`Val HD95: 26.24 mm`)
* **Source**: `val_metrics['val_hd95'] = float(np.mean(all_hd95s))`
* **Definition**: The 95th-percentile Hausdorff Distance measuring maximum distance between predicted and ground-truth tumor surface contours, in millimeters.

---

## 3. Why Clinical Metrics (Dice & HD95) Supersede Surrogate Validation Loss

In standard computer vision (e.g., ImageNet classification), cross-entropy validation loss correlates monotonically with top-1 accuracy. In **3D medical volumetric segmentation**, however, surrogate loss functions diverge from clinical utility:

### 3.1 Surrogate Loss vs. Metric Non-Monotonicity
Deep neural networks are trained using smooth, differentiable surrogate loss functions:

$$\mathcal{L}_{\text{train}} = \mathcal{L}_{\text{Dice}}(P, Y) + \mathcal{L}_{\text{BCE}}(P, Y)$$

Where Soft Dice is continuous:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_v p_v y_v + \epsilon}{\sum_v p_v^2 + \sum_v y_v^2 + \epsilon}$$

However:
1. **Calibration vs. Overlap**: Binary Cross-Entropy penalizes overconfident uncalibrated probabilities in the background parenchyma ($128^3 = 2,097,152$ voxels). A network can improve its BCE loss simply by becoming more conservative on background voxels, even if its actual tumor boundary becomes less accurate.
2. **Boundary Sharpness**: Surrogate loss functions cannot directly differentiate contour topology or surface distance. The 95% Hausdorff Distance ($\text{HD95}$) penalizes spatial outliers, which cannot be captured by voxel-wise BCE or soft Dice losses.
3. **Thresholding Discrepancy**: During validation, predictions are thresholded deterministically ($\hat{Y} = \mathbb{I}[P \ge 0.5]$). Soft loss operates on continuous sigmoid probabilities $p \in (0, 1)$, whereas clinical diagnosis operates on binary segmentation masks.

### 3.2 MICCAI BraTS & International Challenge Standards
In the official **MICCAI BraTS Benchmark**, models are evaluated and ranked strictly on:
1. **Dice Similarity Coefficient (DSC)**
2. **Hausdorff Distance 95% (HD95)**

No medical imaging challenge ranks algorithms by cross-entropy or Dice loss. Evaluating and logging clinical metrics directly aligns model selection with the true evaluation criteria.

---

## 4. Deep Supervision Asymmetry in 3D nnU-Net & Multi-Scale ViT-FPN

In both **3D nnU-Net (`BraTS3DnnUNet`)** and **3D VisReg FPN (`MultiScaleViTSegmentationDecoder3D`)**, training employs **Deep Supervision**:

```text
Decoder Tiers & Supervision Resolutions:
├── Head 0 (Full Resolution):   128 × 128 × 128 voxels (Weight w₀ = 0.533)
├── Head 1 (Half Resolution):    64 ×  64 ×  64 voxels (Weight w₁ = 0.267)
├── Head 2 (Quarter Resolution): 32 ×  32 ×  32 voxels (Weight w₂ = 0.133)
└── Head 3 (Eighth Resolution):  16 ×  16 ×  16 voxels (Weight w₃ = 0.067)
```

The training loss is a weighted sum:

$$\mathcal{L}_{\text{total}} = \sum_{h=0}^3 w_h \left[ \mathcal{L}_{\text{Dice}}(P_h, Y_h) + \mathcal{L}_{\text{BCE}}(P_h, Y_h) \right]$$

### The Validation Asymmetry
* **In Training**: Deep supervision is required to force intermediate decoder stages to learn feature representations with explicit semantic meaning, injecting gradients directly into lower layers and preventing vanishing gradients.
* **In Validation and Clinical Inference**: Auxiliary heads ($64^3, 32^3, 16^3$) are **discarded**. Only the full-resolution prediction ($P_{128}$, head 0) is delivered to the radiologist:
  ```python
  with get_autocast_context(device, enabled=amp):
      out = model(images)
      logits = out[0] if isinstance(out, (list, tuple)) else out  # Extract Head 0
  metrics = compute_volumetric_metrics_3d(logits, masks)
  ```
Computing deep supervision losses over low-resolution auxiliary heads during validation:
1. Wastes GPU computation and VRAM on irrelevant downsampled predictions.
2. Contaminates the validation signal with loss values from coarse $16^3$ feature maps that will never be used clinically.

---

## 5. Model Checkpoint Selection Dynamics (`best_val_dice`)

In [`scripts/train_nnunet_3d.py`](../scripts/train_nnunet_3d.py) (as well as `train_downstream_3d.py` and `train_unet_3d.py`), checkpoint persistence follows an explicit greedy selection criterion:

```python
if val_metrics["val_dice"] > best_val_dice:
    best_val_dice = val_metrics["val_dice"]
    best_ckpt_path = CHECKPOINTS_DIR / "nnunet_3d_best.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_dice": best_val_dice,
        },
        best_ckpt_path,
    )
    logger.info(f"New best model saved: {best_ckpt_path} (Val Dice: {best_val_dice:.4f})")
```

### Why `val_dice` is the Ideal Checkpoint Criterion
* **Guarantees Peak Clinical Generalization**: The saved checkpoint `nnunet_3d_best.pt` represents the exact model state that achieved maximum anatomical overlap on unseen validation patients.
* **Immunity to Overfitting**: If training continues past the optimal point and the network begins to overfit or memorize edge artifacts, validation Dice plateaus or drops. The checkpoint mechanism preserves the peak generalizing weights without requiring manual early stopping.

---

## 6. Empirical Convergence & Diagnostics Profile

Evaluating the training progression from your log output reveals classic, healthy convergence dynamics for a 3D medical segmentation network:

| Epoch | Learning Rate ($\eta$) | Train Loss ($\overline{\mathcal{L}}$) | Val Dice | Val HD95 (mm) | Model Checkpoint Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $8.4 \times 10^{-5}$ | 0.9522 | 0.6770 (67.7%) | 26.24 mm | Saved (`best.pt`) |
| **2** | $1.38 \times 10^{-4}$ | 0.5857 | 0.7501 (75.0%) | 18.98 mm | Saved (`best.pt`) |
| **3** | $1.92 \times 10^{-4}$ | 0.4251 | 0.8031 (80.3%) | 9.71 mm | Saved (`best.pt`) |
| **5** | $3.00 \times 10^{-4}$ | 0.2773 | 0.8330 (83.3%) | 8.27 mm | Saved (`best.pt`) |
| **10** | $2.71 \times 10^{-4}$ | 0.1855 | 0.8558 (85.6%) | 6.55 mm | Saved (`best.pt`) |
| **15** | $1.97 \times 10^{-4}$ | 0.1654 | 0.8715 (87.2%) | 5.87 mm | Saved (`best.pt`) |
| **18** | $1.41 \times 10^{-4}$ | 0.1596 | 0.8830 (88.3%) | 4.97 mm | Saved (`best.pt`) |
| **22** | $7.0 \times 10^{-5}$ | 0.1463 | 0.8857 (88.6%) | 4.67 mm | Saved (`best.pt`) |
| **25** | $3.0 \times 10^{-5}$ | 0.1368 | 0.8876 (88.8%) | 4.30 mm | Saved (`best.pt`) |
| **26** | **$1.9 \times 10^{-5}$** | **0.1341** | **0.8887 (88.9%)** | **4.39 mm** | **Saved (`best.pt`) — PEAK** |
| **27** | $1.1 \times 10^{-5}$ | 0.1349 | 0.8873 (88.7%) | 4.25 mm | Retained Epoch 26 |

### Key Diagnostic Indicators
1. **Rapid Initial Boundary Refinement**: Between Epoch 1 and Epoch 3, Val HD95 plummeted from $26.24\text{ mm}$ to $9.71\text{ mm}$ (a **63% drop** in boundary error) as the network learned gross glioma spatial localization.
2. **Smooth Asymptotic Convergence**: Between Epoch 15 and Epoch 26, Train Loss decreased steadily from $0.1654$ to $0.1341$ while Val Dice increased from $87.15\%$ to **$88.87\%$**.
3. **No Severe Overfitting**: Train loss and validation Dice moved in lockstep. There is no sudden collapse of validation Dice or explosion of HD95, indicating that weight decay ($10^{-4}$), deep supervision regularization, and cosine decay maintained strong generalization throughout the 30-epoch budget.

---

## 7. Thesis Defense & Peer Review Reference Handbook

Use the following formal explanations in your thesis manuscript, defense presentation, or reviewer response:

### Q1: "Why do you not report validation loss during segmentation training?"
> **Defense**: "In medical volumetric segmentation, models are trained on continuous surrogate loss functions (such as combined Soft Dice and Binary Cross-Entropy) because discrete overlap metrics are non-differentiable. However, clinical benchmarks (e.g., MICCAI BraTS) evaluate algorithms based on volumetric Dice Similarity Coefficient (DSC) and boundary distance (HD95). Because cross-entropy loss can decrease due to overconfident background probability estimates without improving anatomical boundary segmentation, validation loss is uninformative. Our validation loop evaluates exact clinical metrics on binarized predictions at native resolution ($128^3$), directly matching the downstream evaluation criteria."

### Q2: "How is the best checkpoint selected if there is no validation loss?"
> **Defense**: "Checkpoints (`nnunet_3d_best.pt`, `visreg_jepa_multiscale_best.pt`, `unet_3d_best.pt`) are selected using the peak Validation Dice Similarity Coefficient (`best_val_dice`). This ensures that the saved weights represent the model state that maximizes clinical tumor overlap on holdout patients, avoiding suboptimal checkpoint selection caused by surrogate loss fluctuations."

### Q3: "What does the loss value shown in the tqdm progress bar represent?"
> **Defense**: "The progress bar postfix (`loss=...`) displays the instantaneous training loss of the most recent mini-batch of size $B=2$. It serves as a real-time monitor of gradient health and numerical stability. The true epoch training loss is computed by accumulating and averaging across all 473 mini-batches, reported in the epoch summary log as `Train Loss`."
