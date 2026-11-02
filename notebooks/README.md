# Research Notebook Suite (BraTS 2024 GLI Multiregion Protocol)

**Protocol: BraTS 2024 WT/TC/ET on the v2 dataset version.** 5-class outputs
(`0=Background`, `1=NETC`, `2=SNFH`, `3=ET`, `4=RC [Resection Cavity]`); evaluation
scores the three clinical glioma sub-regions (`WT={1,2,3}`, `TC={1,3}`, `ET={3}`)
and emits one row per `(model, region)`.

The notebook suite in **`notebooks/colab/`** is tailored for the Google Colab GPU environment (T4 / L4 / A100) with high-resolution $8^3$ voxel patches ($4,096$ token grid), Google Drive checkpoint persistence, isolated parallel execution directories, and 1-click downloadable zip archives.

---

## 1. Google Colab GPU Suite (`notebooks/colab/`)

Numbered sequentially for structured execution. Each notebook streams directly to an isolated subfolder under `My Drive/thesis/downstream/<model_name>/` and bundles all checkpoints, training curves, test metrics, low-data curves, and OOD tables into a standalone `<model_name>_outputs.zip`.

| Colab Notebook | Model & Role | Patch | Batch | Target Drive Output Folder | 1-Click Download Zip |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`01_colab_visreg_pretrain_3d.ipynb`** | 3D VisReg JEPA SSL Pre-training (200 epochs) | $8^3$ | 32 | `thesis/pretrain model/` | `visreg_pretrain_outputs.zip` |
| **`02_colab_finetune_visreg_multiscale_3d.ipynb`** | VisReg + Multi-Scale FPN Downstream (Primary Proposed) | $8^3$ | 4 | `thesis/downstream/visreg_multiscale/` | `visreg_multiscale_outputs.zip` |
| **`03_colab_finetune_visreg_unetr_3d.ipynb`** | VisReg + Hybrid UNETR Downstream (Ablation Decoder) | $8^3$ | 4 | `thesis/downstream/visreg_unetr/` | `visreg_unetr_outputs.zip` |
| **`04_colab_nnunet_3d.ipynb`** | 3D nnU-Net DynUNet Supervised Baseline | Dense | 2 | `thesis/downstream/nnunet/` | `nnunet_outputs.zip` |
| **`05_colab_unet_3d.ipynb`** | 3D Residual U-Net Supervised Baseline | Dense | 4 | `thesis/downstream/unet/` | `unet_outputs.zip` |
| **`06_colab_scratch_multiscale_3d.ipynb`** | 3D ViT + Multi-Scale FPN From Scratch (Zero Pre-training) | $8^3$ | 4 | `thesis/downstream/scratch_multiscale/` | `scratch_multiscale_outputs.zip` |
| **`07_colab_scratch_unetr_3d.ipynb`** | 3D ViT + Hybrid UNETR From Scratch (Zero Pre-training) | $8^3$ | 4 | `thesis/downstream/scratch_unetr/` | `scratch_unetr_outputs.zip` |
| **`08_colab_frozen_visreg_multiscale_3d.ipynb`** | VisReg + Multi-Scale FPN Frozen Encoder Probe | $8^3$ | 4 | `thesis/downstream/frozen_visreg_multiscale/` | `frozen_visreg_multiscale_outputs.zip` |
| **`09_colab_frozen_visreg_unetr_3d.ipynb`** | VisReg + Hybrid UNETR Frozen Encoder Probe | $8^3$ | 4 | `thesis/downstream/frozen_visreg_unetr/` | `frozen_visreg_unetr_outputs.zip` |

---

## 2. Low-Label Cell-by-Cell Early Stopping Suite (`notebooks/low_label/`)

A dedicated suite of 8 parallel Colab notebooks tailored specifically for rapid label-efficiency benchmarking across fractions $\{1\%, 5\%, 10\%, 25\%, 50\%\}$ using a modular **cell-by-cell execution pipeline** with dynamic **warmup-guarded Early Stopping** (`patience=5`, `min_epochs=5`, `max_epochs=50`, `min_steps=250`, `save_freq=10`):

- **Strict Monotonic Nested Sampling**: Every percentage strictly adds up cumulative patient volumes ($\mathcal{S}_{1\%} \subset \mathcal{S}_{5\%} \subset \mathcal{S}_{10\%} \subset \mathcal{S}_{25\%} \subset \mathcal{S}_{50\%}$) via deterministic quantile ranking (`seed=42`).
- **Dedicated Train Cell**: Executes per percentage, saving periodic checkpoints every 10 epochs and the best model (`_best.pt`) directly to Google Drive.
- **Dedicated Held-Out Evaluation Cell**: Immediately evaluates `_best.pt` on all 242 held-out test volumes, printing WT, TC, ET Dice, IoU, and HD95 live in the output.
- **Table 2 Compilation & Plotting**: Aggregates all fractions into canonical Table 2 format and renders the efficiency curve (`low_data_curve.png`).
- **1-Click Packaging**: Archives all checkpoints and logs to `<model>_low_data.zip` on Google Drive.

| Low-Label Notebook | Benchmark Architecture | Decoder | Status | Checkpoint Dependency | Target Drive Folder |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`01_low_label_visreg_multiscale.ipynb`** | 3D VisReg JEPA | Multi-Scale FPN | Fine-Tuned | `visreg_jepa_3d_best.pt` | `thesis/low_label/visreg_multiscale/` |
| **`02_low_label_visreg_unetr.ipynb`** | 3D VisReg JEPA | Hybrid UNETR | Fine-Tuned | `visreg_jepa_3d_best.pt` | `thesis/low_label/visreg_unetr/` |
| **`03_low_label_nnunet.ipynb`** | 3D nnU-Net DynUNet | Deep Supervision | Supervised | None (Trains from init) | `thesis/low_label/nnunet/` |
| **`04_low_label_unet.ipynb`** | 3D Residual U-Net | Residual Conv3D | Supervised | None (Trains from init) | `thesis/low_label/unet/` |
| **`05_low_label_scratch_multiscale.ipynb`** | 3D ViT Baseline | Multi-Scale FPN | Scratch | None (Zero pre-training) | `thesis/low_label/scratch_multiscale/` |
| **`06_low_label_scratch_unetr.ipynb`** | 3D ViT Baseline | Hybrid UNETR | Scratch | None (Zero pre-training) | `thesis/low_label/scratch_unetr/` |
| **`07_low_label_frozen_visreg_multiscale.ipynb`** | 3D VisReg JEPA | Multi-Scale FPN | Frozen Probe | `visreg_jepa_3d_best.pt` | `thesis/low_label/frozen_visreg_multiscale/` |
| **`08_low_label_frozen_visreg_unetr.ipynb`** | 3D VisReg JEPA | Hybrid UNETR | Frozen Probe | `visreg_jepa_3d_best.pt` | `thesis/low_label/frozen_visreg_unetr/` |

> **Parallel Execution & Zero Compute Waste**: Because each Colab notebook writes exclusively to its own directory, you can launch multiple Colab instances simultaneously in parallel without any file collision. Notebooks `02`, `03`, `08`, and `09` automatically consume `visreg_jepa_3d_best.pt` directly from `thesis/pretrain model/`; pre-training (`01`) never needs to be re-run once the checkpoint exists. You only need to run the specific downstream notebooks you wish to evaluate.

---

## Unified Configuration & Reproducibility Schema

All notebooks share a standardized top-level configuration block eliminating hardcoded CLI magic numbers:

```python
SEED = 42                 # Deterministic seed for weight init, augmentations, and data splits
NUM_WORKERS = 4          # Parallel CPU data loading workers (Colab GPU)
PATCH_SIZE = 8           # 8³ voxel patch (4,096 tokens) on Colab GPU
BATCH_SIZE = 4           # Volumetric downstream batch size (nnU-Net: B=2)
EPOCHS = 60              # Supervised training / fine-tuning epochs (pre-training: 200)
LOW_DATA_EPOCHS = 25     # Low-data label efficiency evaluation epochs
LEARNING_RATE = 3e-4     # Base AdamW learning rate (pre-training: 5e-4 with cosine schedule)
WEIGHT_DECAY = 1e-4      # AdamW weight decay (pre-training: 0.05)
```

CLI execution cells uniformly interpolate these parameters:
```bash
!python scripts/train_downstream_3d.py \
    --data_dir "{data_dir}" \
    --output_dir "{output_dir}" \
    --patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE} \
    --seed {SEED} \
    --epochs {EPOCHS} \
    --batch_size {BATCH_SIZE} \
    --num_workers {NUM_WORKERS} \
    --learning_rate {LEARNING_RATE} \
    --weight_decay {WEIGHT_DECAY} \
    --deep_supervision \
    --amp
```

---

## Complete Evaluation Battery

Every downstream notebook runs the three-tier empirical evaluation battery:
1. **Full-Data Held-Out Test Evaluation ($N=242$ volumes, TTA)**: Patient-level Dice (WT, TC, ET), 95% Hausdorff Distance (HD95), and volumetric IoU.
2. **Low-Data Label Efficiency Benchmark**: Measures performance across annotation fractions ($1\%, 5\%, 10\%, 25\%, 50\%, 100\%$) across 25 epochs.
3. **Out-of-Distribution (OOD) Scanner Shift Robustness**: Evaluates resilience under physical 3D Rician scanner noise ($\sigma=0.08$), quadratic RF $B_1$ field bias, and missing sequence triage (T1c-only, FLAIR-only).
4. **1-Click Packaging**: Automatically runs `export_artifacts` to zip all checkpoints, logs, and evaluation reports into `<model>_outputs.zip` in Google Drive.
