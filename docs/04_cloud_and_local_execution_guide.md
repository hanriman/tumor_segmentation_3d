# Cloud & Local Execution Guide (Google Colab GPU Suite & Local CLI)

*Step-by-step operational guide for data preparation, Google Colab GPU pre-training, downstream fine-tuning, and automated benchmark aggregation.*

---

## 1. Hardware Specifications & Optimization Strategy

| Environment | Accelerator | Usable VRAM | Recommended Batch Size ($B$) | Mixed Precision | Purpose |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Local / Workstation** | CPU / Apple Silicon | Shared RAM | $B=1$ (smoke tests) | FP32 / MPS | Unit testing, smoke tests & local artifact aggregation |
| **Google Colab GPU** | NVIDIA A100 / L4 / T4 | 16–80 GB | $B=32$ (Pretrain) / $B=4$ (Downstream) | `--amp` (BF16/FP16) | Canonical $8^3$ patch (4,096-token) SSL pre-training & downstream benchmark |

---

## 2. Canonical Google Colab Notebook Suite

The repository provides 9 self-contained Google Colab notebooks under [`notebooks/colab/`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/notebooks/colab) with unified CLI runner flags and automatic Google Drive synchronization:

```text
notebooks/colab/                                    # Google Colab GPU Suite (8³ patch / 4,096 tokens, Drive sync)
├── 01_colab_visreg_pretrain_3d.ipynb               # 3D VISReg JEPA SSL pre-training (200 epochs)
├── 02_colab_finetune_visreg_multiscale_3d.ipynb    # VISReg + Multi-Scale FPN downstream (60 epochs)
├── 03_colab_finetune_visreg_unetr_3d.ipynb         # VISReg + Hybrid UNETR downstream (60 epochs)
├── 04_colab_nnunet_3d.ipynb                        # 3D nnU-Net baseline (DynUNet with deep supervision)
├── 05_colab_unet_3d.ipynb                          # 3D Residual UNet baseline (with deep supervision)
├── 06_colab_scratch_multiscale_3d.ipynb            # From-scratch ViT-FPN ablation (zero pre-training)
├── 07_colab_scratch_unetr_3d.ipynb                 # From-scratch ViT-UNETR hybrid ablation (zero pre-training)
├── 08_colab_frozen_visreg_multiscale_3d.ipynb      # VISReg + FPN frozen encoder probe (--freeze_encoder)
└── 09_colab_frozen_visreg_unetr_3d.ipynb           # VISReg + UNETR frozen encoder probe (--freeze_encoder)
```

All downstream and baseline notebooks (`02`–`09`) share identical training hyperparameters (`SEED = 42`, `EPOCHS = 60`, `BATCH_SIZE = 4`, `PATCH_SIZE = 8`, `--deep_supervision`, `--amp`) and execute three evaluation stages automatically before packaging outputs to Google Drive:
1. **Holdout Test Evaluation (`scripts/evaluate_3d.py`)**: Computes WT, TC, ET Dice and HD95 (in canonical voxels) over the held-out test split ($N=242$) with 4-fold TTA (`--tta`), plus Effective Spectral Rank ($S^2$) for ViT models.
2. **Low-Data Label Efficiency (`scripts/evaluate_low_data_3d.py`)**: Evaluates $1\%, 5\%, 10\%, 25\%, 50\%, 100\%$ stratified training subsets (default 25 epochs / 75 gradient updates with `--min_steps 0`, selecting the best checkpoint on the validation split before evaluating once on the test split).
3. **OOD Scanner Shift & Missing-Modality Evaluation (`scripts/evaluate_ood_3d.py`)**: Evaluates robustness under 3D Rician noise, $B_1$ RF bias fields, and zero-imputed single-modality inputs (`FLAIR Only`, `T1ce Only`).

---

## 3. Data Preparation Workflow

### 3.1 Preprocess Raw NIfTI Volumes Locally
Run the parallel preprocessing pipeline with grouped `base_id` extraction (guaranteeing zero longitudinal patient leakage), parenchyma Z-score normalization, integer BraTS sub-region masks (`{0, 1, 2, 3}`), and compact `float16` storage:

```bash
# Preprocess using 8 CPU workers into canonical 128^3 .npz volumes
python scripts/prepare_data_3d.py --dtype float16 --num_workers 8
```

This produces `data/processed/brats_gli_3d_v2/` containing `metadata.csv` (with `split`, `volume_quartile`, and `has_et` columns; $1,144$ train / $235$ val / $242$ test) and the compressed `.npz` volumes. Zip this folder as `brats_gli_3d_v2.zip` for upload to Google Drive.

---

## 4. Execution on Google Colab

### 4.1 Google Drive `thesis/` Hierarchy
All Colab notebooks read the dataset and pre-trained checkpoint from `/content/drive/MyDrive/thesis/` and write their experiment outputs back to Google Drive:

```text
My Drive/
└── thesis/
    ├── data/
    │   └── brats_gli_3d_v2.zip                     <-- Upload preprocessed dataset archive here
    ├── visreg_pretrain_outputs/                    <-- Produced by Notebook 01
    │   ├── checkpoints/visreg_jepa_3d_best.pt
    │   └── visreg_pretrain_outputs.zip
    ├── visreg_multiscale_outputs/                  <-- Produced by Notebook 02
    ├── visreg_unetr_outputs/                       <-- Produced by Notebook 03
    ├── nnunet_outputs/                             <-- Produced by Notebook 04
    ├── unet_outputs/                               <-- Produced by Notebook 05
    ├── scratch_multiscale_outputs/                 <-- Produced by Notebook 06
    ├── scratch_unetr_outputs/                      <-- Produced by Notebook 07
    ├── frozen_visreg_multiscale_outputs/           <-- Produced by Notebook 08
    └── frozen_visreg_unetr_outputs/                <-- Produced by Notebook 09
```

### 4.2 Step-by-Step Launch Procedure
1. **Upload Dataset Archive**:
   - Place `brats_gli_3d_v2.zip` into `My Drive/thesis/data/`.
2. **Run Notebook 01 (`01_colab_visreg_pretrain_3d.ipynb`)**:
   - Mounts Google Drive, clones the repository, and extracts `brats_gli_3d_v2.zip` onto Colab's fast local NVMe SSD (`/content/data/brats_gli_3d_v2`).
   - Runs 200-epoch self-supervised 3D VISReg JEPA pre-training (`--patch_size 8 8 8`, `--batch_size 32`, `--epochs 200`, `--reg_warmup_epochs 30`, `--reg_min_scale 0.10`, `--amp`).
   - Runs `scripts/diagnose_visreg_3d.py` and packages `visreg_pretrain_outputs.zip` to `My Drive/thesis/visreg_pretrain_outputs/`.
3. **Run Downstream & Baseline Notebooks (`02` through `09`)**:
   - Notebooks `02`, `03`, `08`, and `09` automatically resolve `visreg_jepa_3d_best.pt` from `My Drive/thesis/visreg_pretrain_outputs/`.
   - Notebooks `04` (nnU-Net), `05` (ResUNet), `06` (Scratch ViT-FPN), and `07` (Scratch ViT-UNETR) require no pre-trained checkpoint and can be launched immediately in parallel with Notebook `01`.

---

## 5. 1-Click Local Benchmark Aggregation

Once cloud runs are complete, download the experiment output folders into `outputs/experiments/` on your local workstation and run:

```bash
python scripts/combine_and_generate_paper_artifacts.py
python paper/latex/figures/gen_figures.py
```

This workflow:
1. Parses all test, low-data, and OOD CSV files across all model experiments in `outputs/experiments/`.
2. Writes unified summary tables (`master_3d_benchmark.csv`, `low_data_3d_summary.csv`, `ood_3d_summary.csv`, `tables_latex.tex`) to `outputs/reports/`.
3. Packages `paper_artifacts.zip` into `outputs/reports/`.
4. Renders publication-grade vector PDF and PNG figures in `paper/latex/figures/`.

---

## 6. Output Architecture, Checkpoint Policy & Compact Packaging

### 6.1 Output Directory Sanitation (`outputs/`)
The top-level `outputs/` directory is strictly partitioned into two subdirectories:
- **`outputs/experiments/`**: Contains dedicated subfolders for each training experiment (e.g. `outputs/experiments/visreg_multiscale_outputs/`, `outputs/experiments/nnunet_outputs/`).
- **`outputs/reports/`**: Contains master benchmark CSVs, Markdown summaries, LaTeX tables, and `paper_artifacts.zip`.

Loose folders (`checkpoints/`, `logs/`, `metrics/`, `figures/`) are prohibited from existing directly in the `outputs/` root.

### 6.2 Zero-Flag Automatic Output Isolation
All training and evaluation scripts resolve their destination directory using `resolve_default_output_dir()` from `brats_jepa_3d.config`:
- **Under Tests & Smoke Tests** (`--smoke_test`, `pytest` active, `PYTEST_CURRENT_TEST` set, or `BRATS3D_TEST_MODE=1`): Artifacts automatically route to `tests/test_outputs/.smoke_tmp` (which is in `.gitignore`). **Zero CLI output flags are needed** to prevent repository pollution.
- **Under Standard Runs**: Omitting `--output_dir` automatically defaults to `outputs/experiments/<experiment_name>`, completely safeguarding the root.

### 6.3 Checkpoint Policy: Elimination of Duplicate `_latest.pt`
- Training scripts save strictly:
  1. `*_best.pt`: Anchored to best validation metric (`val_jepa_loss` for SSL pre-training; `val_dice` for downstream segmentation).
  2. `*_epoch_{N}.pt`: Periodic snapshots (if `save_freq > 0`) and guaranteed final epoch model (e.g. `epoch_60.pt` or `epoch_200.pt`).
- The redundant `_latest.pt` file was eliminated. Previously, `_latest.pt` was identical to `_epoch_{last_epoch}.pt`, wasting 100–300 MB of storage per experiment without being referenced by any evaluation or discovery routines.

### 6.4 Compact Google Drive Archive Packaging
At the conclusion of each Colab notebook, `export_artifacts()` archives artifacts for Google Drive:
- **`only_best_and_latest_checkpoints: bool = True` (Default)**: Automatically filters candidate checkpoints to include **only** the best model (`*_best.pt`) and the latest model (`*_epoch_{max_epoch}.pt`). All intermediate periodic checkpoints (`epoch_5.pt`, `epoch_10.pt`, ..., `epoch_55.pt`) are omitted from the zip archive.
- Reduces Google Drive archive sizes by **~80–90%** while preserving 100% of the diagnostic reports, metrics, and models needed for paper generation.
