# Cloud & Local Execution Guide (Kaggle & Colab Pro A100)

*Step-by-step operational guide for data packaging, cloud GPU pre-training, downstream fine-tuning, and automated benchmark aggregation.*

---

## 1. Hardware Specifications & Optimization Strategy

| Environment | Accelerator | Usable VRAM | Recommended Batch Size ($B$) | Mixed Precision | Purpose |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Local / Workstation** | CPU / Apple Silicon | Shared RAM | $B=1$ (smoke tests) | FP32 / MPS | Unit testing & code verification |
| **Kaggle (Free Tier)** | NVIDIA Tesla T4 / P100 | 16 GB | $B=8$ (Pretrain) / $B=2$ (Downstream) | `--amp` (FP16) | Modular baseline & downstream fine-tuning runs |
| **Colab Pro / Cloud** | NVIDIA A100 SXM4 | 40 GB / 80 GB | $B=16$ (Pretrain) / $B=4$ (Downstream) | `--amp` (BF16/FP16) | Fast large-scale SSL pre-training (100–300 epochs) |

---

## 2. Canonical Notebook Suite Mapping

The repository organizes workflows into self-contained Jupyter notebooks with unified CLI runner compatibility and explicit top-level configuration blocks:

```text
notebooks/                                    # BraTS 2024 WT/TC/ET (5-class, v2 dataset)
├── 01_train_visreg_3d.ipynb               # 3D VisReg JEPA SSL pre-training (exports encoder checkpoint)
├── 02_diagnose_visreg_3d.ipynb            # Label-free pre-training masking diagnostics (D1–D3)
├── 03_train_nnunet_3d.ipynb               # 3D nnU-Net baseline (DynUNet with deep supervision)
├── 04_train_unet_3d.ipynb                 # 3D Residual UNet baseline
├── 05_train_vit_from_scratch_ablation_3d.ipynb # From-scratch ViT-FPN ablation (random initialization)
├── 06_vit_unetr_hybrid_ablation_3d.ipynb  # From-scratch ViT-UNETR hybrid ablation (convolutional stem)
├── 07_finetune_visreg_multiscale_3d.ipynb # VisReg + multiscale FPN downstream fine-tuning (Kaggle T4)
├── 08_finetune_visreg_unetr_3d.ipynb      # VisReg + hybrid UNETR downstream fine-tuning (Kaggle T4)
├── colab_a100_visreg_pretrain.ipynb       # Dedicated NVIDIA A100 SSL pre-training (Google Colab)
├── colab_a100_finetune_visreg_multiscale.ipynb # Dedicated NVIDIA A100 VisReg + multiscale FPN downstream
└── colab_a100_finetune_visreg_unetr.ipynb      # Dedicated NVIDIA A100 VisReg + hybrid UNETR downstream
```

All Kaggle downstream notebooks (`03`–`08`) share a standardized reproducibility configuration cell (`SEED = 42`, `NUM_WORKERS = 4`, `PATCH_SIZE = 16`, `BATCH_SIZE = 2`). The Colab A100 suite utilizes high-throughput batching (`BATCH_SIZE = 4` downstream, `BATCH_SIZE = 32` pretrain) and supports high-resolution `PATCH_SIZE = 8` (4,096 tokens) or canonical `PATCH_SIZE = 16`.

---


## 3. Data Preparation & Packaging Workflow

### 3.1 Step 1: Preprocess Volumes Locally
Run the parallel preprocessing pipeline with grouped `base_id` extraction (guaranteeing zero longitudinal patient leakage) and compact `float16` storage:

```bash
# Preprocess using 8 CPU workers into canonical 128^3 arrays
python scripts/prepare_data_3d.py --dtype float16 --num_workers 8
```

### 3.2 Step 2: Package Archive for Cloud Staging
```bash
# Generates Kaggle/Colab upload archive in dist_kaggle/brats_3d_full.zip (~11 GB)
python scripts/package_for_kaggle.py
```

---

## 4. Execution on Kaggle (T4 16GB)

### 4.1 Staging the Dataset
1. Navigate to [kaggle.com/datasets](https://www.kaggle.com/datasets) $\to$ **New Dataset**.
2. Title the dataset `brats-3d-full`.
3. Upload `dist_kaggle/brats_3d_full.zip` and create the dataset.
4. Kaggle mounts the files at `/kaggle/input/brats-3d-full/` (internal root `brats_gli_3d_full/`).

### 4.2 Running Modular Workflows
Each downstream notebook evaluates its checkpoints automatically across three evaluation regimes before generating output archives:
1. **Holdout Test Evaluation**: Computes WT, TC, ET Dice and HD95 (in canonical voxels) over held-out volumes ($N=242$).
2. **Low-Data Label Efficiency**: Evaluates performance under $1\%, 5\%, 10\%, 25\%$ label fractions.
3. **OOD Scanner Shift Evaluation**: Assesses robustness against Rician noise, $B_1$ field bias, and missing MRI modalities.

Outputs are exported as self-contained zip archives:
- `visreg_pretrain_outputs.zip`
- `visreg_unetr_outputs.zip`
- `visreg_multiscale_outputs.zip`
- `nnunet_outputs.zip`
- `unet_outputs.zip`

---

## 5. Execution on Google Colab (NVIDIA A100 SXM4)

For high-resolution self-supervised pre-training ($8^3$ voxel patch size $\to$ $16^3 = 4,096$ token grid) and downstream fine-tuning with full batch capacity ($B=32$ pretrain, $B=4$ downstream) and regularizer schedule annealing, use the dedicated Colab suite:
* `notebooks/colab_a100_visreg_pretrain.ipynb`: 200-epoch SSL pre-training with Anisotropic 3D Target Jitter.
* `notebooks/colab_a100_finetune_visreg_multiscale.ipynb`: Downstream fine-tuning with Multi-Scale 3D FPN decoder.
* `notebooks/colab_a100_finetune_visreg_unetr.ipynb`: Downstream fine-tuning with Hybrid UNETR decoder.

### 5.1 Google Drive `thesis/` Hierarchy
All Colab notebooks operate on a standardized, organized Google Drive directory layout:

```text
My Drive/
└── thesis/
    ├── data/
    │   └── brats-3d-v2.zip              <-- Upload your dataset .zip here
    ├── pretrain model/                  <-- Pre-training checkpoints and exported zip
    │   ├── visreg_jepa_3d_best.pt
    │   └── visreg_pretrain_outputs_a100.zip
    └── downstream/                      <-- Fine-tuning checkpoints and evaluation archives
        ├── multiscale/
        │   └── visreg_multiscale_outputs_a100.zip
        └── unetr/
            └── visreg_unetr_outputs_a100.zip
```

### 5.2 Step-by-Step Launch Procedure
1. **Prepare Google Drive**:
   - In Google Drive, create the folder `thesis/` with subfolders `data/` and `pretrain model/`.
   - Upload your `brats-3d-v2.zip` (or `brats_3d_full.zip`) into `My Drive/thesis/data/`.
2. **Open Notebook in Colab**:
   - Open `notebooks/colab_a100_visreg_pretrain.ipynb` from GitHub (`hanriman/tumor_segmentation_3d`).
3. **Select A100 GPU**:
   - Menu $\to$ **Runtime** $\to$ **Change runtime type**.
   - Hardware accelerator: **A100 GPU**, Runtime shape: **High-RAM**.
4. **Mount Drive & Local NVMe Extraction**:
   - Cell 1 mounts Google Drive.
   - Cell 6 automatically detects `brats-3d-v2.zip` in `/content/drive/MyDrive/thesis/data/` and unzips it once to `/content/data/` (local NVMe SSD), ensuring 100% GPU saturation without Google Drive FUSE latency.
5. **Execute Pre-training & Export**:
   - Runs pre-training with explicit parameters (`--patch_size 8 8 8`, `--batch_size 32`, `--epochs 200`, `--reg_warmup_epochs 30`, `--reg_min_scale 0.10`, `--amp`).
   - Exports weights directly to `My Drive/thesis/pretrain model/`.
6. **Execute Downstream Fine-Tuning**:
   - Open `notebooks/colab_a100_finetune_visreg_multiscale.ipynb` or `notebooks/colab_a100_finetune_visreg_unetr.ipynb`.
   - Both notebooks automatically detect the pre-trained weights in `My Drive/thesis/pretrain model/`, run 30-epoch fine-tuning, execute full test + low-data ($1\%$–$100\%$) + OOD evaluations, and package reports directly into `My Drive/thesis/downstream/`.

---

## 6. 1-Click Local Benchmark Aggregation

Once cloud runs are complete, download the output zip archives into `outputs/` on your local workstation and run:

```bash
python scripts/combine_and_generate_paper_artifacts.py
```

This automated aggregator:
1. Parses all test, low-data, and OOD CSV files across the 6 model evaluations.
2. Renders publication-grade vector PDF and PNG figures in `paper/latex/figures/`.
3. Formats camera-ready LaTeX tables (`table_main_benchmark.tex`, `table_low_data.tex`, etc.).
4. Builds the complete `paper_artifacts.zip` package in seconds.
