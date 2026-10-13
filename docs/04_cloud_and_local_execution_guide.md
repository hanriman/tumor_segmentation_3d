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

The repository organizes workflows into self-contained Jupyter notebooks with unified CLI runner compatibility:

```text
notebooks/                                    # BraTS 2024 WT/TC/ET (5-class, v2 dataset)
├── 01_train_visreg_3d.ipynb               # 3D VisReg JEPA SSL pre-training (exports encoder checkpoint)
├── 02_diagnose_visreg_3d.ipynb            # Label-free pre-training masking diagnostics (D1–D3)
├── 03_train_nnunet_3d.ipynb               # 3D nnU-Net baseline (DynUNet with deep supervision)
├── 04_train_unet_3d.ipynb                 # 3D Residual UNet baseline
├── 05_train_vit_from_scratch_ablation_3d.ipynb # From-scratch ViT-FPN ablation (random initialization)
├── 06_vit_unetr_hybrid_ablation_3d.ipynb  # From-scratch ViT-UNETR hybrid ablation (convolutional stem)
├── 07_finetune_visreg_multiscale_3d.ipynb # VisReg + multiscale FPN downstream fine-tuning
├── 08_finetune_visreg_unetr_3d.ipynb      # VisReg + hybrid UNETR downstream fine-tuning
└── colab_a100_visreg_pretrain.ipynb       # Dedicated NVIDIA A100 (Google Colab Web UI) pre-training
```

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

For large-scale self-supervised pre-training ($8^3 = 512$ token grid) with full batch capacity ($B=16$) and regularizer schedule annealing, use the dedicated notebook:
`notebooks/colab_a100_visreg_pretrain.ipynb`.

### 5.1 Why the Colab Web UI?
- **Seamless Secrets Integration**: Secure access to private Kaggle datasets using Colab's built-in Secret Manager (`KAGGLE_USERNAME`, `KAGGLE_KEY`).
- **Resilient Execution**: Does not drop background processes if a local VS Code remote SSH tunnel disconnects.
- **Drive Persistence**: Checkpoints and diagnostics stream directly to `/content/drive/MyDrive/checkpoints/visreg_jepa_a100_50ep/`.

### 5.2 Step-by-Step Launch Procedure
1. **Open the Notebook**: In Google Colab, open `notebooks/colab_a100_visreg_pretrain.ipynb` from GitHub (`hanriman/tumor_segmentation_3d`).
2. **Select A100 GPU**:
   - Menu $\to$ **Runtime** $\to$ **Change runtime type**.
   - Hardware accelerator: **A100 GPU**.
   - Runtime shape: **High-RAM**.
3. **Authenticate Private Kaggle Dataset**:
   - Open **Secrets** (key 🔑 icon) in the left sidebar.
   - Add `KAGGLE_USERNAME` and `KAGGLE_KEY` with notebook access toggled on (or upload `kaggle.json` interactively when prompted).
4. **Fast NVMe Extraction**:
   - Set `KAGGLE_DATASET = "yourusername/brats-3d-full"` in Cell 6.
   - The notebook downloads and unzips directly to `/content/data/` on the local NVMe, avoiding slow Google Drive I/O during DataLoader iterations.
5. **Pre-Flight Smoke Test**:
   - Runs a 2-minute verification pass:
     ```bash
     python scripts/train_jepa_3d.py --model_type visreg_jepa --smoke_test --reg_warmup_epochs 2 --reg_min_scale 0.10
     ```
6. **50-Epoch Production Training**:
   - Executes using the tuned A100 configuration ([`configs/experiment/colab_a100_pretrain.yaml`](../configs/experiment/colab_a100_pretrain.yaml)):
     ```bash
     python scripts/train_jepa_3d.py \
         --model_type visreg_jepa \
         --data_dir /content/data \
         --output_dir /content/outputs/visreg_jepa_a100 \
         --experiment_config configs/experiment/colab_a100_pretrain.yaml
     ```
   - Features: $B=16$, AMP FP16, Option B Anisotropic Target Masking Jitter (97 tokens across 6 cuboids), and Cosine Regularizer Annealing (15 warmup epochs, min scale 0.10).
7. **Diagnostics & Drive Mirroring**:
   - Executes `scripts/diagnose_visreg_3d.py` to plot Effective Rank, Cosine Similarity, and Feature Variance curves.
   - Automatically synchronizes all weights and metrics to Google Drive.

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
