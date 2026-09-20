# 🚀 Complete Guide: 3D Multimodal JEPA Training & Benchmarking on Kaggle GPU

This guide provides an end-to-end, step-by-step walkthrough for running the entire **BraTS 3D Volumetric Multimodal JEPA Representation Learning and Segmentation Benchmark** (`thesis_3d`) on Kaggle's free GPU resources (NVIDIA Tesla T4 16GB / P100 16GB).

---

## 1. Hardware & Acceleration on Kaggle

| Feature | Kaggle Free Tier Spec | Impact on 3D Volumetric Processing |
| :--- | :--- | :--- |
| **GPU Accelerator** | NVIDIA Tesla T4 (16 GB VRAM) or 2x T4 | Ample memory for 3D ViT patches ($8^3 = 512$ tokens) and 4-channel $128^3$ tensors |
| **Mixed Precision (AMP)** | Tensor Cores supported via FP16 | **~3.2x training speedup** with `--amp`, reducing 3D activation memory by >50% |
| **CPU & System RAM** | 4 vCPU cores, 30 GB System RAM | Enables fast memory-mapped caching of `.npz` volumetric arrays in `BraTS3DDataset` |
| **Runtime Limits** | Up to 12 hours per session / 30 hours per week | Ample for individual models; modular notebooks avoid session timeout |
| **Concurrent Sessions**| **Up to 2 active GPU sessions simultaneously** | **Train 3D VisReg on GPU 1 and 3D nnU-Net on GPU 2 at the exact same time!** |
| **Pre-installed Stack** | PyTorch 2.x, CUDA 12.x, torchvision, SciPy | Minimal setup overhead; only `monai`, `nibabel`, and `tabulate` needed |

---

## 2. Notebook Suite & Streamlined Local Aggregation Flow

To eliminate the hassle of downloading gigabytes of checkpoints only to re-upload them to Kaggle for evaluation, the repository uses a **Self-Contained Model Suite + 1-Click Local Aggregation** workflow:

```text
thesis_3d/notebooks/
├── 01_train_visreg_3d.ipynb          [~1h 09m (5 ep) / ~4.8 hrs (80 ep)]  Primary Method (SSL Pre-train + FPN + Test Eval + Low-Data + OOD)
├── 02_train_nnunet_3d.ipynb          [~1h 13m (5 ep) / ~4.0 hrs (30 ep)]  SOTA Baseline (DynUNet with Deep Supervision + Test Eval + Low-Data + OOD)
└── 03_train_unet_3d.ipynb            [~50 mins (5 ep) / ~2.5 hrs (30 ep)] Classical Baseline (MONAI Residual UNet + Test Eval + Low-Data + OOD)
```

### Why This Workflow is 10x Faster:
1. **Self-Contained Evaluation on Kaggle**: Each model notebook (01, 02, 03) evaluates its own checkpoints directly on the GPU (test split evaluation, low-data label efficiency, and OOD stress test) *before* exporting.
2. **Zero Cloud Re-Upload**: You download the 3 output archives (`visreg_outputs.zip`, `nnunet_outputs.zip`, `unet_outputs.zip`) directly to your local workstation.
3. **1-Click Local Aggregator**: Running `python scripts/combine_and_generate_paper_artifacts.py` aggregates all benchmarks, renders publication-grade vector PDF/PNG figures, formats LaTeX tables, and creates `paper_artifacts.zip` locally in **2 seconds**!

---

## 3. Step-by-Step Execution Workflow

### Step 1: Prepare the Dataset for Kaggle

You have two options depending on your preference:

#### Option A: Upload Preprocessed Volumes (Recommended for instant training)
1. On your local machine, run the multi-worker parallel preprocessing and packaging pipeline:
   ```bash
   # Preprocess with 8 parallel CPU workers and compact float16 storage
   python scripts/prepare_data_3d.py --dtype float16 --num_workers 8

   # Package into Kaggle-ready upload archive
   python scripts/package_for_kaggle.py
   ```
   This creates `dist_kaggle/brats_3d_full.zip` (~11 GB for all 1,621 volumes: train1_v2 + additional, grouped patient split; completed in ~3.5 mins).
   > **Note on `float16`**: Reduces disk and upload size by 50% (~6.7 MB per volume vs 13.8 MB) with negligible quantization error ($5.5 \times 10^{-5}$, two orders below MRI scanner noise). Arrays are automatically cast back to `float32` in RAM upon loading in `BraTS3DDataset`.
2. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) -> Click **New Dataset**.
3. Set the Title to: `brats-3d-full`.
4. Drag and drop `dist_kaggle/brats_3d_full.zip` and click **Create**.
5. The dataset will be mounted automatically at `/kaggle/input/brats-3d-full/` (archive prefix `brats_gli_3d_full/`).

#### Option B: Preprocess directly on Kaggle
Attach the raw BraTS 2024 GLI dataset on Kaggle, and the notebook will run:
```bash
python scripts/prepare_data_3d.py --limit 100 --dtype float16 --num_workers 4
```
This extracts non-zero bounding boxes and resamples volumes to canonical $128^3$ grids directly in `/kaggle/working/data/processed/brats_gli_3d_full` (or run `scripts/merge_and_resplit_3d.py` for the grouped patient split).

---

### Step 2: Running the Parallel Model Training Suite

#### Job 1 (GPU Session A): Run 3D VisReg JEPA
1. In Kaggle, click **Create** -> **New Notebook** -> **File** -> **Import Notebook** -> Upload [`notebooks/01_train_visreg_3d.ipynb`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/notebooks/01_train_visreg_3d.ipynb).
2. Attach dataset via **+ Add Input** -> `brats-3d-full`.
3. Set Accelerator to **GPU T4 x1** and turn **Internet ON**.
4. Click **Run All** (or **Save Version** -> **Run all with Save** to execute in the background).
5. Output: `visreg_outputs.zip` containing `visreg_jepa_best.pt`, downstream FPN weights, test split evaluations, low-data label efficiency curves, and OOD stress test metrics.

#### Job 2 (GPU Session B): Run 3D nnU-Net Baseline (in Parallel!)
1. Open a second Kaggle tab: **New Notebook** -> Import [`notebooks/02_train_nnunet_3d.ipynb`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/notebooks/02_train_nnunet_3d.ipynb).
2. Attach `brats-3d-full`, set **GPU T4 x1**, and click **Run All**.
3. Output: `nnunet_outputs.zip` containing DynUNet checkpoints, test metrics, and low-data curves.

#### Job 3: Run 3D Residual UNet Baseline
1. Import [`notebooks/03_train_unet_3d.ipynb`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/notebooks/03_train_unet_3d.ipynb).
2. Attach `brats-3d-full`, set **GPU T4 x1**, and click **Run All**.
3. Output: `unet_outputs.zip`.

---

### Step 3: Combine Results Locally & Generate Paper Artifacts (1-Click!)

Once Jobs 1, 2, and 3 are finished:
1. Download the three `.zip` files from Kaggle (`visreg_outputs.zip`, `nnunet_outputs.zip`, `unet_outputs.zip`).
2. Unpack them into your project's `outputs/` directory by experiment name:
   ```text
   outputs/
   ├── kaggle_visreg_5_epoch/
   │   ├── checkpoints/
   │   ├── metrics/
   │   └── logs/
   ├── kaggle_nnunet_5_epoch/
   │   ├── checkpoints/
   │   ├── metrics/
   │   └── logs/
   └── kaggle_unet_5_epoch/
       ├── checkpoints/
       ├── metrics/
       └── logs/
   ```
3. Run the local master aggregator script:
   ```bash
   python scripts/combine_and_generate_paper_artifacts.py
   ```
4. **In ~2 seconds**, this script will:
   - Automatically discover all experiment directories.
   - Aggregate test split metrics across all models into `outputs/master_3d_benchmark.csv` and `.md`.
   - Aggregate low-data label efficiency curves into `outputs/low_data_3d_summary.csv` and `.md`.
   - Aggregate OOD scanner shift evaluations into `outputs/ood_3d_summary.csv` and `.md`.
   - Generate all 4 publication-grade figures matching [`paper/latex/`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/paper/latex/) in both vector PDF and high-res PNG.
   - Format ready-to-paste LaTeX tables into `outputs/tables_latex.tex`.
   - Package all figures, tables, and metrics into **`outputs/paper_artifacts.zip`**.

---

## 4. Empirical Runtime & VRAM Budgeting (Tesla T4 GPU)

### Measured Durations from Completed Kaggle Runs:

| Task / Notebook | Measured Per-Epoch Duration | Total Duration (5-Epoch Verification) | Projected Duration (Full 30 / 50 Ep Budget) | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: |
| **01: 3D VisReg JEPA** (Pretrain + FPN + Low-Data + Eval) | Pretrain: $3.0\text{ min}$<br/>FPN: $3.5\text{ min}$ | **$1\text{h }09\text{m}$** | $\approx 4.5 - 5.0\text{ hours}$ ($50\text{ ep} + 30\text{ ep}$) | $\approx 7.8\text{ GB}$ |
| **02: 3D nnU-Net** (DynUNet + Deep Supervision + Low-Data + Eval) | $7.1\text{ min}$ | **$1\text{h }13\text{m}$** | $\approx 3.8 - 4.2\text{ hours}$ ($30\text{ ep}$) | $\approx 10.5\text{ GB}$ |
| **03: 3D Residual UNet** (MONAI UNet + Low-Data + Eval) | $3.5 - 3.7\text{ min}$ | **$50\text{ mins}$** | $\approx 2.5\text{ hours}$ ($30\text{ ep}$) | $\approx 8.4\text{ GB}$ |
| **Local Master Aggregator** (`combine_and_generate_paper_artifacts.py`) | - | **$\approx 2.2\text{ seconds}$** | $\approx 2.5\text{ seconds}$ | CPU |
| **Total Parallel Wall-Clock Time** (Jobs 1 & 2 concurrent on Kaggle) | - | **$\approx 1\text{h }15\text{m}$** | **$\approx 5.0\text{ hours}$** | $\le 11\text{ GB}$ |

Each notebook runs comfortably within Kaggle's **12-hour session limit** and **16 GB VRAM envelope**.

---

## 5. Experiment-First Directory Hierarchy

The project organizes all outputs first by experiment name to avoid collisions, allow parallel branch tracking, and enable seamless multi-run comparison:

```text
outputs/
├── <experiment_name>/          # e.g., kaggle_visreg_5_epoch, kaggle_nnunet_5_epoch, kaggle_unet_5_epoch
│   ├── checkpoints/            # Model weights (*_best.pt, *_epoch_*.pt)
│   ├── figures/                # Qualitative slice renders (*.png)
│   ├── logs/                   # Training logs and epoch step metrics (*.log, *.csv, *.json)
│   └── metrics/                # Benchmark metrics (master_3d_benchmark.csv, low_data_3d_summary.csv)
│
├── master_3d_benchmark.csv     # Combined master benchmark across all discovered experiments
├── master_3d_benchmark.md      # Formatted Markdown master table
├── low_data_3d_summary.csv     # Combined low-data efficiency comparison across all models
├── low_data_3d_summary.md      # Formatted Markdown low-data comparison
├── tables_latex.tex            # Ready-to-paste LaTeX tables for paper/latex/
├── figures/                    # Publication figures in dual vector PDF & PNG
└── paper_artifacts.zip         # 1-Click comprehensive download archive
```
