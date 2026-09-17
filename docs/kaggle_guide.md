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

## 2. Notebook Architectures: Modular vs. All-in-One

To ensure **zero risk of Kaggle's 12-hour session timeout** and enable **parallel GPU execution**, the repository provides both a **4-Notebook Modular Suite** (recommended for full training) and an **All-in-One Runner** (ideal for fast smoke-testing):

```text
thesis_3d/notebooks/
├── 01_train_visreg_3d.ipynb          [~3.0 - 3.5 hrs]  Primary Method (SSL Pre-training + FPN + Low-Data)
├── 02_train_nnunet_3d.ipynb          [~3.0 - 3.5 hrs]  SOTA Baseline (DynUNet with Deep Supervision)
├── 03_train_unet_3d.ipynb            [~2.0 - 2.5 hrs]  Classical Baseline (MONAI Residual UNet)
├── 04_evaluation_and_figures_3d.ipynb [~15 - 20 mins]   Master Benchmark, OOD Shifts & Paper Figures
└── kaggle_runner_3d.ipynb            [Interactive]     All-in-One Runner with Toggles (Dry-Run / Smoke Test)
```

### Why Use the Modular Suite?
1. **Zero Timeout Risk**: Each individual model finishes in 2 to 3.5 hours, well below Kaggle's 12-hour continuous execution cap.
2. **2x Speedup via Parallel GPUs**: Run `01_train_visreg_3d` on GPU 1 and `02_train_nnunet_3d` on GPU 2 simultaneously.
3. **Fault Isolation**: If a baseline needs re-running, you only re-run that notebook—not the preceding 10 hours.
4. **Seamless Output Chaining**: The evaluation notebook mounts checkpoints from previous notebooks via Kaggle's native `+ Add Input` -> `Your Work` feature.

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
   This creates `dist_kaggle/brats_3d_datasets.zip` (~9.1 GB for all 1,350 volumes; completed in ~2.5 mins).
   > **Note on `float16`**: Reduces disk and upload size by 50% (~6.7 MB per volume vs 13.8 MB) with negligible quantization error ($5.5 \times 10^{-5}$, two orders below MRI scanner noise). Arrays are automatically cast back to `float32` in RAM upon loading in `BraTS3DDataset`.
2. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) -> Click **New Dataset**.
3. Set the Title to: `brats-3d-datasets`.
4. Drag and drop `dist_kaggle/brats_3d_datasets.zip` and click **Create**.
5. The dataset will be mounted automatically at `/kaggle/input/brats-3d-datasets/`.

#### Option B: Preprocess directly on Kaggle
Attach the raw BraTS 2024 GLI dataset on Kaggle, and the notebook will run:
```bash
python scripts/prepare_data_3d.py --limit 100 --dtype float16 --num_workers 4
```
This extracts non-zero bounding boxes and resamples volumes to canonical $128^3$ grids directly in `/kaggle/working/data/processed/brats_gli_3d`.

---

### Step 2: Running the Modular Training Suite

#### Job 1 (GPU Session A): Run 3D VisReg JEPA
1. In Kaggle, click **Create** -> **New Notebook** -> **File** -> **Import Notebook** -> Upload [`notebooks/01_train_visreg_3d.ipynb`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/notebooks/01_train_visreg_3d.ipynb).
2. Attach dataset via **+ Add Input** -> `brats-3d-datasets`.
3. Set Accelerator to **GPU T4 x1** and turn **Internet ON**.
4. Click **Run All** (or **Save Version** -> **Run all with Save** to execute in the background).
5. Output: `visreg_outputs.zip` containing `visreg_jepa_best.pt`, downstream FPN weights, and low-data CSVs.

#### Job 2 (GPU Session B): Run 3D nnU-Net Baseline (in Parallel!)
1. Open a second Kaggle tab: **New Notebook** -> Import [`notebooks/02_train_nnunet_3d.ipynb`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/notebooks/02_train_nnunet_3d.ipynb).
2. Attach `brats-3d-datasets`, set **GPU T4 x1**, and click **Run All**.
3. Output: `nnunet_outputs.zip` containing DynUNet checkpoints and evaluation metrics.

#### Job 3: Run 3D Residual UNet Baseline
1. Import [`notebooks/03_train_unet_3d.ipynb`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/notebooks/03_train_unet_3d.ipynb).
2. Attach `brats-3d-datasets`, set **GPU T4 x1**, and click **Run All**.
3. Output: `unet_outputs.zip`.

---

### Step 3: Run Master Evaluation & Generate Paper Figures

Once Jobs 1, 2, and 3 are finished:
1. Import [`notebooks/04_evaluation_and_figures_3d.ipynb`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/notebooks/04_evaluation_and_figures_3d.ipynb).
2. In the top-right corner, click **+ Add Input** -> **Your Work** (or **Notebooks**):
   * Add the output of `01_train_visreg_3d`
   * Add the output of `02_train_nnunet_3d`
   * Add the output of `03_train_unet_3d`
   * Add `brats-3d-datasets`
3. Click **Run All**.
4. In ~15 minutes, this notebook will:
   - Compute exact test set metrics (Dice, IoU, cKDTree HD95 in mm, inference latency in ms per volume).
   - Evaluate Out-of-Distribution (OOD) shifts (3D Rician noise $\sigma=0.08$, $B_1$ field bias).
   - Evaluate Emergency Triage under missing pulse sequences (T1c-only, FLAIR-only).
   - Generate multi-planar orthogonal tumor visualizations (Axial, Coronal, Sagittal).
   - Generate all publication-grade figures matching [`paper/latex/`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/paper/latex/) in both vector PDF and high-res PNG.
   - Package everything into **`paper_artifacts.zip`** for instant 1-click download.

---

### Step 4: Using the All-in-One Runner (Fast Smoke Testing)

If you prefer testing everything in a single notebook before running full multi-hour experiments:
1. Import [`notebooks/kaggle_runner_3d.ipynb`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/notebooks/kaggle_runner_3d.ipynb).
2. Append `--smoke_test` to the commands to verify the entire pipeline in **~5 minutes**.
3. Toggle specific models on/off using the cell switches:
   ```python
   RUN_VISREG_PRETRAIN = True
   RUN_VISREG_FINETUNE = True
   RUN_NNUNET_BASELINE = False
   RUN_UNET_BASELINE = False
   ```

---

## 4. Runtime & VRAM Budgeting (Tesla T4 GPU)

| Task | Batch Size | Epochs | Estimated Time (T4 GPU + AMP) | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: |
| **01: 3D VisReg Pre-training (50 ep) + FPN (30 ep)** | $4 / 2$ | $50 + 30$ | $\approx 3.2\text{ hours}$ | $\approx 7.8\text{ GB}$ |
| **02: 3D nnU-Net Baseline (DynUNet with Deep Supervision)** | $2$ | $30$ | $\approx 3.0\text{ hours}$ | $\approx 10.5\text{ GB}$ |
| **03: 3D Residual UNet Baseline (MONAI)** | $2$ | $30$ | $\approx 2.0\text{ hours}$ | $\approx 8.4\text{ GB}$ |
| **04: Master Benchmark, OOD & Publication Figures** | $2$ | - | $\approx 15 - 20\text{ mins}$ | $\approx 5.2\text{ GB}$ |
| **Total Parallel Wall-Clock Time (Jobs 1 & 2 concurrent)** | - | - | **$\approx 3.5\text{ hours}$** | $\le 11\text{ GB}$ |

Each notebook runs safely within Kaggle's **12-hour session limit** and **16 GB VRAM envelope**.
