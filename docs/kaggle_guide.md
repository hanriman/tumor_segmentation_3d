# 🚀 Complete Guide: 3D Multimodal JEPA Training & Benchmarking on Kaggle GPU

This guide provides an end-to-end, step-by-step walkthrough for running the entire **BraTS 3D Volumetric Multimodal JEPA Representation Learning and Segmentation Benchmark** (`thesis_3d`) on Kaggle's free GPU resources (NVIDIA Tesla T4 16GB / P100 16GB).

---

## 1. Hardware & Acceleration on Kaggle

| Feature | Kaggle Free Tier Spec | Impact on 3D Volumetric Processing |
| :--- | :--- | :--- |
| **GPU Accelerator** | NVIDIA Tesla T4 (16 GB VRAM) or 2x T4 | Ample memory for 3D ViT patches ($8^3 = 512$ tokens) and 4-channel $128^3$ tensors |
| **Mixed Precision (AMP)** | Tensor Cores supported via FP16 | **~3.2x training speedup** with `--amp`, reducing 3D activation memory by >50% |
| **CPU & System RAM** | 4 vCPU cores, 30 GB System RAM | Enables fast memory-mapped caching of `.npz` volumetric arrays in `BraTS3DDataset` |
| **Runtime Limits** | Up to 12 hours per session / 30 hours per week | Sufficient to run 50-epoch 3D SSL pre-training + downstream tasks + baselines |
| **Pre-installed Stack** | PyTorch 2.x, CUDA 12.x, torchvision, SciPy | Minimal setup overhead; only `monai`, `nibabel`, and `tabulate` needed |

---

## 2. Pipeline Architecture on Kaggle

```text
[Data Options]
  Option A (Upload Preprocessed):
    Local: python scripts/package_for_kaggle.py -> dist_kaggle/brats_3d_datasets.zip
    Kaggle: Upload to /kaggle/input/brats-3d-datasets/
  Option B (Preprocess in Notebook from Raw):
    Attach raw BraTS dataset -> python scripts/prepare_data_3d.py
       │
       ▼
[Kaggle Cloud Environment]
  GitHub Repo / Codebase
       │
       ▼  (!pip install -e .)
  /kaggle/working/thesis_3d/
       │
       ├── SSL Pre-training (I-JEPA, SigReg, VisReg)
       ├── Supervised Baselines (3D UNet, 3D nnU-Net DynUNet)
       ├── Downstream Fine-Tuning (Hierarchical 3D FPN)
       ├── Master Benchmark (Dice, IoU, cKDTree HD95, Latency, EffRank, CosSim)
       ├── Low-Data Efficiency (1% to 100% labels)
       ├── OOD Scanner Shifts (Rician noise, B1 field bias, missing modalities)
       └── Figure Generation (multi-planar orthogonal views, comparison charts)
       │
       ▼  (1-Click Download)
  /kaggle/working/outputs.zip (checkpoints, metrics CSV/MD, publication PDF/PNG figures)
```

---

## 3. Step-by-Step Instructions

### Step 1: Prepare the Dataset for Kaggle

You have two options depending on your preference:

#### Option A: Upload Preprocessed Volumes (Recommended for instant training)
1. On your local machine, run the packaging script:
   ```bash
   python scripts/package_for_kaggle.py
   ```
   This creates `dist_kaggle/brats_3d_datasets.zip`.
2. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) -> Click **New Dataset**.
3. Set the Title to: `brats-3d-datasets`.
4. Drag and drop `dist_kaggle/brats_3d_datasets.zip` and click **Create**.
5. The dataset will be mounted automatically at `/kaggle/input/brats-3d-datasets/`.

#### Option B: Preprocess directly on Kaggle
Attach the raw BraTS 2024 GLI dataset on Kaggle, and the notebook will run:
```bash
python scripts/prepare_data_3d.py --limit 100
```
This extracts non-zero bounding boxes and resamples volumes to canonical $128^3$ grids directly in `/kaggle/working/data/processed/brats_gli_3d`.

---

### Step 2: Create & Configure the Kaggle Notebook

1. In Kaggle, click **Create** -> **New Notebook**.
2. Go to **File** -> **Import Notebook** -> Upload [`notebooks/kaggle_runner_3d.ipynb`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/notebooks/kaggle_runner_3d.ipynb).
3. In the right-hand **Notebook Settings** panel:
   - **Accelerator**: Select **GPU T4 x1** (or **GPU T4 x2**).
   - **Internet**: Toggle to **On** (required for `pip install monai nibabel tabulate`).
   - **Persistence**: "Variables and files" (optional).
4. In the top-right corner, click **+ Add Input**:
   - If using Option A: Search for your uploaded `brats-3d-datasets` and click **Add**.
   - If using Option B: Search for `brats2024` or `brats-gli-2024` and click **Add**.

---

### Step 3: Run the Notebook Sections

The notebook is divided into modular, automated cells:

#### Section 1 & 2: Environment Verification & Dependencies
- Verifies CUDA allocation and GPU memory (`!nvidia-smi`).
- Installs `monai`, `nibabel`, and `tabulate` in ~15 seconds.

#### Section 3: Codebase Setup
- Clones or installs `thesis_3d` in editable mode (`pip install -e .`).
- Verifies that `brats_jepa_3d` imports cleanly and prints git commit hash.

#### Section 4: Dataset Discovery & Sanity Checks
- Verifies that `metadata.csv` and `.npz` volumes are discovered.
- Validates 4-channel $128^3$ tensor shapes and whole tumor mask distributions.

#### Section 5: Self-Supervised JEPA Pre-training (50 Epochs, AMP)
```bash
# 3D SigReg JEPA (Epps-Pulley Gaussianity test on 1D rays, scale factor N)
!python scripts/train_jepa_3d.py --model_type sigreg_jepa --epochs 50 --batch_size 4 --amp

# 3D VisReg JEPA (Decoupled Center, Scale, Shape SWD)
!python scripts/train_jepa_3d.py --model_type visreg_jepa --epochs 50 --batch_size 4 --amp

# 3D I-JEPA (EMA Teacher baseline)
!python scripts/train_jepa_3d.py --model_type ijepa --epochs 50 --batch_size 4 --amp
```
*Tip: For a rapid dry-run smoke test, append `--smoke_test` or use `--epochs 2`.*

#### Section 6: Supervised 3D Baselines (30 Epochs, AMP)
```bash
# 3D Residual UNet (MONAI)
!python scripts/train_unet_3d.py --epochs 30 --batch_size 4 --amp

# 3D nnU-Net DynUNet with Deep Supervision (MONAI)
!python scripts/train_nnunet_3d.py --epochs 30 --batch_size 2 --amp
```

#### Section 7: Downstream 3D Volumetric Fine-Tuning (30 Epochs, AMP)
```bash
# Fine-tune with Hierarchical 3D Multi-Scale FPN Decoder
!python scripts/train_downstream_3d.py --model_type sigreg_jepa --decoder_type multiscale --epochs 30 --amp
!python scripts/train_downstream_3d.py --model_type visreg_jepa --decoder_type multiscale --epochs 30 --amp
!python scripts/train_downstream_3d.py --model_type ijepa --decoder_type multiscale --epochs 30 --amp
```

#### Section 8: Master 3D Volumetric Benchmark Evaluation
```bash
!python scripts/evaluate_3d.py --batch_size 2 --amp
```
Evaluates all models on the test split:
- **3D Dice Score (%)** & **3D IoU (%)**
- **Exact 3D 95th Percentile Hausdorff Distance (HD95 mm)** via `cKDTree`
- **Inference Latency (ms / 128³ volume)**
- **Effective Rank ($S_k^2$)** & **Centered Cosine Similarity**

#### Section 9: Low-Data Label Efficiency Benchmark
```bash
!python scripts/evaluate_low_data_3d.py --fractions 0.01 0.05 0.10 0.25 0.50 1.00 --amp
```
Quantifies label efficiency of SSL pre-trained encoders against supervised baselines across label availability regimes ($1\%$ to $100\%$).

#### Section 10: Out-of-Distribution (OOD) Scanner Shift Benchmark
```bash
!python scripts/evaluate_ood_3d.py --amp
```
Evaluates robustness across:
1. **3D Rician Scanner Noise** ($\sigma_{\text{noise}} = 0.08$)
2. **3D RF Coil B1 Field Bias Inhomogeneity** (2nd-order polynomial)
3. **Missing MRI Sequences** (T1c-only and FLAIR-only triage)

#### Section 11 & 12: Publication Figures & 1-Click Output Download
- Runs `python scripts/generate_figures_3d.py` to generate:
  - Multi-planar orthogonal slice visualization ([Axial, Coronal, Sagittal](file:///Users/hanriman/Documents/master/thesis/thesis_3d/outputs/figures/orthogonal_multi_planar_figure.png))
  - Benchmark metric comparison bar charts
  - Low-data efficiency curves
  - OOD robustness degradation bar charts
- Zips all checkpoints, metrics, and figures into `/kaggle/working/outputs.zip` for instant 1-click download.

---

## 4. Runtime & VRAM Budgeting (Tesla T4 GPU)

| Task | Batch Size | Epochs | Estimated Time (T4 GPU + AMP) | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: |
| **3D SigReg JEPA Pre-training** | $4$ | $50$ | $\approx 55\text{ min}$ | $\approx 7.2\text{ GB}$ |
| **3D VisReg JEPA Pre-training** | $4$ | $50$ | $\approx 52\text{ min}$ | $\approx 6.8\text{ GB}$ |
| **3D I-JEPA Pre-training (EMA)** | $4$ | $50$ | $\approx 68\text{ min}$ | $\approx 9.5\text{ GB}$ |
| **3D Downstream Fine-Tuning** | $4$ | $30$ | $\approx 35\text{ min}$ | $\approx 7.8\text{ GB}$ |
| **3D Residual UNet Baseline** | $4$ | $30$ | $\approx 38\text{ min}$ | $\approx 8.4\text{ GB}$ |
| **3D nnU-Net (DynUNet)** | $2$ | $30$ | $\approx 48\text{ min}$ | $\approx 10.5\text{ GB}$ |
| **Evaluation & Probing Suite** | $2$ | - | $\approx 10\text{ min}$ | $\approx 5.2\text{ GB}$ |
| **Full End-to-End Run** | - | - | $\approx 5.2\text{ hours}$ | $\le 11\text{ GB}$ |

All stages easily execute well within Kaggle's **12-hour continuous runtime window** and **16 GB VRAM envelope**.
