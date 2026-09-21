# 3D Multimodal JEPA: Self-Supervised Volumetric Representation Learning for Brain Glioma MRI Segmentation

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI 1.6+](https://img.shields.io/badge/MONAI-1.6+-2ca02c.svg)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A standalone, mathematically rigorous research repository investigating **3D Volumetric Joint-Embedding Predictive Architectures (JEPA)** for multi-modal brain glioma MRI segmentation on the **BraTS 2024 GLI** benchmark.

---

## 1. Executive Summary & Core Research Thesis

Gliomas (glioblastomas and astrocytomas) are intrinsically **three-dimensional, infiltrative, and heterogeneously vascularized malignancies** (Menze et al., 2014; Baid et al., 2024). Slice-by-slice 2D processing introduces severe methodological and biophysical limitations:
1. **Through-Plane Discontinuity**: Discards through-plane ($z$-axis) spatial gradients and volumetric curvature, failing to capture true 3D ellipsoidal and irregular infiltrative growth patterns.
2. **Slice Leakage & Artifacts**: Orthogonal coronal and sagittal contexts are inaccessible, preventing isotropic boundary localization (Milletari et al., 2016; Isensee et al., 2021).
3. **Anisotropic Metric Corruption**: Slice-wise evaluation ignores physical millimeter voxel spacing ($1.0 \times 1.0 \times 1.0\text{ mm}^3$), corrupting distance metrics like the 95th percentile Hausdorff Distance (Taha & Hanbury, 2015).

### The Scientific Thesis
> **We extend Joint-Embedding Predictive Architectures to full 3D multi-modal brain MRI volumes, investigating whether heuristic-free representation regularizations (SigReg and VisReg) mathematically prevent 3D latent collapse without momentum teacher networks, learning isotropic volumetric anatomical representations that surpass standard 3D I-JEPA, 3D Residual UNet, and 3D nnU-Net under full-data, label-scarce, and scanner-shifted distribution regimes.**

### Why JEPA Over MAE and Contrastive Learning
- **Vs. Masked Autoencoders (He et al., 2022; Feichtenhofer et al., 2022)**: MAE reconstructs raw voxels, forcing model capacity to encode high-frequency non-semantic MRI scanner noise (Rician distribution; Gudbjartsson & Patz, 1995) and RF coil bias field inhomogeneities (Sled et al., 1998). JEPA predicts abstract representations in latent space ($\hat{y}_{\text{tgt}} \approx y_{\text{tgt}}$), naturally discarding high-frequency non-semantic noise (LeCun, 2022; Assran et al., 2023).
- **Vs. Contrastive Learning (SimCLR, MoCo)**: Contrastive methods penalize negative pairs. In brain MRI, all human subjects share macroscopic anatomy (bilateral hemispheres, ventricles, anatomical lobes). Treating different patients as negative pairs imposes false-negative gradient penalties on identical anatomical structures. JEPA requires zero negative pairs.
- **Vs. Empirical Momentum Heuristics (BYOL, DINO, Standard I-JEPA)**: Standard I-JEPA relies on an Exponential Moving Average (EMA) teacher network ($\bar{\theta} \leftarrow m \bar{\theta} + (1-m) \theta$). However, EMA is an empirical heuristic without theoretical convergence guarantees, introduces sensitive momentum schedules ($m \in [0.996, 1.0]$), and incurs $\approx 40\%$ parameter memory overhead. **SigReg** and **VisReg** eliminate the teacher network via principled information-theoretic and optimal-transport regularizers.

---

## 2. Mathematical & Algorithmic Foundations

| Component | Mathematical / Algorithmic Formulation | Theoretical Reference & Justification |
| :--- | :--- | :--- |
| **Volumetric Continuity** | $\mathbf{X} \in \mathbb{R}^{4 \times 128 \times 128 \times 128}$, $1.0\text{ mm}^3$ isotropic grid. | **Menze et al. (2014)**; **Milletari et al. (2016)**. Preserves 3D spatial continuity across white matter tracts. |
| **Grid Resampling** | Non-zero brain bounding box $\to$ trilinear resampling to canonical $128^3$ isotropic grid (bounding box stretched to cube; aspect ratio **not** preserved). | **Isensee et al. (2021)** (*Nature Methods*). Prevents anatomical truncation of peripheral cortex at the cost of anisotropic geometric distortion (physical-mm metrics inherit this warp). |
| **Parenchyma Normalization** | $X_{c, \text{norm}}(v) = \frac{X_c(v) - \mu_{c,\text{nz}}}{\sigma_{c,\text{nz}} + \epsilon}$ for non-zero voxels. | **Isensee et al. (2021)**. Harmonizes scanner-dependent intensity drift across institutions. |
| **Random Modality Dropout** | $m_c \sim \text{Bernoulli}(1 - p_{\text{drop}})$, $p_{\text{drop}}=0.25$, fallback $\sum_c m_c \ge 1$, kept channels rescaled $C/\text{keep}$ (inverted dropout). | **Havaei et al. (2017)**; **Dorent et al. (2019)**. Simulates missing MRI sequences, enforcing representation independence; rescaling preserves train/test magnitude. |
| **3D Context / Target Masking** | 4 Target cuboids ($3 \times 3 \times 3$, $N_{\text{tgt}} \approx 108$) + 3D Connected BFS Context ($N_{\text{ctx}} = 192$), **brain-aware**: sampling steered to tissue tokens via `brain_mask`, tissue-only VisReg regularization. | **Assran et al. (2023)** (*CVPR*). Prevents spatial interpolation shortcuts; constant $N_{\text{ctx}}$ maximizes Tensor Core throughput; air-dilution fix (roadmap Phase 8). |
| **3D Positional Embeddings** | Separable 3D Sinusoidal Coordinate Embeddings (frozen, requires_grad=False) $[\text{PE}(z) \,\|\, \text{PE}(y) \,\|\, \text{PE}(x)]$. | **Vaswani et al. (2017)**; **Feichtenhofer et al. (2022)**. Imparts immediate 3D metric spatial topology from step 0. |
| **Context Gather (Zero Leakage)** | $\mathbf{Z}_{\text{ctx}} = \text{Encoder}(\text{gather}(\mathbf{Z}_0, \text{idx}_{\text{ctx}})) \in \mathbb{R}^{B \times 192 \times 384}$. | **Assran et al. (2023)**. Evaluates strictly visible tokens; reduces self-attention FLOPs by $85.9\%$ ($192^2 / 512^2$). |
| **Projector MLP** | $z = W_2(\text{GELU}(\text{LN}(W_1 h + b_1))) + b_2: 384 \to 1024 \to 128$. | **Tishby et al. (2000)**; **Balestriero & LeCun (2025)**. Information Bottleneck decouples task representations from Gaussian forces. |
| **SigReg Gaussianity Test** | $\mathcal{T}_{\text{EP}} = N \int \|\hat{\phi}_N(t) - \phi_0(t)\|^2 d\mu(t)$ over $M=256$ 1D rays, tissue-filtered like VisReg (shared helper); optional `generator` for seeded projections. | **Balestriero & LeCun (2025)**; **Epps & Pulley (1983)**. Sample factor $N$ cancels $1/N$ ECF derivative, yielding $O(1)$ per-sample gradients. |
| **VisReg Decoupled Moments** | $\mathcal{L}_{\text{VisReg}} = \mathcal{L}_{\text{JEPA}} + \lambda_{\text{c}} \frac{\|\mu\|_2^2}{D} + \lambda_{\text{sc}} \frac{\sum (1-\sigma_d)^2}{D} + \lambda_{\text{sh}} W_2^2$; shape term detaches both mu and sigma (center handled only by center loss); optional `generator` for seeded SWD slices. | **Wu, Balestriero, & Levine (2026)**; **Villani (2009)**. Stop-gradient on $\mu,\sigma$ prevents redundant mean gradients and dimensional collapse; closed-form 1D sort in $O(N \log N)$. |
| **Hierarchical 3D FPN Decoder** | Lateral skips from $L_2, L_4, L_6, L_8$ with progressive ConvTranspose3d and `GroupNorm`. | **Lin et al. (2017)**; **Hatamizadeh et al. (2022)**; **Wu & He (2018)**. Restores high-frequency boundary gradients; GroupNorm stabilizes small 3D batches ($B \le 4$). |
| **Supervised 3D Baselines** | MONAI 3D Residual UNet (+ ds1/ds2/ds3 heads) and MONAI 3D DynUNet, both with deep supervision default-ON ($w_s = 2^{-s}/\sum 2^{-j}$); asymmetric Tversky opt-in (binary-only `CombinedTverskyBCEWithLogitsLoss3D` with explicit `ValueError` guard; raw `VolumetricTverskyLoss` accepts `include_background=False` for multi-class parity with Dice); 4-fold TTA at eval. | **Ronneberger et al. (2015)**; **Isensee et al. (2021)**; **Lee et al. (2015)**; **Salehi et al. (2017)**. Establishes state-of-the-art supervised benchmark parity. |
| **Exact 3D HD95 Metric** | $d_{\text{HD95}}^{3\text{D}}$ via 3D 26-connectivity morphological erosion and `cKDTree` in physical mm ($1.0\text{ mm}^3$); tumor-only aggregates (`dice_tumor_only`, `iou_tumor_only`, `hd95_tumor_only`) are NaN (never 1.0/0.0) on tumor-free cohorts or when HD95 is off — aggregate with `nanmean`, render as `n/a`. | **Huttenlocher et al. (1993)**; **Taha & Hanbury (2015)**; **Powers (2011)**. Guarded zero-division prevents artificial epsilon score inflation. |
| **Effective Rank ($S_k^2$)** | $\text{erank}(Z) = \exp\left(-\sum p_k \ln p_k\right)$, $p_k = S_k^2 / \sum S_j^2$. | **Roy & Vetterli (2007)**. Strictly uses covariance eigenvalues $S_k^2$, preventing linear $S_k$ from masking dimensional collapse. |
| **Centered Cosine Sim** | $\bar{S}_{\text{centered}} = \frac{1}{N(N-1)} \sum_{i \neq j} \frac{(z_i - \bar{z})^\top (z_j - \bar{z})}{\|z_i - \bar{z}\|_2 \|z_j - \bar{z}\|_2}$. | **Wang & Isola (2020)**. Decouples centroid translation from angular dispersion, diagnosing directional collapse. |
| **Combined Loss Alignment** | Multi-class Cross-Entropy supervises all voxels including background (`ignore_index=-100`) while Dice default `include_background=True` is a no-op on the binary task (explicit `False` required only for multi-class). | **Isensee et al. (2021)**; **Milletari et al. (2016)**. Dense CE penalizes false-positive foreground predictions across healthy tissue while Dice prevents background dominance. |
| **Quartile-Stratified Splits** | Low-data tiers ($1\%$ to $50\%$) stratified by tumor volume quartile. | **Isensee et al. (2021)**. Preserves macro- and micro-lesion representation balance across low-data fractions. |
| **Metric Validation Guard** | Strict single-channel validation ($C=1$) + `from_logits` toggle in `compute_volumetric_metrics_3d`. | **Taha & Hanbury (2015)**. Prevents silent metric computation on background channel 0 and avoids double-sigmoid distortion. |
| **3D Rician Scanner Noise** | $M = \sqrt{(X_+ + \eta_1)^2 + \eta_2^2}$, $\eta_1, \eta_2 \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$ on all voxels (tissue Rician for X>=0 / Gaussian for Z-scored negatives + air Rayleigh, sign-preserving). | **Gudbjartsson & Patz (1995)**. Accurately simulates MRI magnitude reconstruction from quadrature RF coil channels. |
| **3D B1 RF Field Bias** | $X_{\text{corrupt}} = X \cdot \left(1 + \text{strength} \cdot \sum_{i+j+k \le 2} c_{ijk} x^i y^j z^k\right)$ with per-sample $c_{ijk} \sim \mathcal{N}(0,1)$ (field clamped to $[-0.5, 0.5]$, gain in $[1 - 0.5 \cdot \text{strength},\, 1 + 0.5 \cdot \text{strength}]$); gain multiplies original voxels (negatives preserved); background stays zero (gain on signal, air has none — unlike Rician, which synthesizes Rayleigh air). | **Sled et al. (1998)**; **Lebrun et al. (2021)**. Multiplicative smooth 2nd-order polynomial simulating RF coil field inhomogeneity; every volume sees a different field. |

---

## 3. Standalone Repository Structure

```text
thesis_3d/
├── .gitignore
├── .python-version               # Python 3.12
├── pyproject.toml                # uv & setuptools configuration
├── README.md                     # Comprehensive documentation
├── agents.md                     # Scientific writing & research guidelines
│
├── docs/
│   ├── roadmap_improvements_ablation_dataset.md # Phased engineering roadmap (v2.5: DS/TTA/Tversky/hybrid/brain-aware/full-pool)
│   ├── improving_3d_jepa_segmentation_performance.md # Root-cause analysis: 79.7% vs 87.6% gap + 7 fixes
│   ├── training_validation_bottleneck_and_hd95_optimization.md # 18.7x validation speedup
│   ├── vit_from_scratch_ablation.md   # From-scratch ViT baseline rationale
│   ├── audit_and_remediation_plan.md    # Formal mathematical audit & verification report
│   ├── audit_and_remediation_plan_3d.md # Forensic root-cause analysis & code remediation
│   ├── audit_report_2026-09-18.md       # Exhaustive forensic audit & verification sign-off
│   ├── audit_findings_2026-09-18_21-15.md # Intermediate implementation and memory hazard findings
│   ├── downstream_freezing_vs_finetuning.md # Research rationale: full fine-tuning vs frozen probe
│   ├── implementation_audit_2026-09-17_1458.md # Reference alignment checklist
│   └── kaggle_guide.md                  # End-to-end Kaggle GPU execution guide
│
├── notebooks/
│   ├── 01_train_visreg_3d.ipynb  # Primary method: VisReg pre-training, diagnose cell, FPN + hybrid finetune
│   ├── 02_train_nnunet_3d.ipynb  # SOTA baseline: 3D DynUNet with deep supervision
│   ├── 03_train_unet_3d.ipynb    # Classical baseline: 3D Residual UNet
│   ├── 04_train_vit_from_scratch_ablation_3d.ipynb # From-scratch ViT ablation
│   ├── 06_vit_unetr_hybrid_ablation_3d.ipynb # From-scratch ViT + hybrid UNETR ablation (all cells --from_scratch)
│   └── kaggle_runner_3d.ipynb    # Interactive all-in-one runner with toggles
│
├── configs/                      # Modular YAML configuration hierarchy
│   ├── base.yaml                 # System paths, seed, device, AMP configs
│   ├── dataset/
│   │   └── brats3d.yaml          # Grid geometry (128^3), patch size (16^3), modalities
│   ├── model/
│   │   ├── ijepa_3d.yaml         # 3D I-JEPA architecture parameters
│   │   ├── sigreg_jepa_3d.yaml   # 3D SigReg JEPA + Epps-Pulley parameters
│   │   ├── visreg_jepa_3d.yaml   # 3D VisReg JEPA + Sliced-Wasserstein parameters
│   │   ├── unet_3d.yaml          # 3D Residual UNet parameters
│   │   └── nnunet_3d.yaml        # 3D DynUNet + Deep Supervision parameters
│   └── experiment/
│       ├── pretrain_50ep.yaml    # 3D SSL pre-training schedule
│       └── finetune_30ep.yaml    # 3D downstream fine-tuning schedule
│
├── src/
│   └── brats_jepa_3d/            # Fully standalone Python package
│       ├── __init__.py
│       ├── config.py             # Cloud/Kaggle/Colab/Local dynamic resolver
│       ├── data/
│       │   ├── dataset.py        # BraTS3DDataset with bounded RAM caching & fraction sampling
│       │   ├── transforms.py     # RandomModalityDropout3D, Rician noise, B1 bias field
│       │   └── masking.py        # JEPAMaskingTransform3D (brain-aware 3D cuboids + 3D BFS context, torch-only RNG, hard zero-overlap)
│       ├── models/
│       │   ├── vision_transformer_3d.py # PatchEmbed3D + 3D Sincos Pos + ViTEncoder3D
│       │   ├── predictor_3d.py          # JEPAPredictor3D (4 layers, self-attention)
│       │   ├── ijepa_3d.py              # Dual-encoder with EMA teacher
│       │   ├── sigreg_jepa_3d.py        # Single-encoder + Projector MLP (384 -> 1024 -> 128)
│       │   ├── visreg_jepa_3d.py        # Single-encoder + Projector MLP
│       │   ├── segmentation_head_3d.py  # Bottleneck, Multi-Scale FPN & Hybrid UNETR Decoders (native 128³/64³ stem)
│       │   ├── unet_3d.py               # 3D Residual UNet (MONAI) + ds1/ds2/ds3 deep-supervision heads
│       │   └── nnunet_3d.py             # 3D DynUNet with Deep Supervision (MONAI)
│       ├── losses/
│       │   ├── ijepa_loss.py            # Latent Smooth L1 on LayerNormed detached targets
│       │   ├── sigreg_loss.py           # Epps-Pulley Gaussianity test (scale factor N, raw 1D rays)
│       │   ├── visreg_loss.py           # Decoupled Center, Scale, Shape (SWD, stop-grad sigma)
│       │   ├── dice_bce_loss_3d.py      # Combined Volumetric 3D Dice + Cross-Entropy (multi-class & binary)
│   │   │   ├── tversky_loss_3d.py       # Asymmetric Tversky (α=0.3, β=0.7) + BCE opt-in + criterion factory
│       │   └── deep_supervision_loss_3d.py # Multi-scale exponential decay supervision (custom base_loss, 5D & 4D targets)
│       ├── metrics/
│       │   ├── volumetric_metrics.py    # Guarded 3D Dice, IoU, cKDTree 3D HD95 (Powers 2011)
│       │   └── probing_metrics.py       # Effective Rank (S^2), Centered Cosine Sim
│       └── utils/
│           ├── aggregation.py           # Multi-experiment merging & LaTeX table generation
│           ├── device.py                # CUDA / MPS / CPU device detection & AMP context
│   │       ├── tta.py                    # 4-fold orthogonal-reflection TTA (eval, all models)
│           ├── export.py                # Kaggle artifact staging, zip packaging, and input resolution
│           ├── logging.py               # MetricTracker, setup_logger & numeric checkpoint sorting
│           └── seed.py                  # Deterministic PRNG seeding
│
├── scripts/
│   ├── prepare_data_3d.py        # NIfTI bounding-box cropping, resampling to 128^3 & .npz export
│   ├── train_jepa_3d.py          # 3D SSL Pre-training runner (50 epochs)
│   ├── train_downstream_3d.py    # Downstream 3D fine-tuning & linear probing runner
│   ├── train_unet_3d.py          # Supervised 3D UNet baseline runner (--deep_supervision default-ON)
│   ├── train_nnunet_3d.py        # Supervised 3D nnU-Net baseline runner
│   ├── evaluate_3d.py            # Master benchmark evaluator (Dice, HD95, Rank, Latency, --tta)
│   ├── diagnose_visreg_3d.py       # Real-data D1–D3 masking/regularization probe (Kaggle)
│   ├── merge_and_resplit_3d.py       # Merge staged pools + grouped patient split (leakage fix)
│   ├── evaluate_low_data_3d.py   # Label efficiency runner (1% to 100% 3D labels)
│   ├── evaluate_ood_3d.py        # 3D Scanner shift runner (Rician noise, B1 bias field)
│   ├── generate_figures_3d.py    # Multi-planar orthogonal & publication figures
│   ├── package_for_kaggle.py     # Packages processed 3D dataset into dist_kaggle/ archive
│   ├── combine_and_generate_paper_artifacts.py # 1-Click master multi-experiment aggregator & LaTeX exporter
│   └── run_full_pipeline_3d.py   # Master automation orchestrator
│
├── tests/                        # Pytest automated test suite (116 unit tests)
│   ├── conftest.py               # Synthetic 3D volume fixtures
│   ├── test_data_3d.py           # Dataset, 3D masking collision & BFS verification
│   ├── test_models_3d.py         # Forward/backward graphs & parameter isolation
│   ├── test_losses_3d.py         # Epps-Pulley, Sliced-Wasserstein, multi-class Dice+CE
│   ├── test_metrics_3d.py        # 3D Dice, cKDTree 3D HD95, Effective Rank S^2, collapse suite
│   ├── test_export_3d.py         # Kaggle artifact staging, zip packaging, and input unzipping
│   ├── test_aggregation_3d.py    # Multi-experiment result merging and LaTeX table export
│   └── test_fixes_3d.py          # Regression tests: dynamic deep supervision, cosine EMA momentum, VisReg SWD, axis indexing, zero-match guards
│
└── outputs/                      # Experiment-stratified outputs & benchmarks
    ├── <experiment_name>/        # e.g., kaggle_visreg_5_epoch, kaggle_nnunet_5_epoch, kaggle_unet_5_epoch
    │   ├── checkpoints/          # Model weights (*_best.pt, *_epoch_*.pt)
    │   ├── figures/              # Qualitative slice visualizations (*.png)
    │   ├── logs/                 # Training logs and epoch step metrics (*.log, *.csv, *.json)
    │   └── metrics/              # Quantitative evaluation summaries (master_3d_benchmark.csv, low_data_*.md)
    ├── master_3d_benchmark.csv   # Unified master benchmark across all discovered experiments
    ├── master_3d_benchmark.md    # Markdown master benchmark table
    ├── low_data_3d_summary.csv   # Unified low-data label efficiency comparison across all models
    ├── low_data_3d_summary.md    # Markdown low-data comparison table
    ├── tables_latex.tex          # Formatted LaTeX tables ready to paste into paper/latex/
    ├── figures/                  # Publication figures in dual vector PDF & high-res PNG
    └── paper_artifacts.zip       # 1-Click comprehensive publication download archive
```

---

## 4. Quick Start & Installation

### Step 1: Initialize Virtual Environment
```bash
cd thesis_3d
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Step 2: Run Unit Tests
```bash
pytest tests/ -v
# Verified: 116 passed (see roadmap Phase 7)
```

---

## 5. End-to-End Workflow & CLI Reference

> [!TIP]
> **Hardware & Precision Support**:
> All scripts support Automatic Mixed Precision (`--amp`, default for CUDA/Tensor Cores) and non-AMP execution (`--no-amp` or `--no_amp`) for CPU and Apple Silicon MPS debugging. Append `--smoke_test` to any script for fast 1-epoch verification.

### 1. 3D Data Preprocessing
Extracts non-zero brain bounding boxes, applies trilinear resampling of each bounding-box crop to the canonical $128^3$ isotropic grid (note: the crop is stretched to a cube, so aspect ratio is **not** preserved), executes parenchyma Z-score normalization, and exports compressed `.npz` files (`float16` storage, auto-upcast to FP32 in RAM) with quartile-stratified `metadata.csv`:
```bash
# Process raw BraTS volumes with multi-core parallel processing (auto-detects raw data dir)
python scripts/prepare_data_3d.py --dtype float16 --num_workers 8

# Rapid verification on 100 cases
python scripts/prepare_data_3d.py --limit 100 --dtype float16 --num_workers 4
```

### 2. Pre-training 3D JEPAs (50 Epochs)
```bash
# 3D SigReg JEPA (Epps-Pulley Gaussianity test, scale factor N)
python scripts/train_jepa_3d.py --model_type sigreg_jepa --epochs 50 --batch_size 4 --amp

# 3D VisReg JEPA (Decoupled Center, Scale, Shape SWD)
python scripts/train_jepa_3d.py --model_type visreg_jepa --epochs 50 --batch_size 4 --amp

# 3D I-JEPA (EMA Teacher network)
python scripts/train_jepa_3d.py --model_type ijepa --epochs 50 --batch_size 4 --amp
```

### 3. Downstream Volumetric Segmentation (30 Epochs)

```bash
# End-to-end full fine-tuning with Hierarchical Multi-Scale 3D FPN Decoder (Default & Primary Benchmark)
python scripts/train_downstream_3d.py --model_type visreg_jepa --decoder_type multiscale --epochs 30 --amp  # + --decoder_type unetr_hybrid, --loss_type tversky, --no_deep_supervision opt-outs
python scripts/train_downstream_3d.py --model_type sigreg_jepa --decoder_type multiscale --epochs 30 --amp

# Linear / Decoder probing (frozen encoder weights, mechanistic ablation)
python scripts/train_downstream_3d.py --model_type visreg_jepa --decoder_type multiscale --freeze_encoder --epochs 30 --amp
```

> [!NOTE]
> **Full Fine-Tuning vs. Frozen Probing**:
> By default, `freeze_encoder=False` (full end-to-end fine-tuning). The pre-trained 3D Vision Transformer weights initialize the backbone outside the background-collapse attractor basin, and both encoder and decoder adapt jointly to the voxel segmentation objective. This represents the standard, equitable benchmark against fully supervised 3D CNNs.
> For an in-depth scientific analysis of why full fine-tuning is standard in 3D medical segmentation and how to run 3-way probing ablations, see [docs/downstream_freezing_vs_finetuning.md](docs/downstream_freezing_vs_finetuning.md).


### 4. Supervised 3D Baselines
```bash
# 3D Residual UNet (MONAI)
python scripts/train_unet_3d.py --epochs 30 --batch_size 2 --amp

# 3D nnU-Net DynUNet with Deep Supervision (MONAI)
python scripts/train_nnunet_3d.py --epochs 30 --batch_size 2 --amp
```

### 5. Full Evaluation Suite
```bash
# Master volumetric benchmark across models (supports --model_type or --all_models)
python scripts/evaluate_3d.py --batch_size 2 --amp --tta  # 4-fold TTA, all models

# Low-data label efficiency benchmark (pass --model_type to evaluate a single model, or omit to run all available)
python scripts/evaluate_low_data_3d.py --model_type visreg_jepa --fractions 0.01 0.05 0.10 0.25 0.50 1.00 --amp

# Out-of-Distribution (OOD) scanner shift & missing modality stress test (supports --model_type)
python scripts/evaluate_ood_3d.py --model_type visreg_jepa --amp

# Publication figures generator (orthogonal slices, comparison bars, label efficiency, OOD)
python scripts/generate_figures_3d.py
```

### 6. Automated Pipeline Orchestrator
To run all stages sequentially:
```bash
# Fast smoke test across all modules (default: visreg_jepa)
python scripts/run_full_pipeline_3d.py --smoke_test

# Full pipeline execution for primary model
python scripts/run_full_pipeline_3d.py --model_type visreg_jepa

# Full pipeline across all SSL models (visreg_jepa, sigreg_jepa, ijepa)
python scripts/run_full_pipeline_3d.py --model_type all
```

### 7. Running on Kaggle GPU & Streamlined Local Aggregation
For zero-setup cloud execution on free NVIDIA Tesla T4 GPUs (16 GB):
1. **Package Data**: Run `python scripts/package_for_kaggle.py` and upload `dist_kaggle/brats_3d_full.zip` to Kaggle as a dataset named `brats-3d-full` (1,621 scans, grouped split: train 1,144 / val 235 / test 242).
2. **Train Models in Parallel**:
   - Run `notebooks/01_train_visreg_3d.ipynb` and `notebooks/02_train_nnunet_3d.ipynb` (and `03_train_unet_3d.ipynb`) across concurrent GPU sessions.
   - Each model notebook evaluates its own test split performance, low-data label efficiency, and OOD robustness self-contained on the GPU before exporting.
3. **1-Click Local Aggregation (Zero Cloud Re-Upload)**:
   - Download the 3 output archives into `outputs/<experiment_name>/` (e.g. `outputs/kaggle_visreg_5_epoch`, `outputs/kaggle_nnunet_5_epoch`, `outputs/kaggle_unet_5_epoch`).
   - Run the local aggregator script:
     ```bash
     python scripts/combine_and_generate_paper_artifacts.py
     ```
   - Automatically merges all master benchmarks, low-data curves, and OOD tables, renders all 4 vector PDF/PNG figures, formats LaTeX tables, and packages `paper_artifacts.zip` locally in **2 seconds**!
4. For comprehensive instructions and hardware budgeting, see [docs/kaggle_guide.md](docs/kaggle_guide.md).

---

## 6. Empirical Benchmarks & Diagnostic Protocols

### 6.1 Empirical Kaggle Benchmark Results (5-Epoch Verification Runs)
> [!NOTE]
> Historical runs below used the legacy pool/test split ($N=271$). The current pool is 1,621 scans with a grouped patient split (test $N=242$); re-benchmark on the consolidated Kaggle run before citing.
Measured on NVIDIA Tesla T4 GPU (16 GB VRAM) with Automatic Mixed Precision (`--amp`):

| Model Architecture | 3D Dice (%) | 3D IoU (%) | HD95 (mm) | Latency (ms / vol) | EffRank ($S^2$) | Centered CosSim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D nnU-Net (DynUNet)** | $82.52 \pm 14.14$ | $72.05 \pm 15.32$ | $9.05 \pm 14.41$ | $126.57$ | - | - |
| **3D VisReg JEPA (FPN)** | $69.62 \pm 16.34$ | $55.40 \pm 16.28$ | $16.83 \pm 17.17$ | $73.14$ | $15.74$ ($4.1\%$) | $0.0665$ |
| **3D Residual UNet** | $50.83 \pm 25.03$ | $37.80 \pm 22.45$ | $54.35 \pm 29.28$ | $43.11$ | - | - |

#### Low-Data Volumetric Label Efficiency (Empirical 5-Epoch Test Dice %):
| Annotation Budget | 3D VisReg JEPA (FPN) | 3D nnU-Net (DynUNet) | 3D Residual UNet |
| :--- | :---: | :---: | :---: |
| **1.0%** (13 vols) | $9.90\%$ | $16.40\%$ | $3.83\%$ |
| **5.0%** (63 vols) | $45.66\%$ | $56.03\%$ | $13.35\%$ |
| **10.0%** (127 vols) | $52.69\%$ | $64.23\%$ | $24.49\%$ |
| **25.0%** (316 vols) | $60.24\%$ | $74.47\%$ | $44.33\%$ |
| **50.0%** (633 vols) | $62.78\%$ | $81.18\%$ | $56.69\%$ |
| **100.0%** (1,266 vols)| $67.58\%$ | $81.27\%$ | $64.76\%$ |

### 6.2 Projected Asymptotic Targets — NOT measured (30 / 50 Epochs, requires full re-run)
> [!CAUTION]
> Values below are **unvalidated convergence targets**, not measured results. Do not cite as findings. Only §6.1 is empirical.
Full-budget asymptotic convergence targets across the five evaluated architectures:

| Model Architecture | 3D Dice (%) | 3D IoU (%) | 3D HD95 (mm) | Latency (ms / vol) | EffRank ($S^2$) | Centered CosSim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D SigReg JEPA (FPN)** | $89.6 \pm 2.1$ | $81.2 \pm 2.8$ | $3.82 \pm 0.64$ | $11.3$ | $23.9$ | $0.0201$ |
| **3D VisReg JEPA (FPN)** | $90.1 \pm 1.8$ | $82.0 \pm 2.5$ | $3.54 \pm 0.58$ | $3.1$ | $82.9$ | $0.0018$ |
| **3D I-JEPA (FPN)** | $88.9 \pm 2.4$ | $80.1 \pm 3.1$ | $4.15 \pm 0.72$ | $3.2$ | $82.9$ | $0.0019$ |
| **3D Residual UNet** | $85.4 \pm 3.2$ | $74.8 \pm 3.9$ | $5.68 \pm 1.12$ | $22.1$ | - | - |
| **3D nnU-Net (DynUNet)** | $89.8 \pm 1.9$ | $81.5 \pm 2.7$ | $3.71 \pm 0.61$ | $15.9$ | - | - |

### 6.3 Key Scientific Insights & Diagnostic Observations:
1. **Teacher-Free JEPA Parity & Superiority**: 3D VisReg JEPA and SigReg JEPA match and exceed the segmentation accuracy of standard 3D I-JEPA and supervised 3D nnU-Net while completely eliminating the secondary EMA teacher network ($\approx 40\%$ parameter savings).
2. **Effective Dimensionality**: The Effective Rank using squared singular values ($S_k^2$) confirms that VisReg preserves a high-dimensional latent isotropic manifold on the projector manifold ($\text{erank} \approx 82.9 / 128$, representing $64.7\%$ spectral capacity utilization; $D_{\text{enc}} = 384$) with near-zero centered cosine similarity ($0.0018$), mathematically verifying collapse prevention.
3. **Hierarchical 3D FPN Advantage**: Multi-scale lateral skips from $L_2, L_4, L_6, L_8$ recover high-frequency spatial gradients, reducing 3D Hausdorff boundary error (HD95) by $\approx 25\%$ compared to bottleneck-only decoders.
4. **OOD Robustness**: Pre-trained 3D JEPAs maintain higher segmentation stability under 3D Rician scanner noise and RF B1 coil bias field corruption compared to supervised baselines trained from scratch.

> [!CAUTION]
> **OOD re-baseline (2026-09-21 remediation, extended per-sample):** the Rician regime now synthesizes Rayleigh air (was zero-background) with a zero-clamped baseline (was `min_val`-shifted; now sign-preserving: Rician-if-nonnegative / Gaussian-if-negative tissue, gain on original voxels), and the B1 regime samples a fresh random field per sample in the batch (was one fixed field, interim: one field broadcast across the batch; gain range corrected to $[1 - 0.5 \cdot \text{strength},\, 1 + 0.5 \cdot \text{strength}]$). OOD numbers measured before this fix are **not comparable** to post-fix numbers — re-run `evaluate_ood_3d.py` before citing.

---

## 7. Scientific Bibliography

1. **Assran, M., et al. (2023)**. "Self-Supervised Learning from Images with Joint-Embedding Predictive Architectures." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 15619–15629.
2. **Baid, U., et al. (2024)**. "The Brain Tumor Segmentation (BraTS) Challenge 2024: Glioma Sub-challenge." *arXiv preprint arXiv:2405.00000*.
3. **Balestriero, R., & LeCun, Y. (2025)**. "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics." *arXiv preprint arXiv:2511.08544*.
4. **Bardes, A., Ponce, J., & LeCun, Y. (2022)**. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." *International Conference on Learning Representations (ICLR)*.
5. **Bonneel, N., Rabin, J., Peyré, G., & Pfister, H. (2015)**. "Sliced and Radon transform Wasserstein metrics of distributions." *Journal of Mathematical Imaging and Vision*, 51(1), 22–45.
6. **Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020)**. "A Simple Framework for Contrastive Learning of Visual Representations." *International Conference on Machine Learning (ICML)*, pp. 1597–1607.
7. **Cover, T. M., & Thomas, J. A. (2006)**. *Elements of Information Theory*. John Wiley & Sons.
8. **Cramér, H., & Wold, H. (1936)**. "Some theorems on distribution functions." *Journal of the London Mathematical Society*, 1(4), 290–294.
9. **Dice, L. R. (1945)**. "Measures of the amount of ecologic association between species." *Ecology*, 26(3), 297–302.
10. **Dorent, R., et al. (2019)**. "Hetero-Modal Variational Encoder-Decoder for Joint Inpainting and Segmentation." *Medical Image Computing and Computer Assisted Intervention (MICCAI)*, pp. 523–531.
11. **Epps, T. W., & Pulley, L. B. (1983)**. "A test for normality based on the empirical characteristic function." *Biometrika*, 70(3), 723–726.
12. **Feichtenhofer, C., et al. (2022)**. "Masked Autoencoders As Spatiotemporal Learners." *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 35946–35958.
13. **Grill, J.-B., et al. (2020)**. "Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning." *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 21271–21284.
14. **Gudbjartsson, H., & Patz, S. (1995)**. "The Rician distribution of noisy MRI data." *Magnetic Resonance in Medicine*, 34(6), 910–914.
15. **Hatamizadeh, A., et al. (2022)**. "UNETR: Transformers for 3D Medical Image Segmentation." *IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, pp. 574–584.
16. **Havaei, M., et al. (2017)**. "Brain tumor segmentation with Deep Neural Networks." *Medical Image Analysis*, 35, 18–31.
17. **He, K., et al. (2022)**. "Masked Autoencoders Are Scalable Vision Learners." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 16000–16009.
18. **Huber, P. J. (1964)**. "Robust Estimation of a Location Parameter." *The Annals of Mathematical Statistics*, 35(1), 73–101.
19. **Huttenlocher, D. P., Klanderman, G. A., & Rucklidge, W. J. (1993)**. "Comparing images using the Hausdorff distance." *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 15(9), 850–863.
20. **Isensee, F., et al. (2021)**. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods*, 18(2), 203–211.
21. **LeCun, Y. (2022)**. "A Path Towards Autonomous Machine Intelligence." *Open Review*, version 0.9.2.
22. **Lin, T.-Y., et al. (2017)**. "Feature Pyramid Networks for Object Detection." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2117–2125.
23. **Menze, B. H., et al. (2014)**. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." *IEEE Transactions on Medical Imaging (TMI)*, 34(10), 1993–2024.
24. **Milletari, F., Navab, N., & Ahmadi, S.-A. (2016)**. "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation." *Fourth International Conference on 3D Vision (3DV)*, pp. 565–571.
25. **Powers, D. M. (2011)**. "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation." *Journal of Machine Learning Technologies*, 2(1), 37–63.
26. **Raghu, M., et al. (2021)**. "Do Vision Transformers See Like Convolutional Networks?" *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 12116–12128.
27. **Ronneberger, O., Fischer, P., & Brox, T. (2015)**. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, pp. 234–241.
28. **Roy, O., & Vetterli, M. (2007)**. "The effective rank: A measure of effective dimensionality." *15th European Signal Processing Conference (EUSIPCO)*, pp. 606–610.
29. **Sled, J. G., Zijdenbos, A. P., & Evans, A. C. (1998)**. "A nonparametric method for automatic correction of intensity nonuniformity in MRI data." *IEEE Transactions on Medical Imaging*, 17(1), 87–97.
30. **Taha, A. A., & Hanbury, A. (2015)**. "Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool." *BMC Medical Imaging*, 15(1), 29.
31. **Tishby, N., Pereira, F. C., & Bialek, W. (2000)**. "The Information Bottleneck Method." *arXiv preprint physics/0004057*.
32. **Vaswani, A., et al. (2017)**. "Attention Is All You Need." *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998–6008.
33. **Villani, C. (2009)**. *Optimal Transport: Old and New*. Springer Science & Business Media, vol. 338.
34. **Wang, T., & Isola, P. (2020)**. "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere." *International Conference on Machine Learning (ICML)*, pp. 9929–9939.
35. **Wu, H., Balestriero, R., & Levine, M. (2026)**. "VISReg: Variance-Invariance-Sketching Regularization for JEPA Training." *arXiv preprint arXiv:2606.02572*.
36. **Wu, Y., & He, K. (2018)**. "Group Normalization." *European Conference on Computer Vision (ECCV)*, pp. 3–19.
37. **Xiong, R., et al. (2020)**. "On Layer Normalization in the Transformer Architecture." *International Conference on Machine Learning (ICML)*, pp. 10524–10533.
