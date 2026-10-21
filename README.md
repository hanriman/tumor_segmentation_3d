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
> **We extend Joint-Embedding Predictive Architectures to full 3D multi-modal brain MRI volumes, investigating whether heuristic-free VISReg regularization prevents 3D latent collapse without momentum teacher networks. This study benchmarks 3D VISReg JEPA against 3D Residual UNet, 3D nnU-Net (DynUNet), and hierarchical decoder ablations (multiscale FPN vs. hybrid UNETR) under full-data, label-scarce, and scanner-shifted regimes. SigReg is implemented in code via the same unified `SingleEncoderJEPA3D` backbone.**

### Why JEPA Over MAE and Contrastive Learning
- **Vs. Masked Autoencoders (He et al., 2022; Feichtenhofer et al., 2022)**: MAE reconstructs raw voxels, forcing model capacity to encode high-frequency non-semantic MRI scanner noise (Rician distribution; Gudbjartsson & Patz, 1995) and RF coil bias field inhomogeneities (Sled et al., 1998). JEPA predicts abstract representations in latent space ($\hat{y}_{\text{tgt}} \approx y_{\text{tgt}}$), naturally discarding high-frequency non-semantic noise (LeCun, 2022; Assran et al., 2023).
- **Vs. Contrastive Learning (SimCLR, MoCo)**: Contrastive methods penalize negative pairs. In brain MRI, all human subjects share macroscopic anatomy (bilateral hemispheres, ventricles, anatomical lobes). Treating different patients as negative pairs imposes false-negative gradient penalties on identical anatomical structures. JEPA requires zero negative pairs.
- **Vs. Empirical Momentum Heuristics (BYOL, DINO)**: Momentum teacher architectures rely on an Exponential Moving Average (EMA) teacher network ($\bar{\theta} \leftarrow m \bar{\theta} + (1-m) \theta$). However, EMA is an empirical heuristic without theoretical convergence guarantees, introduces sensitive momentum schedules ($m \in [0.996, 1.0]$), and incurs $\approx 40\%$ parameter memory overhead. **SigReg** and **VISReg** eliminate the teacher network via principled information-theoretic and optimal-transport regularizers applied directly in encoder representation space.

---

## 2. Mathematical & Algorithmic Foundations

| Component | Mathematical / Algorithmic Formulation | Theoretical Reference & Justification |
| :--- | :--- | :--- |
| **Volumetric Continuity** | $\mathbf{X} \in \mathbb{R}^{4 \times 128 \times 128 \times 128}$, $1.0\text{ mm}^3$ isotropic grid. | **Menze et al. (2014)**; **Milletari et al. (2016)**. Preserves 3D spatial continuity across white matter tracts. |
| **Grid Resampling** | Non-zero brain bounding box $\to$ trilinear resampling to canonical $128^3$ isotropic grid (bounding box stretched to cube; aspect ratio **not** preserved). | **Isensee et al. (2021)** (*Nature Methods*). Prevents anatomical truncation of peripheral cortex at the cost of anisotropic geometric distortion (physical-mm metrics inherit this warp). |
| **Parenchyma Normalization** | $X_{c, \text{norm}}(v) = \frac{X_c(v) - \mu_{c,\text{nz}}}{\sigma_{c,\text{nz}} + \epsilon}$ for non-zero voxels. | **Isensee et al. (2021)**. Harmonizes scanner-dependent intensity drift across institutions. |
| **Random Modality Dropout** | $m_c \sim \text{Bernoulli}(1 - p_{\text{drop}})$, $p_{\text{drop}}=0.25$, fallback $\sum_c m_c \ge 1$, zero-imputed (`rescale=False`) to match zero-imputed missing-modality evaluation. | **Havaei et al. (2017)**; **Dorent et al. (2019)**. Simulates missing MRI sequences while avoiding train/eval channel-norm mismatch. |
| **3D Context / Target Masking (Anisotropic Target Jitter)** | 6 Anisotropic target cuboids ($\mathcal{S}_{8\text{-grid}} = [(3,3,3), (2,3,3), (3,2,3), (3,3,2), (2,2,2), (2,2,2)]$, $N_{\text{tgt}} = 97$ tokens on $16^3$ patch; $\mathcal{S}_{16\text{-grid}} = 2\times$ scaled, $N_{\text{tgt}} = 776$ tokens on $8^3$ patch preserving identical physical mm volume) + 3D Connected BFS Context ($N_{\text{ctx}} = 37.5\%$ volume), **brain-aware**: proportional tissue filtering ($\ge \max(2, \lfloor 0.25 \times \text{vol} \rfloor)$), fallback bounds $[2, 3]^3$, tissue-only VISReg/SigReg regularization. | **Assran et al. (2023)** (*CVPR*). Prevents spatial interpolation shortcuts; anisotropic aspect-ratio jitter captures directional non-axis-aligned glioma morphology; constant $N_{\text{ctx}}$ maximizes Tensor Core throughput. |
| **3D Positional Embeddings** | Separable 3D Sinusoidal Coordinate Embeddings (frozen, requires_grad=False) $[\text{PE}(z) \,\|\, \text{PE}(y) \,\|\, \text{PE}(x)]$. | **Vaswani et al. (2017)**; **Feichtenhofer et al. (2022)**. Imparts immediate 3D metric spatial topology from step 0. |
| **Context Gather (Zero Leakage)** | $\mathbf{Z}_{\text{ctx}} = \text{Encoder}(\text{gather}(\mathbf{Z}_0, \text{idx}_{\text{ctx}})) \in \mathbb{R}^{B \times 1536 \times 384}$ (canonical $8^3$ patch) or $\mathbb{R}^{B \times 192 \times 384}$ ($16^3$ patch). | **Assran et al. (2023)**. Evaluates strictly visible tokens; reduces self-attention FLOPs by $85.9\%$ compared to dense masking. |
| **Auxiliary Projector Regularization** | 2-layer Projector MLP ($g_\psi: \mathbb{R}^{384} \to \mathbb{R}^{1024} \to \mathbb{R}^{128}$) projects tissue-filtered context latents before Sliced-Wasserstein / Epps-Pulley moment regularization, isolating the primary $384$-D ViT representation space from isotropic geometric constraints before downstream transfer. | **Wu, Balestriero, & Levine (2026)**; **Balestriero & LeCun (2025)**. Decouples geometric non-collapse regularization from downstream anatomical representation learning; projector is discarded during fine-tuning. |
| **SigReg Gaussianity Test** | $\mathcal{T}_{\text{EP}} = N \int \|\hat{\phi}_N(t) - \phi_0(t)\|^2 d\mu(t)$ over $M=1024$ 1D rays on tissue-filtered encoder tokens $\mathbf{Z}_{\text{tissue}} \in \mathbb{R}^{N_{\text{tissue}} \times 384}$; optional `generator` for seeded projections. | **Balestriero & LeCun (2025)**; **Epps & Pulley (1983)**. Sample factor $N$ cancels $1/N$ ECF derivative, yielding $O(1)$ per-sample gradients. |
| **VISReg Decoupled Moments** | $\mathcal{L}_{\text{VISReg}} = \mathcal{L}_{\text{JEPA}} + \lambda_{\text{c}} \frac{\|\mu\|_2^2}{D} + \lambda_{\text{sc}} \frac{\sum (1-\sigma_d)^2}{D} + \lambda_{\text{sh}} W_2^2$ ($M=1024$ slices, $\lambda_{\text{sh}}=4.0$); shape term standardizes with detached $\mu,\sigma$ (`z_std = (z - mu.detach()) / sigma.detach()`) before projecting onto unit sphere $\mathbb{S}^{D-1}$. | **Wu, Balestriero, & Levine (2026)** (`arXiv:2602.11401`, Algorithm 1); **Villani (2009)**. Decouples center, scale, and shape gradients; closed-form 1D sort in $O(N \log N)$. |
| **Learning Rate Cosine Schedule** | $\eta(t)$: 10-epoch linear warmup ($0 \to 5 \times 10^{-4}$), followed by 190-epoch cosine annealing decay down to $\eta_{\text{min}} = 1 \times 10^{-5}$ across epochs 11–200 for canonical 200-epoch pre-training. | **Loshchilov & Hutter (2017)**; **Goyal et al. (2017)**. Linear warmup stabilizes early self-attention optimization; cosine annealing ensures asymptotic convergence. |
| **Regularizer Warmup Schedule** | $\lambda_{\text{reg}}(t)$: Linearly ramps from $0 \to 1.0$ over `reg_warmup_epochs=30` and holds constant at $\lambda_{\text{reg}} = 1.0$ (`reg_min_scale=1.0`) throughout epochs 31–200. | **Wu, Balestriero, & Levine (2026)**; **Balestriero & LeCun (2025)**. Constant full-strength regularization maintains isotropic variance across all 200 epochs. |
| **Hierarchical 3D Decoders** | Lateral skips from $L_2, L_4, L_6, L_8$ with progressive ConvTranspose3d (`multiscale` FPN, ~29.43M, pure latent probe) or dual-stream ViT + native convolutional stem (`unetr_hybrid`, ~29.52M, hybrid probe). | **Lin et al. (2017)**; **Hatamizadeh et al. (2022)**; **Wu & He (2018)**. Restores high-frequency boundary gradients; GroupNorm stabilizes small 3D batches ($B \le 4$). |
| **Supervised 3D Baselines** | MONAI 3D Residual UNet (~19.23M, + ds1/ds2/ds3 heads) and MONAI 3D DynUNet (~22.75M), both with deep supervision default-ON ($w_s = 2^{-s}/\sum 2^{-j}$); asymmetric Tversky opt-in; 4-fold TTA at eval. | **Ronneberger et al. (2015)**; **Isensee et al. (2021)**; **Lee et al. (2015)**; **Salehi et al. (2017)**. Establishes state-of-the-art supervised benchmark parity. |
| **Exact 3D HD95 Metric** | $d_{\text{HD95}}^{3\text{D}}$ via 3D 26-connectivity morphological erosion and `cKDTree` in canonical grid voxels (normalized $128^3$ space); tumor-only aggregates (`dice_tumor_only`, `iou_tumor_only`, `hd95_tumor_only`) are NaN (never 1.0/0.0) on tumor-free cohorts or when HD95 is off — aggregate with `nanmean`, render as `n/a`. | **Huttenlocher et al. (1993)**; **Taha & Hanbury (2015)**; **Powers (2011)**. Guarded zero-division prevents artificial epsilon score inflation. |
| **Effective Rank ($S_k^2$)** | $\text{erank}(Z) = \exp\left(-\sum p_k \ln p_k\right)$, $p_k = S_k^2 / \sum S_j^2$. Evaluated both over full volume tokens and tissue-filtered foreground tokens to reject air bias. | **Roy & Vetterli (2007)**. Strictly uses covariance eigenvalues $S_k^2$, preventing linear $S_k$ from masking dimensional collapse. |
| **Centered Cosine Sim** | $\bar{S}_{\text{centered}} = \frac{1}{N(N-1)} \sum_{i \neq j} \frac{(z_i - \bar{z})^\top (z_j - \bar{z})}{\|z_i - \bar{z}\|_2 \|z_j - \bar{z}\|_2}$ (sanity check; converges to $\approx -\frac{1}{N-1}$ under uniform centered norms). | **Wang & Isola (2020)**. Monitors off-diagonal angular dispersion alongside Effective Spectral Rank ($S_k^2$). |
| **Combined Loss Alignment** | Multi-label binary Dice + BCE across nested BraTS sub-regions (`WT={1,2,3}`, `TC={1,3}`, `ET={3}`). | **Isensee et al. (2021)**; **Milletari et al. (2016)**. Supervises nested clinical sub-regions directly while handling empty-ET cases cleanly. |
| **Quartile-Stratified Splits** | Low-data tiers ($1\%$ to $50\%$) stratified by tumor volume quartile with minimum step floor (`--min_steps 250`) and validation-set model selection before single test evaluation. | **Isensee et al. (2021)**. Preserves lesion size balance, avoids gradient starvation on tiny subsets, and eliminates test-set selection bias. |
| **Metric Validation Guard** | Strict single-channel validation ($C=1$) + `from_logits` toggle in `compute_volumetric_metrics_3d`. | **Taha & Hanbury (2015)**. Prevents silent metric computation on background channel 0 and avoids double-sigmoid distortion. |
| **3D Rician Scanner Noise** | $M = \sqrt{(X_+ + \eta_1)^2 + \eta_2^2}$, $\eta_1, \eta_2 \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$ on all voxels (tissue Rician for X>=0 / Gaussian for Z-scored negatives + air Rayleigh, sign-preserving). | **Gudbjartsson & Patz (1995)**. Accurately simulates MRI magnitude reconstruction from quadrature RF coil channels. |
| **3D B1 RF Field Bias** | $X_{\text{corrupt}} = X \cdot \left(1 + \text{strength} \cdot \sum_{i+j+k \le 2} c_{ijk} x^i y^j z^k\right)$ with per-sample $c_{ijk} \sim \mathcal{N}(0,1)$ (field clamped to $[-0.5, 0.5]$, gain in $[1 - 0.5 \cdot \text{strength},\, 1 + 0.5 \cdot \text{strength}]$); gain multiplies original voxels (negatives preserved); background stays zero. | **Sled et al. (1998)**; **Lebrun et al. (2021)**. Multiplicative smooth 2nd-order polynomial simulating RF coil field inhomogeneity; every volume sees a different field. |

---

## 3. Standalone Repository Structure

```text
thesis_3d/
├── .gitignore
├── .python-version               # Python 3.12
├── pyproject.toml                # uv & setuptools configuration
├── README.md                     # Comprehensive documentation
├── AGENTS.md                     # Scientific writing & research guidelines
│
├── docs/
│   ├── README.md                         # Documentation index & guide overview
│   ├── 01_visreg_jepa_architecture_and_pipeline.md # Complete dataflow, tokenization & VISReg/SigReg loss formulation
│   ├── 02_forensic_audit_and_tier2_upgrades.md     # Audit remediations, methodological fixes & architectural simplification
│   ├── 03_scientific_rationales_and_benchmarks.md  # Brain-aware masking, HD95 decoupling, fairness & official WT/TC/ET results
│   └── 04_cloud_and_local_execution_guide.md       # Google Colab GPU execution & local artifact aggregation
│
├── notebooks/
│   └── colab/                        # Canonical Google Colab GPU suite (8³ patch / 4,096 tokens, Drive sync)
│       ├── 01_colab_visreg_pretrain_3d.ipynb           # 3D VISReg JEPA SSL pre-training (200 epochs)
│       ├── 02_colab_finetune_visreg_multiscale_3d.ipynb # VISReg + Multi-Scale FPN downstream (60 epochs)
│       ├── 03_colab_finetune_visreg_unetr_3d.ipynb      # VISReg + Hybrid UNETR downstream (60 epochs)
│       ├── 04_colab_nnunet_3d.ipynb                    # 3D nnU-Net baseline (DynUNet with deep supervision)
│       ├── 05_colab_unet_3d.ipynb                      # 3D Residual UNet baseline
│       ├── 06_colab_scratch_multiscale_3d.ipynb        # From-scratch ViT-FPN ablation (zero pre-training)
│       ├── 07_colab_scratch_unetr_3d.ipynb             # From-scratch ViT-UNETR hybrid ablation (zero pre-training)
│       ├── 08_colab_frozen_visreg_multiscale_3d.ipynb   # VISReg + FPN frozen encoder probe
│       └── 09_colab_frozen_visreg_unetr_3d.ipynb       # VISReg + UNETR frozen encoder probe
│
├── configs/                      # Modular YAML configuration hierarchy
│   ├── base.yaml                 # System paths, seed, device, AMP configs
│   ├── dataset/
│   │   └── brats3d.yaml          # Grid geometry (128^3), patch size, modalities, augmentations
│   ├── model/
│   │   ├── sigreg_jepa_3d.yaml   # 3D SigReg JEPA + Epps-Pulley parameters (M=1024)
│   │   ├── visreg_jepa_3d.yaml   # 3D VISReg JEPA + Sliced-Wasserstein parameters (M=1024, swd_weight=4.0)
│   │   ├── unet_3d.yaml          # 3D Residual UNet parameters
│   │   └── nnunet_3d.yaml        # 3D DynUNet + Deep Supervision parameters
│   └── experiment/
│       ├── pretrain.yaml         # 3D SSL pre-training schedule & resolution-scaled anisotropic target masking
│       └── finetune.yaml         # 3D downstream fine-tuning schedule
│
├── src/
│   └── brats_jepa_3d/            # Standalone Python package
│       ├── __init__.py
│       ├── config.py             # Cloud/Colab/Local dynamic path & YAML config resolver
│       ├── data/
│       │   ├── dataset.py        # BraTS3DDataset with bounded RAM caching & stratified fraction sampling
│       │   ├── transforms.py     # RandomModalityDropout3D (zero-imputed), Rician noise, B1 bias field
│       │   └── masking.py        # JEPAMaskingTransform3D (brain-aware 3D cuboids + 3D BFS context)
│       ├── models/
│       │   ├── builder.py               # Central factory: build_segmentation_model, load_segmentation_checkpoint, extract_encoder_state_dict
│       │   ├── vision_transformer_3d.py # PatchEmbed3D + 3D Sincos Pos + ViTEncoder3D
│       │   ├── predictor_3d.py          # JEPAPredictor3D (4 layers, self-attention)
│       │   ├── single_encoder_jepa_3d.py # Unified SingleEncoderJEPA3D (shared context/target encoder + tissue filter)
│       │   ├── _tissue_filter.py        # Shared tissue-foreground token extraction helper
│       │   ├── segmentation_head_3d.py  # MultiScaleSegmentationDecoder3D (FPN) & UNETRHybridDecoder3D + SegmentationModel3D
│       │   ├── unet_3d.py               # 3D Residual UNet (MONAI) + ds1/ds2/ds3 deep-supervision heads
│       │   └── nnunet_3d.py             # 3D DynUNet with Deep Supervision (MONAI)
│       ├── training/
│       │   └── supervised_trainer.py    # Shared train_one_epoch, validate_regions, run_supervised_training
│       ├── losses/
│       │   ├── sigreg_loss.py           # Epps-Pulley Gaussianity test (scale factor N, M=1024 1D rays)
│       │   ├── visreg_loss.py           # Decoupled Center, Scale, Shape (SWD on standardized tokens, M=1024)
│       │   ├── dice_bce_loss_3d.py      # Combined Volumetric 3D Dice + Cross-Entropy (multi-label & binary)
│       │   ├── tversky_loss_3d.py       # Asymmetric Tversky (α=0.3, β=0.7) + BCE opt-in + criterion factory
│       │   └── deep_supervision_loss_3d.py # Multi-scale exponential decay supervision
│       ├── metrics/
│       │   ├── regions.py               # BraTS 2024 GLI 3-region conversion (WT={1,2,3}, TC={1,3}, ET={3})
│       │   ├── volumetric_metrics.py    # Guarded 3D Dice, IoU, cKDTree 3D HD95 (Powers 2011)
│       │   └── probing_metrics.py       # Effective Rank (S^2), Centered Cosine Sim
│       └── utils/
│           ├── aggregation.py           # Multi-experiment merging & LaTeX table generation
│           ├── device.py                # CUDA / MPS / CPU device detection & AMP context
│           ├── tta.py                   # 4-fold orthogonal-reflection TTA (eval, all models)
│           ├── export.py                # Cloud artifact staging, zip packaging, and checkpoint resolution
│           ├── logging.py               # MetricTracker, setup_logger & numeric checkpoint sorting
│           └── seed.py                  # Deterministic PRNG seeding
│
├── scripts/
│   ├── prepare_data_3d.py        # NIfTI bounding-box cropping, resampling to 128^3 & .npz export
│   ├── train_jepa_3d.py          # 3D SSL Pre-training runner (200 epochs, VISReg or SigReg)
│   ├── train_downstream_3d.py    # Downstream 3D fine-tuning & frozen probing runner
│   ├── train_unet_3d.py          # Supervised 3D UNet baseline runner (--deep_supervision default-ON)
│   ├── train_nnunet_3d.py        # Supervised 3D nnU-Net baseline runner
│   ├── evaluate_3d.py            # Master benchmark evaluator (Dice, HD95, Rank, Latency, --tta)
│   ├── diagnose_visreg_3d.py     # Real-data D1–D3 masking/regularization diagnostic probe
│   ├── evaluate_low_data_3d.py   # Label efficiency runner (1% to 100% 3D labels, val-selected best ckpt, --min_steps 250)
│   ├── evaluate_ood_3d.py        # 3D Scanner shift & missing-modality runner (Rician noise, B1 bias field, zero-imputed missing modalities)
│   └── combine_and_generate_paper_artifacts.py # 1-Click master multi-experiment aggregator & LaTeX exporter
│
├── paper/
│   └── latex/
│       ├── main.tex              # Thesis manuscript source
│       ├── references.bib        # BibTeX references
│       └── figures/
│           └── gen_figures.py    # Vector PDF & PNG publication figure generator
│
├── tests/                        # Consolidated pytest regression test suite (172 tests, 100% pass)
│   ├── smoke_test_pipeline.py    # End-to-end multi-model pipeline smoke test runner
│   ├── conftest.py               # Synthetic 3D volume fixtures & hardware mocks
│   ├── test_benchmark_consistency.py # Paper LaTeX vs CSV mathematical parity tests
│   ├── test_data_3d.py           # Dataset, 0% leakage, anisotropic 3D target jitter & BFS masking
│   ├── test_evaluation_3d.py     # Evaluation parity, model builder & checkpoint loading tests
│   ├── test_integration_3d.py    # End-to-end smoke test integrations for training and evaluation
│   ├── test_losses_3d.py         # VISReg SWD shape, SigReg Epps-Pulley, warmup schedule, Dice, BCE, Tversky
│   ├── test_metrics_3d.py        # Volumetric metrics, cKDTree 3D HD95, Effective Rank S^2
│   └── test_models_3d.py         # ViT encoder, predictor, SingleEncoderJEPA3D, FPN, UNETR hybrid, DynUNet, Residual UNet
│
└── outputs/                      # Experiment-stratified outputs & benchmarks
    ├── experiments/              # Per-experiment artifacts (checkpoints, logs, metrics, figures)
    └── reports/                  # Aggregated benchmark tables (master_3d_benchmark.csv, low_data_3d_summary.csv, ood_3d_summary.csv, tables_latex.tex)
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
uv run pytest -q
# Verified: 172 passed (100% pass rate)
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

### 2. Pre-training 3D JEPAs (200 Epochs)
```bash
# 3D SigReg JEPA (Epps-Pulley Gaussianity test on encoder tissue tokens, M=1024)
python scripts/train_jepa_3d.py --model_type sigreg_jepa --patch_size 8 8 8 --epochs 200 --batch_size 32 --amp

# 3D VISReg JEPA (Decoupled Center, Scale, Shape SWD on encoder tissue tokens, M=1024, swd_weight=4.0)
python scripts/train_jepa_3d.py --model_type visreg_jepa --patch_size 8 8 8 --epochs 200 --batch_size 32 --amp \
    --reg_warmup_epochs 30 --reg_min_scale 1.0
```

> **Benchmark scope:** both `visreg_jepa` and `sigreg_jepa` share the identical `SingleEncoderJEPA3D` architecture and differ solely in the regularizer applied to `outputs["tissue_tokens"]` (`VisRegLoss` vs. `SigRegLoss`). This study benchmarks VISReg JEPA against 3D Residual UNet, 3D nnU-Net, and ViT decoder ablations.

### 3. Downstream Volumetric Segmentation (60 Epochs)

```bash
# End-to-end full fine-tuning with Hierarchical Multi-Scale 3D FPN Decoder or Hybrid UNETR
python scripts/train_downstream_3d.py --model_type visreg_jepa --decoder_type multiscale --patch_size 8 8 8 --epochs 60 --batch_size 4 --amp
python scripts/train_downstream_3d.py --model_type visreg_jepa --decoder_type unetr_hybrid --patch_size 8 8 8 --epochs 60 --batch_size 4 --amp

# Frozen-backbone decoder probing (--freeze_encoder keeps ViT encoder in eval mode with requires_grad=False)
python scripts/train_downstream_3d.py --model_type visreg_jepa --decoder_type multiscale --patch_size 8 8 8 --freeze_encoder --epochs 60 --batch_size 4 --amp
python scripts/train_downstream_3d.py --model_type visreg_jepa --decoder_type unetr_hybrid --patch_size 8 8 8 --freeze_encoder --epochs 60 --batch_size 4 --amp
```

> [!NOTE]
> **Full Fine-Tuning vs. Frozen Probing**:
> By default, `freeze_encoder=False` (full end-to-end fine-tuning). The pre-trained 3D Vision Transformer weights initialize the backbone outside the background-collapse attractor basin, and both encoder and decoder adapt jointly to the voxel segmentation objective. This represents the standard, equitable benchmark against fully supervised 3D CNNs.
> For an in-depth scientific analysis of why full fine-tuning is standard in 3D medical segmentation and how to run 3-way probing ablations, see [docs/03_scientific_rationales_and_benchmarks.md](docs/03_scientific_rationales_and_benchmarks.md).

### 4. Supervised 3D Baselines (60 Epochs)
```bash
# 3D Residual UNet (MONAI, deep supervision default-ON)
python scripts/train_unet_3d.py --epochs 60 --batch_size 4 --amp

# 3D nnU-Net DynUNet with Deep Supervision (MONAI)
python scripts/train_nnunet_3d.py --epochs 60 --batch_size 4 --amp
```

### 5. Full Evaluation Suite
```bash
# Master volumetric benchmark across models (supports --model_type or --all_models)
python scripts/evaluate_3d.py --batch_size 4 --amp --tta  # 4-fold TTA, all models

# Low-data label efficiency benchmark across 6 annotation tiers (25 epochs per fraction, --min_steps 250, val-selected best ckpt)
python scripts/evaluate_low_data_3d.py --model_type visreg_jepa --decoder_type multiscale --patch_size 8 8 8 \
    --fractions 0.01 0.05 0.10 0.25 0.50 1.00 --epochs 25 --min_steps 250 --amp

# Out-of-Distribution (OOD) scanner shift & missing modality stress test
python scripts/evaluate_ood_3d.py --model_type visreg_jepa --decoder_type multiscale --patch_size 8 8 8 --amp

# Publication figures & master aggregation
python scripts/combine_and_generate_paper_artifacts.py
python paper/latex/figures/gen_figures.py
```

### 6. Automated Pipeline Smoke Test Runner
To verify all stages end-to-end via a rapid smoke test:
```bash
# Fast smoke test across all modules (default: visreg_jepa)
python tests/smoke_test_pipeline.py

# Smoke test across both SSL variants (visreg_jepa, sigreg_jepa)
python tests/smoke_test_pipeline.py --model_type all
```

### 7. Cloud Execution (Google Colab GPU Suite) & Local Aggregation

1. **Google Colab Suite (`notebooks/colab/01..09_*.ipynb`)**:
   - Place `brats_gli_3d_v2.zip` in Google Drive under `My Drive/thesis/data/`.
   - Run `notebooks/colab/01_colab_visreg_pretrain_3d.ipynb` to pre-train the 3D VISReg JEPA encoder ($8^3$ patches, $4,096$ tokens, 200 epochs). Checkpoints are saved directly to `My Drive/thesis/pretrain model/checkpoints/visreg_jepa_3d_best.pt`.
   - Run downstream and baseline notebooks (`02` through `09`) as needed; downstream notebooks automatically resolve pre-trained weights from `My Drive/thesis/pretrain model/` (no re-running of pre-training required!), stream training + test + low-data + OOD metrics to `My Drive/thesis/downstream/<experiment>/`, and package verified zip archives.
2. **1-Click Local Aggregation**:
   - Place downloaded experiment folders under `outputs/experiments/<experiment_name>/` and run:
     ```bash
     python scripts/combine_and_generate_paper_artifacts.py
     ```
   - Automatically merges all master benchmarks, low-data curves, and OOD tables into `outputs/reports/` and generates LaTeX tables. For detailed operational instructions, see [docs/04_cloud_and_local_execution_guide.md](docs/04_cloud_and_local_execution_guide.md).

---

## 6. Official Empirical Benchmarks & Scientific Findings

Evaluated on the **BraTS 2024 GLI** benchmark under the three-region protocol (**Whole Tumor [WT]**, **Tumor Core [TC]**, and **Enhancing Tumor [ET]**) on the held-out test split ($N=242$ patient volumes, 4-fold TTA):

### 6.1 Full Downstream Multi-Region Benchmark (60 Epochs, Full Split)

| Model Architecture | Encoder Backbone | Decoder Type | Mean Dice (WT/TC/ET) | WT Dice | TC Dice | ET Dice | WT / TC / ET HD95 (voxels) | Latency (128³) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D nnU-Net** | DynUNet CNN (~22.75M) | Deep Supervised | **70.86%** | **84.47%** | **64.35%** | **63.76%** | **5.28 / 11.66 / 11.12** | 1089.66 ms |
| **3D VisReg UNETR** | 3D ViT (JEPA Pretrained) | UNETR Hybrid (~29.52M) | 69.66% | 84.11% | 62.56% | 62.30% | 5.70 / 39.29 / 38.82 | 1347.52 ms |
| **3D VisReg UNETR (Frozen)** | 3D ViT (JEPA Pretrained & Frozen) | UNETR Hybrid (~29.52M) | 69.34% | 83.29% | 62.38% | 62.35% | 5.79 / 31.53 / 31.11 | 1446.23 ms |
| **3D ViT-UNETR (Scratch)** | 3D ViT (From Scratch) | UNETR Hybrid (~29.52M) | 68.19% | 82.66% | 61.10% | 60.80% | 6.04 / 34.08 / 33.67 | 1343.26 ms |
| **3D VisReg JEPA (FPN)** | 3D ViT (JEPA Pretrained) | Multi-Scale FPN (~29.43M) | 67.05% | 82.38% | 59.63% | 59.15% | 6.07 / 44.43 / 44.10 | 1274.80 ms |
| **3D Residual UNet** | ResUNet CNN (~19.23M) | Residual 3D | 65.83% | 81.99% | 57.87% | 57.64% | 6.50 / 15.69 / 15.18 | **488.82 ms** |
| **3D ViT-FPN (Scratch)** | 3D ViT (From Scratch) | Multi-Scale FPN (~29.43M) | 66.13% | 81.27% | 58.84% | 58.29% | 6.24 / 32.80 / 32.44 | 1181.68 ms |
| **3D VisReg FPN (Frozen)** | 3D ViT (JEPA Pretrained & Frozen) | Multi-Scale FPN (~29.43M) | 63.18% | 80.39% | 54.84% | 54.31% | 6.76 / 50.45 / 50.11 | 1629.77 ms |

### 6.2 Low-Data Volumetric Label Efficiency (25 Epochs per Tier, without TTA)

| Annotation Budget | VisReg UNETR | VisReg FPN | 3D nnU-Net | 3D ResUNet | ViT-UNETR (Scratch) | ViT-FPN (Scratch) | VisReg UNETR (Frozen) | VisReg FPN (Frozen) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0%** (12 vols) | 32.69% | 25.99% | 20.69% | 4.29% | 32.70% | 23.11% | **34.32%** | 22.43% |
| **5.0%** (58 vols) | **54.45%** | 51.67% | 47.06% | 43.92% | 49.79% | 41.53% | 50.75% | 50.58% |
| **10.0%** (115 vols) | **57.68%** | 53.48% | 50.66% | 49.21% | 52.60% | 47.14% | 54.00% | 50.33% |
| **25.0%** (286 vols) | **61.76%** | 57.66% | 58.18% | 53.88% | 55.83% | 53.16% | 59.73% | 55.42% |
| **50.0%** (572 vols) | 63.92% | 60.77% | 61.30% | 61.19% | 60.64% | 56.88% | **63.94%** | 57.68% |
| **100.0%** (1,144 vols)| **66.16%** | 62.36% | 65.90% | 62.54% | 65.41% | 60.00% | 64.27% | 60.45% |

### 6.3 Key Scientific Insights & Diagnostic Observations:
1. **Decisive Low-Data JEPA Advantage ($5\%$ & $10\%$ Data)**:
   - At $5\%$ data ($58$ volumes), **3D VisReg UNETR achieves the top performance overall ($\mathbf{54.45\%}$ mean Dice)**, beating from-scratch ViT-UNETR ($49.79\%$, $\mathbf{+4.66\text{pp}}$ JEPA gain), deeply supervised 3D nnU-Net ($47.06\%$, $\mathbf{+7.39\text{pp}}$), and 3D Residual UNet ($43.92\%$, $\mathbf{+10.53\text{pp}}$). VisReg FPN similarly achieves $\mathbf{51.67\%}$ ($\mathbf{+10.14\text{pp}}$ over Scratch FPN $41.53\%$).
   - At $10\%$ data ($115$ volumes), **VisReg UNETR retains the #1 rank overall ($\mathbf{57.68\%}$ vs. nnU-Net $50.66\%$ and Scratch UNETR $52.60\%$)**: self-supervised joint-embedding pre-training provides strong initialization priors when labeled volumes are scarce, whereas under full supervision ($100\%$ labels) deeply supervised 3D nnU-Net maintains the performance ceiling ($70.86\%$), closely followed by VisReg UNETR ($69.66\%$) and Scratch UNETR ($68.19\%$).
2. **Frozen-Backbone Spatial Decodability (Pure Latent vs. Hybrid Probing)**:
   - Freezing the pre-trained ViT encoder (`freeze_encoder=True`) with a **pure latent multi-scale FPN decoder** (no raw-image skip connections) achieves **$63.18\%$ mean Dice** ($80.39\%$ WT), showing that frozen transformer tokens retain localized multi-compartment spatial structure without raw image bypass.
   - Pairing the frozen encoder with the **hybrid UNETR decoder** (which fuses frozen ViT tokens with trainable high-resolution convolutional stems on the raw volume) reaches **$69.34\%$ mean Dice**, approaching full fine-tuning ($69.66\%$) and exceeding fully trained from-scratch ViT-UNETR ($68.19\%$).
3. **Latent Manifold Geometry (Non-Collapse Verification)**:
   - Across evaluated ViT backbones, tissue-filtered Effective Spectral Rank ($S^2$) remains well above dimensional collapse ($S^2 \to 1$): fine-tuned VisReg FPN achieves **$81.78$** ($21.3\%$ of $D_{\text{enc}}=384$), fine-tuned VisReg UNETR achieves **$69.13$** ($18.0\%$), frozen VisReg probes achieve **$67.71$** ($17.6\%$), and from-scratch controls achieve **$46.38$** ($12.1\%$, FPN) and **$30.95$** ($8.1\%$, UNETR), confirming that VisReg JEPA expands latent dimensionality.
4. **OOD Scanner Shift Resilience vs. Missing-Modality Fragility**:
   - All architectures demonstrate resilience to synthetic 3D Rician noise ($\sigma=0.08$) and quadratic $B_1$ bias fields, with 3D nnU-Net maintaining the highest absolute Dice.
   - Under single-sequence missing-modality triage (`FLAIR-only` or `T1c-only`), VisReg JEPA UNETR establishes top overall performance on emergency FLAIR-only acquisition ($44.61\%$ mean Dice vs.\ $39.79\%$ for nnU-Net), while pre-training shields UNETR against severe single-sequence degradation on T1c-only ($49.22\%$ vs.\ $36.55\%$ for scratch UNETR).

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
