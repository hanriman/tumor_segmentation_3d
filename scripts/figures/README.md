# 3D Model Architecture Figure Generation Suite

This directory contains standalone Python generators that construct publication-grade 3D isometric architecture schematics for the thesis paper and documentation using [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) and LaTeX TikZ.

---

## 1. Generator Scripts Overview

| Script | Model Architecture | Role & Highlights |
| :--- | :--- | :--- |
| **`plot_jepa_pretrain_3d.py`** | **3D VisReg JEPA (Pre-Training)** | 3D multi-modal input ($4 \times 128^3$), $8^3$ patch embedding, 4,096 tokens, context-target masking, ViT encoder, 4-layer predictor, direct latent space regularization ($M=1024$ rays). |
| **`plot_downstream_fpn_3d.py`** | **ViT + Multi-Scale FPN** | Pure latent probe with zero raw-image skips; lateral taps at L2, L4, L6, L8; 3D transposed conv upsampling pyramid; 5-channel output head. |
| **`plot_downstream_unetr_3d.py`** | **ViT + Hybrid UNETR** | Dual-stream hybrid decoder; early convolutional stem; lateral taps at L2, L4, L6, L8 with deconvolutional bridges and skip fusion. |
| **`plot_unet_3d.py`** | **3D Residual U-Net Baseline** | MONAI ResUNet baseline; 5-level channel pyramid (32, 64, 128, 256, 512); 4 deep supervision auxiliary heads. |
| **`plot_nnunet_3d.py`** | **3D nnU-Net Baseline** | DynUNet baseline; anisotropic kernel strides; instance norm + leaky ReLU blocks; multi-scale deep supervision taps. |
| **`build_all_figures.py`** | **Master Batch Compiler** | Sequentially executes all 5 generators, compiles TikZ code, verifies vector PDFs and 300-DPI PNGs across output directories. |

---

## 2. Compilation Flow & Output Hierarchy

The compilation pipeline is non-polluting: all intermediate build artifacts (`.aux`, `.log`, `.out`, temporary `.pdf`) are automatically purged, keeping `scripts/figures/` strictly for source Python scripts:

```text
scripts/figures/ (Python code only)
       │
       ▼
docs/figures/tex/ (TikZ LaTeX source: arch_*.tex)
       │  [pdflatex]
       ▼
Vector PDF & 300-DPI PNG (Rendered via sips / pdftoppm)
       ├── docs/figures/        (For documentation, web, and presentation)
       └── paper/latex/figures/ (For LaTeX thesis manuscript inclusion)
```

---

## 3. How to Build Figures

### Prerequisites:
- Python 3.10+
- `pdflatex` (MacTeX / TeX Live)
- `sips` (macOS native) or `pdftoppm`

### Build Command:
```bash
# Build and compile all 5 architecture diagrams:
python scripts/figures/build_all_figures.py

# Or build an individual architecture:
python scripts/figures/plot_downstream_fpn_3d.py
```
