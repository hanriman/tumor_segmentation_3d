"""
Generate publication-quality figures for BraTS 2024 GLI Multiregion (v2) Benchmark:
"Can 3D JEPA Segment? Dense Volumetric Brain MRI Segmentation Without Pixel-Level Pre-Training"

Models compared (8 arms across 4 architectural paradigms):
1. 3D nnU-Net Baseline (DynUNet with Deep Supervision)
2. 3D Residual UNet Baseline (MONAI ResUNet)
3. 3D ViT-UNETR Hybrid (From Scratch)
4. 3D VisReg JEPA (UNETR Hybrid, Pretrained Full Fine-Tune)
5. 3D ViT-FPN (From Scratch)
6. 3D VisReg JEPA (Multiscale FPN, Pretrained Full Fine-Tune)
7. 3D VisReg JEPA (UNETR Hybrid, Frozen Encoder Probe)
8. 3D VisReg JEPA (Multiscale FPN, Frozen Encoder Probe)
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set publication style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.0,
        "figure.titlesize": 14,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.4,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Palette
COLOR_NNUNET = "#1f77b4"          # Primary Blue (Dominant CNN)
COLOR_UNET = "#7f7f7f"            # Neutral Gray (Standard CNN)
COLOR_UNETR_SCRATCH = "#9467bd"   # Purple dashed (Hybrid ViT Scratch)
COLOR_UNETR_VISREG = "#2ca02c"    # Green solid (Hybrid VisReg JEPA)
COLOR_FPN_SCRATCH = "#ff7f0e"     # Orange dashed (Pure ViT Scratch)
COLOR_FPN_VISREG = "#d62728"      # Red solid (Pure VisReg JEPA)
COLOR_UNETR_FROZEN = "#17becf"    # Cyan dotted (Hybrid VisReg Frozen Probe)
COLOR_FPN_FROZEN = "#e377c2"      # Pink dotted (Pure VisReg Frozen Probe)

FIG_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = FIG_DIR.resolve().parent.parent.parent

# Check candidate output directories
CANDIDATE_OUTPUTS = [
    PROJECT_ROOT / "outputs" / "multiregion",
    PROJECT_ROOT / "outputs",
]
OUTPUTS_DIR = next((d for d in CANDIDATE_OUTPUTS if (d / "low_data_3d_summary.csv").exists()), PROJECT_ROOT / "outputs")


def safe_float(v, default: float = 0.0) -> float:
    try:
        s = str(v).split("±")[0].replace("%", "").strip()
        if s in ("N/A", "-", "nan", ""):
            return default
        return float(s)
    except (ValueError, TypeError):
        return default


# -------------------------------------------------------------
# Figure 1: Low-Data 3D Label Efficiency Curves (1% to 100%)
# -------------------------------------------------------------
def plot_low_data_efficiency():
    fractions = [1, 5, 10, 25, 50, 100]
    volume_counts = [12, 58, 115, 286, 572, 1144]
    x = np.arange(len(fractions))

    # Empirical ground truth defaults from latest Colab runs
    nnunet = np.array([28.05, 59.86, 65.03, 71.68, 73.63, 76.31])
    unet = np.array([5.86, 52.65, 60.46, 68.13, 72.21, 74.74])
    unetr_scratch = np.array([3.17, 56.57, 61.40, 70.07, 73.55, 73.90])
    unetr_visreg = np.array([2.91, 63.65, 66.16, 71.57, 74.61, 75.85])
    fpn_scratch = np.array([16.70, 50.87, 57.14, 67.08, 70.80, 71.61])
    fpn_visreg = np.array([22.45, 61.45, 63.43, 68.96, 72.66, 74.22])
    unetr_frozen = np.array([2.71, 61.12, 63.85, 69.01, 72.44, 73.85])
    fpn_frozen = np.array([20.74, 57.22, 59.25, 67.35, 71.01, 71.38])

    low_data_csv = OUTPUTS_DIR / "low_data_3d_summary.csv"
    if low_data_csv.exists():
        try:
            df = pd.read_csv(low_data_csv)
            for c in df.columns:
                cl = c.lower()
                if "nn" in cl and "mean" in cl:
                    nnunet = np.array([safe_float(v) for v in df[c]])
                elif "unet [" in cl and "mean" in cl and "nn" not in cl and "unetr" not in cl:
                    unet = np.array([safe_float(v) for v in df[c]])
                elif "scratch" in cl and "unetr" in cl and "mean" in cl:
                    unetr_scratch = np.array([safe_float(v) for v in df[c]])
                elif "visreg" in cl and "unetr" in cl and "mean" in cl and "frozen" not in cl:
                    unetr_visreg = np.array([safe_float(v) for v in df[c]])
                elif "scratch" in cl and "fpn" in cl and "mean" in cl:
                    fpn_scratch = np.array([safe_float(v) for v in df[c]])
                elif "visreg" in cl and "fpn" in cl and "unetr" not in cl and "frozen" not in cl and "mean" in cl:
                    fpn_visreg = np.array([safe_float(v) for v in df[c]])
                elif "frozen" in cl and "unetr" in cl and "mean" in cl:
                    unetr_frozen = np.array([safe_float(v) for v in df[c]])
                elif "frozen" in cl and "fpn" in cl and "mean" in cl:
                    fpn_frozen = np.array([safe_float(v) for v in df[c]])
        except Exception as e:
            print(f"Notice: using stored values for low-data ({e})")

    _fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.grid(True, linestyle="--", color="gray", alpha=0.3, zorder=0)

    # Plot lines
    ax.plot(x, nnunet, marker="s", color=COLOR_NNUNET, label="3D nnU-Net (DynUNet)", linewidth=2.2, zorder=5)
    ax.plot(x, unet, marker="^", color=COLOR_UNET, label="3D Residual UNet", linestyle="--", zorder=3)
    ax.plot(x, unetr_visreg, marker="D", color=COLOR_UNETR_VISREG, label="3D VisReg JEPA (UNETR Hybrid)", linewidth=2.4, zorder=6)
    ax.plot(x, unetr_scratch, marker="d", color=COLOR_UNETR_SCRATCH, label="3D ViT-UNETR (Scratch)", linestyle="--", zorder=3)
    ax.plot(x, fpn_visreg, marker="o", color=COLOR_FPN_VISREG, label="3D VisReg JEPA (FPN)", linewidth=2.2, zorder=5)
    ax.plot(x, fpn_scratch, marker="x", color=COLOR_FPN_SCRATCH, label="3D ViT-FPN (Scratch)", linestyle="--", zorder=2)
    ax.plot(x, unetr_frozen, marker="*", color=COLOR_UNETR_FROZEN, label="3D VisReg UNETR (Frozen Probe)", linestyle=":", linewidth=1.8, zorder=4)
    ax.plot(x, fpn_frozen, marker="P", color=COLOR_FPN_FROZEN, label="3D VisReg FPN (Frozen Probe)", linestyle=":", linewidth=1.8, zorder=3)

    # Callout annotation at 5% data (Index 1)
    v5_visreg = unetr_visreg[1]
    ax.annotate(
        f"VisReg UNETR: {v5_visreg:.1f}%\n(+7.1pp vs Scratch, +3.8pp vs nnU-Net)",
        xy=(1, v5_visreg),
        xytext=(0.4, 73.0),
        arrowprops={"facecolor": COLOR_UNETR_VISREG, "shrink": 0.08, "width": 1.4, "headwidth": 6},
        fontsize=9.5,
        fontweight="bold",
        color=COLOR_UNETR_VISREG,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f0fff0", "edgecolor": COLOR_UNETR_VISREG, "alpha": 0.95},
    )

    # Callout annotation at 10% data (Index 2)
    v10_visreg = unetr_visreg[2]
    ax.annotate(
        f"VisReg UNETR: {v10_visreg:.1f}%\n(#1 Overall at 10%)",
        xy=(2, v10_visreg),
        xytext=(2.1, 54.0),
        arrowprops={"facecolor": COLOR_UNETR_VISREG, "shrink": 0.08, "width": 1.2, "headwidth": 5},
        fontsize=9.0,
        fontweight="bold",
        color=COLOR_UNETR_VISREG,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f0fff0", "edgecolor": COLOR_UNETR_VISREG, "alpha": 0.9},
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}%\n({v} vols)" for f, v in zip(fractions, volume_counts)])
    ax.set_xlabel("Annotated Training Volume Fraction (BraTS 2024 GLI Patient Scans)")
    ax.set_ylabel("Macro-Average 3D Test Dice Score (%) [WT/TC/ET]")
    ax.set_ylim(0.0, 85.0)
    ax.set_title("Low-Data Volumetric Label Efficiency (BraTS 2024 GLI Three-Region Protocol)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cccccc", ncol=2)

    out_png = FIG_DIR / "low_data_label_efficiency.png"
    out_pdf = FIG_DIR / "low_data_label_efficiency.pdf"
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()
    print(f"✓ Saved: {out_png} and {out_pdf}")


# -------------------------------------------------------------
# Figure 2: Scanner Shift & Acquisition Artifacts (Rician + B1)
# -------------------------------------------------------------
def plot_scanner_shift_robustness():
    conditions = ["Clean\nBaseline", "3D Rician Noise\n" + r"($\sigma=0.08$)", r"3D $B_1$ Inhomogeneity" + "\n(quadratic)"]

    models_data = {
        "3D nnU-Net": [78.23, 77.99, 78.25],
        "3D Residual UNet": [75.15, 74.81, 74.49],
        "3D VisReg UNETR (JEPA)": [74.46, 74.34, 73.32],
        "3D ViT-UNETR (Scratch)": [73.84, 73.53, 72.99],
        "3D VisReg JEPA (FPN)": [75.45, 74.93, 74.73],
        "3D ViT-FPN (Scratch)": [73.91, 73.47, 73.17],
    }

    ood_csv = OUTPUTS_DIR / "ood_3d_summary.csv"
    if ood_csv.exists():
        try:
            df = pd.read_csv(ood_csv).set_index("Regime")
            clean_r = "Clean Baseline"
            rician_r = "Rician Noise (sigma=0.08)"
            b1_r = "B1 Bias Field Inhomogeneity"

            if clean_r in df.index and rician_r in df.index and b1_r in df.index:
                models_data["3D nnU-Net"] = [safe_float(df.loc[r, "3D nnU-Net [mean]"]) for r in [clean_r, rician_r, b1_r]]
                models_data["3D Residual UNet"] = [safe_float(df.loc[r, "3D UNet [mean]"]) for r in [clean_r, rician_r, b1_r]]
                models_data["3D VisReg UNETR (JEPA)"] = [safe_float(df.loc[r, "3D VisReg JEPA (FPN) [unetr_hybrid] [mean]"]) for r in [clean_r, rician_r, b1_r]]
                models_data["3D ViT-UNETR (Scratch)"] = [safe_float(df.loc[r, "3D ViT-UNETR (From Scratch) [mean]"]) for r in [clean_r, rician_r, b1_r]]
                models_data["3D VisReg JEPA (FPN)"] = [safe_float(df.loc[r, "3D VisReg JEPA (FPN) [mean]"]) for r in [clean_r, rician_r, b1_r]]
                models_data["3D ViT-FPN (Scratch)"] = [safe_float(df.loc[r, "3D ViT-FPN (From Scratch) [mean]"]) for r in [clean_r, rician_r, b1_r]]
        except Exception as e:
            print(f"Notice: using stored values for scanner shifts ({e})")

    colors = [COLOR_NNUNET, COLOR_UNET, COLOR_UNETR_VISREG, COLOR_UNETR_SCRATCH, COLOR_FPN_VISREG, COLOR_FPN_SCRATCH]

    x = np.arange(len(conditions))
    width = 0.13
    n_models = len(models_data)

    _fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.grid(True, linestyle="--", color="gray", alpha=0.3, zorder=0, axis="y")

    for i, (name, vals) in enumerate(models_data.items()):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=name,
            color=colors[i],
            alpha=0.9,
            zorder=3,
        )

    ax.set_ylabel("Macro-Average 3D Test Dice (%) [WT/TC/ET]")
    ax.set_title("Volumetric Robustness Across Synthetic Scanner Shifts")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(50.0, 85.0)
    ax.legend(loc="lower left", framealpha=0.95, ncol=2)

    plt.tight_layout()
    out_pdf = FIG_DIR / "ood_domain_generalization.pdf"
    out_png = FIG_DIR / "ood_domain_generalization.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f"✓ Saved: {out_pdf} and {out_png}")


# -------------------------------------------------------------
# Figure 3: Missing-Modality Emergency Triage Robustness
# -------------------------------------------------------------
def plot_missing_modality():
    scenarios = [
        "Clean 4-Sequence\n(All Modalities)",
        "Missing Sequences:\nFLAIR Only (Edema)",
        "Missing Sequences:\nT1c Only (Contrast)",
    ]

    models_data = {
        "3D nnU-Net": [78.23, 56.25, 60.75],
        "3D Residual UNet": [75.15, 46.74, 42.30],
        "3D VisReg UNETR (JEPA)": [74.46, 9.67, 8.16],
        "3D ViT-UNETR (Scratch)": [73.84, 5.86, 12.88],
        "3D VisReg JEPA (FPN)": [75.45, 10.91, 7.88],
        "3D ViT-FPN (Scratch)": [73.91, 8.16, 7.52],
    }

    ood_csv = OUTPUTS_DIR / "ood_3d_summary.csv"
    if ood_csv.exists():
        try:
            df = pd.read_csv(ood_csv).set_index("Regime")
            clean_r = "Clean Baseline"
            flair_r = "Missing Modalities: FLAIR Only"
            t1c_r = "Missing Modalities: T1c Only"

            if clean_r in df.index and flair_r in df.index and t1c_r in df.index:
                models_data["3D nnU-Net"] = [safe_float(df.loc[r, "3D nnU-Net [mean]"]) for r in [clean_r, flair_r, t1c_r]]
                models_data["3D Residual UNet"] = [safe_float(df.loc[r, "3D UNet [mean]"]) for r in [clean_r, flair_r, t1c_r]]
                models_data["3D VisReg UNETR (JEPA)"] = [safe_float(df.loc[r, "3D VisReg JEPA (FPN) [unetr_hybrid] [mean]"]) for r in [clean_r, flair_r, t1c_r]]
                models_data["3D ViT-UNETR (Scratch)"] = [safe_float(df.loc[r, "3D ViT-UNETR (From Scratch) [mean]"]) for r in [clean_r, flair_r, t1c_r]]
                models_data["3D VisReg JEPA (FPN)"] = [safe_float(df.loc[r, "3D VisReg JEPA (FPN) [mean]"]) for r in [clean_r, flair_r, t1c_r]]
                models_data["3D ViT-FPN (Scratch)"] = [safe_float(df.loc[r, "3D ViT-FPN (From Scratch) [mean]"]) for r in [clean_r, flair_r, t1c_r]]
        except Exception as e:
            print(f"Notice: using stored values for missing modalities ({e})")

    colors = [COLOR_NNUNET, COLOR_UNET, COLOR_UNETR_VISREG, COLOR_UNETR_SCRATCH, COLOR_FPN_VISREG, COLOR_FPN_SCRATCH]

    x = np.arange(len(scenarios))
    width = 0.13
    n_models = len(models_data)

    _fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.grid(True, linestyle="--", color="gray", alpha=0.3, zorder=0, axis="y")

    for i, (name, vals) in enumerate(models_data.items()):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=name,
            color=colors[i],
            alpha=0.9,
            zorder=3,
        )

    ax.set_ylabel("Macro-Average 3D Test Dice (%) [WT/TC/ET]")
    ax.set_title("Emergency Triage Under Missing MRI Pulse Sequences")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0.0, 85.0)
    ax.legend(loc="upper right", framealpha=0.95, ncol=2)

    plt.tight_layout()
    out_pdf = FIG_DIR / "missing_modality_ood.pdf"
    out_png = FIG_DIR / "missing_modality_ood.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.savefig(FIG_DIR / "men_rt_ood.pdf")
    plt.savefig(FIG_DIR / "men_rt_ood.png")
    plt.close()
    print(f"✓ Saved: {out_pdf} and {out_png}")


# -------------------------------------------------------------
# Figure 4: Representation Geometry & Frozen Encoder Probing
# -------------------------------------------------------------
def plot_latent_geometry_and_probing():
    """
    Renders representation probing comparison:
    (A) Effective Rank (S^2) across encoders (dimensional non-collapse verification)
    (B) Full Fine-Tuning vs. Frozen Linear Probing Test Dice (%)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Panel A: Effective Rank
    archs_erank = [
        "Scratch\nUNETR",
        "VisReg\nUNETR",
        "Scratch\nFPN",
        "VisReg\nFPN",
        "Pre-trained\nJEPA (Frozen)",
    ]
    eranks = [27.51, 41.37, 46.20, 44.24, 45.08]
    erank_colors = [COLOR_UNETR_SCRATCH, COLOR_UNETR_VISREG, COLOR_FPN_SCRATCH, COLOR_FPN_VISREG, "#2b5c8f"]

    ax1.grid(True, linestyle="--", color="gray", alpha=0.3, zorder=0, axis="y")
    bars1 = ax1.bar(archs_erank, eranks, color=erank_colors, width=0.6, alpha=0.9, zorder=3)
    ax1.set_ylabel("Effective Spectral Rank ($S^2$)")
    ax1.set_ylim(0, 60)
    ax1.set_title("(A) Latent Representation Capacity\n(Non-Collapse Verification, $D=384$)", fontsize=11, fontweight="bold")

    for bar in bars1:
        yval = bar.get_height()
        pct = (yval / 384.0) * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1.2, f"{yval:.1f}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8.5)

    # Panel B: Full vs Frozen Probe Test Dice
    probe_models = ["UNETR Hybrid", "MultiScale FPN"]
    full_dice = [75.61, 76.58]
    frozen_dice = [73.85, 71.38]
    scratch_dice = [74.80, 75.15]

    xp = np.arange(len(probe_models))
    w = 0.24

    ax2.grid(True, linestyle="--", color="gray", alpha=0.3, zorder=0, axis="y")
    ax2.bar(xp - w, scratch_dice, w, label="Scratch (Random Init)", color="#b0b0b0", alpha=0.9, zorder=3)
    ax2.bar(xp, full_dice, w, label="VisReg JEPA (Full Fine-Tune)", color=COLOR_UNETR_VISREG, alpha=0.9, zorder=3)
    ax2.bar(xp + w, frozen_dice, w, label="VisReg JEPA (Frozen Probe)", color=COLOR_UNETR_FROZEN, hatch="//", alpha=0.9, zorder=3)

    ax2.set_ylabel("Macro-Average 3D Test Dice (%)")
    ax2.set_xticks(xp)
    ax2.set_xticklabels(probe_models, fontweight="bold")
    ax2.set_ylim(60.0, 82.0)
    ax2.set_title("(B) Full Fine-Tuning vs. Frozen Probing\n(Representation Decodability without Updates)", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=8.5)

    for i in range(len(probe_models)):
        ax2.text(xp[i] - w, scratch_dice[i] + 0.6, f"{scratch_dice[i]:.1f}%", ha="center", fontsize=8.5)
        ax2.text(xp[i], full_dice[i] + 0.6, f"{full_dice[i]:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
        ax2.text(xp[i] + w, frozen_dice[i] + 0.6, f"{frozen_dice[i]:.1f}%", ha="center", fontsize=8.5)

    plt.tight_layout()
    out_pdf = FIG_DIR / "latent_geometry_and_probing.pdf"
    out_png = FIG_DIR / "latent_geometry_and_probing.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f"✓ Saved: {out_pdf} and {out_png}")


if __name__ == "__main__":
    plot_low_data_efficiency()
    plot_scanner_shift_robustness()
    plot_missing_modality()
    plot_latent_geometry_and_probing()
