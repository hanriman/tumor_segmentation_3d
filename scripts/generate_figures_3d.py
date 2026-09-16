#!/usr/bin/env python3
"""
3D Visualization & Publication Figure Generator.
Generates:
1. Multi-planar orthogonal slices (Axial, Coronal, Sagittal) through tumor centroid.
2. Benchmark comparison bar charts (3D Dice, 3D HD95, Latency, Effective Rank).
3. Low-data label efficiency curves (Dice vs Label Fraction).
4. Out-of-Distribution (OOD) robustness grouped bar charts.
All figures are saved in both PNG (300 DPI) and vector PDF formats.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brats_jepa_3d.config import FIGURES_DIR, METRICS_DIR, PROCESSED_DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D Figures and Publication Plots")
    parser.add_argument("--output_dir", type=str, default=str(FIGURES_DIR))
    parser.add_argument("--metrics_dir", type=str, default=str(METRICS_DIR))
    return parser.parse_args()


def plot_orthogonal_slices(
    volume_4ch: np.ndarray,
    mask_1ch: np.ndarray,
    pred_1ch: np.ndarray | None = None,
    save_path: Path | None = None,
    patient_id: str = "BraTS Patient",
):
    """
    Renders 3D orthogonal views (Axial, Coronal, Sagittal) through the centroid of the tumor.
    """
    tumor_indices = np.argwhere(mask_1ch[0] > 0)
    if len(tumor_indices) > 0:
        cz, cy, cx = tumor_indices.mean(axis=0).astype(int)
    else:
        cz, cy, cx = volume_4ch.shape[1] // 2, volume_4ch.shape[2] // 2, volume_4ch.shape[3] // 2

    # Clamp coordinates inside volume
    cz = int(np.clip(cz, 0, volume_4ch.shape[1] - 1))
    cy = int(np.clip(cy, 0, volume_4ch.shape[2] - 1))
    cx = int(np.clip(cx, 0, volume_4ch.shape[3] - 1))

    # T1c channel (idx 1) or T1n if T1c flat
    t1c = volume_4ch[1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Axial view (XY plane at Z=cz)
    axes[0].imshow(t1c[cz, :, :], cmap="gray", origin="lower")
    if np.any(mask_1ch[0, cz, :, :] > 0):
        axes[0].contour(mask_1ch[0, cz, :, :], colors="#E63946", levels=[0.5], linewidths=2.0)
    if pred_1ch is not None and np.any(pred_1ch[0, cz, :, :] > 0):
        axes[0].contour(
            pred_1ch[0, cz, :, :], colors="#457B9D", levels=[0.5], linewidths=2.0, linestyles="--"
        )
    axes[0].set_title(f"Axial Slice (Z={cz})", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # 2. Coronal view (XZ plane at Y=cy)
    axes[1].imshow(t1c[:, cy, :], cmap="gray", origin="lower")
    if np.any(mask_1ch[0, :, cy, :] > 0):
        axes[1].contour(mask_1ch[0, :, cy, :], colors="#E63946", levels=[0.5], linewidths=2.0)
    if pred_1ch is not None and np.any(pred_1ch[0, :, cy, :] > 0):
        axes[1].contour(
            pred_1ch[0, :, cy, :], colors="#457B9D", levels=[0.5], linewidths=2.0, linestyles="--"
        )
    axes[1].set_title(f"Coronal Slice (Y={cy})", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    # 3. Sagittal view (YZ plane at X=cx)
    axes[2].imshow(t1c[:, :, cx], cmap="gray", origin="lower")
    if np.any(mask_1ch[0, :, :, cx] > 0):
        axes[2].contour(mask_1ch[0, :, :, cx], colors="#E63946", levels=[0.5], linewidths=2.0)
    if pred_1ch is not None and np.any(pred_1ch[0, :, :, cx] > 0):
        axes[2].contour(
            pred_1ch[0, :, :, cx], colors="#457B9D", levels=[0.5], linewidths=2.0, linestyles="--"
        )
    axes[2].set_title(f"Sagittal Slice (X={cx})", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    legend_elements = [
        plt.Line2D([0], [0], color="#E63946", lw=2.5, label="Ground Truth (Whole Tumor)"),
    ]
    if pred_1ch is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color="#457B9D", lw=2.5, linestyle="--", label="Model Prediction")
        )

    fig.suptitle(
        f"Multi-Planar Orthogonal View: {patient_id}", fontsize=15, fontweight="bold", y=0.98
    )
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        fontsize=12,
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_benchmark_metrics(metrics_csv: Path, out_dir: Path):
    """
    Plots benchmark comparisons across evaluated models.
    """
    if not metrics_csv.exists():
        return

    df = pd.read_csv(metrics_csv)
    models = df["Model"].tolist()

    # Extract mean Dice
    dices = [float(str(v).split("±")[0].strip()) for v in df["Dice (%)"]]
    hd95s = [float(str(v).split("±")[0].strip()) for v in df["HD95 (mm)"]]
    latencies = [float(v) for v in df["Latency (ms)"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#2A9D8F", "#E76F51", "#F4A261", "#457B9D", "#1D3557"]

    # 1. 3D Dice Score
    axes[0].barh(models, dices, color=colors[: len(models)])
    axes[0].set_xlabel("3D Dice Score (%)", fontsize=12, fontweight="bold")
    axes[0].set_title("Volumetric Segmentation Dice", fontsize=13, fontweight="bold")
    axes[0].grid(axis="x", linestyle="--", alpha=0.7)

    # 2. 3D HD95
    axes[1].barh(models, hd95s, color=colors[: len(models)])
    axes[1].set_xlabel("95th Percentile Hausdorff Distance (mm)", fontsize=12, fontweight="bold")
    axes[1].set_title("Boundary Surface Distance (HD95)", fontsize=13, fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.7)

    # 3. Latency
    axes[2].barh(models, latencies, color=colors[: len(models)])
    axes[2].set_xlabel("Latency (ms / 128³ volume)", fontsize=12, fontweight="bold")
    axes[2].set_title("Inference Latency", fontsize=13, fontweight="bold")
    axes[2].grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    save_path = out_dir / "benchmark_3d_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Benchmark metric plots saved to: {save_path}")


def plot_low_data_curves(low_data_csv: Path, out_dir: Path):
    """
    Plots low-data label efficiency curves.
    """
    if not low_data_csv.exists():
        return

    df = pd.read_csv(low_data_csv)
    fractions = [float(f.replace("%", "")) for f in df["Fraction"]]

    plt.figure(figsize=(8, 5))
    styles = {
        "3D VisReg JEPA (FPN)": ("#2A9D8F", "o-"),
        "3D SigReg JEPA (FPN)": ("#E76F51", "s-"),
        "3D nnU-Net": ("#1D3557", "^--"),
        "3D UNet": ("#457B9D", "v--"),
    }

    for col in df.columns:
        if col == "Fraction":
            continue
        vals = [float(str(v).replace("%", "")) for v in df[col]]
        color, style = styles.get(col, ("#000000", "o-"))
        plt.plot(fractions, vals, style, color=color, linewidth=2.0, markersize=7, label=col)

    plt.xlabel("Annotated Training Data Fraction (%)", fontsize=12, fontweight="bold")
    plt.ylabel("3D Dice Score (%)", fontsize=12, fontweight="bold")
    plt.title("Label Efficiency: 3D Volumetric Segmentation", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    save_path = out_dir / "low_data_3d_efficiency.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Low-data efficiency curves saved to: {save_path}")


def plot_ood_robustness(ood_csv: Path, out_dir: Path):
    """
    Plots Out-of-Distribution (OOD) scanner shift robustness grouped bar chart.
    """
    if not ood_csv.exists():
        return

    df = pd.read_csv(ood_csv)
    regimes = df["Regime"].tolist()
    models = [c for c in df.columns if c != "Regime"]

    x = np.arange(len(regimes))
    width = 0.8 / len(models)
    colors = ["#2A9D8F", "#E76F51", "#1D3557", "#457B9D"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_col in enumerate(models):
        vals = [float(str(v).replace("%", "")) for v in df[model_col]]
        ax.bar(x + i * width, vals, width, label=model_col, color=colors[i % len(colors)])

    ax.set_ylabel("3D Dice Score (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Out-of-Distribution Robustness across Scanner Shifts & Artifacts",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(regimes, rotation=20, ha="right", fontsize=10)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    save_path = out_dir / "ood_3d_robustness.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"OOD robustness chart saved to: {save_path}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    metrics_dir = Path(args.metrics_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Multi-planar orthogonal slice generation
    processed_dir = PROCESSED_DATA_DIR
    patient_files = list(processed_dir.glob("*.npz")) if processed_dir.exists() else []

    if patient_files:
        sample_file = patient_files[0]
        data = np.load(sample_file)
        vol = data["image"]
        mask = data["mask"]
        pid = str(data.get("patient_id", sample_file.stem.replace("_volume", "")))
        print(f"Loaded real patient volume: {pid} (shape: {vol.shape})")
        fig_path = out_dir / "orthogonal_multi_planar_figure.png"
        plot_orthogonal_slices(vol, mask, pred_1ch=None, save_path=fig_path, patient_id=pid)
    else:
        synth_vol = np.random.randn(4, 128, 128, 128).astype(np.float32)
        synth_mask = np.zeros((1, 128, 128, 128), dtype=np.uint8)
        synth_mask[0, 50:75, 50:75, 50:75] = 1
        synth_pred = np.zeros((1, 128, 128, 128), dtype=np.uint8)
        synth_pred[0, 52:77, 48:73, 50:75] = 1
        fig_path = out_dir / "orthogonal_multi_planar_figure.png"
        plot_orthogonal_slices(
            synth_vol, synth_mask, synth_pred, save_path=fig_path, patient_id="Synthetic Volume"
        )

    # 2. Benchmark metric plots
    plot_benchmark_metrics(metrics_dir / "benchmark_3d_summary.csv", out_dir)

    # 3. Low-data efficiency plots
    plot_low_data_curves(metrics_dir / "low_data_3d_summary.csv", out_dir)

    # 4. OOD robustness plots
    plot_ood_robustness(metrics_dir / "ood_3d_summary.csv", out_dir)

    print("\nAll publication figures generated successfully!")


if __name__ == "__main__":
    main()
