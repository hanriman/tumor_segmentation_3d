#!/usr/bin/env python3
"""
1-Click Master Aggregator & Paper Artifacts Generator for BraTS 3D Volumetric Benchmark.
Discovers multi-experiment outputs (e.g., outputs/kaggle_visreg_5_epoch, outputs/kaggle_nnunet_5_epoch, outputs/kaggle_unet_5_epoch),
combines their test benchmark metrics and low-data curves, renders all 4 publication figures, formats LaTeX tables,
and packages paper_artifacts.zip locally in seconds with zero cloud re-uploading.

Usage:
    python scripts/combine_and_generate_paper_artifacts.py
    python scripts/combine_and_generate_paper_artifacts.py --outputs_dir outputs
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
from brats_jepa_3d.utils.aggregation import (
    aggregate_low_data_summaries,
    aggregate_master_benchmarks,
    aggregate_ood_summaries,
    discover_experiment_directories,
    export_latex_tables,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-experiment results, generate publication figures, and package paper artifacts."
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        help="Root outputs directory containing experiment subfolders.",
    )
    parser.add_argument(
        "--experiment_dirs",
        nargs="+",
        default=None,
        help="Explicit list of experiment directories to aggregate (default: auto-discover).",
    )
    parser.add_argument(
        "--no_figures",
        action="store_true",
        help="Skip generating publication figures.",
    )
    parser.add_argument(
        "--no_zip",
        action="store_true",
        help="Skip building paper_artifacts.zip.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outputs_root = Path(args.outputs_dir).resolve()
    t0 = time.time()

    print("=" * 70)
    print("📊 BraTS 3D Multi-Experiment Aggregator & Paper Artifacts Generator")
    print("=" * 70)
    print(f"📁 Outputs Root: {outputs_root}")

    # 1. Discover experiment directories
    if args.experiment_dirs:
        exp_dirs = [Path(d).resolve() for d in args.experiment_dirs]
    else:
        exp_dirs = discover_experiment_directories(outputs_root)

    print(f"\n🔍 Discovered {len(exp_dirs)} experiment folder(s):")
    for d in exp_dirs:
        print(f"  • {d.name} ({d})")

    if not exp_dirs:
        print("⚠️ No experiment directories found! Ensure experiment folders exist in outputs/.")
        sys.exit(0)

    # 2. Aggregate Master 3D Benchmarks
    print("\n" + "-" * 50)
    print("1️⃣ Aggregating Master Volumetric Test Benchmarks...")
    master_df = aggregate_master_benchmarks(exp_dirs, output_dir=outputs_root)
    if not master_df.empty:
        print(master_df.to_markdown(index=False))
    else:
        print("⚠️ No master benchmark files found across experiment folders.")

    # 3. Aggregate Low-Data Label Efficiency Curves
    print("\n" + "-" * 50)
    print("2️⃣ Aggregating Low-Data Label Efficiency Benchmarks...")
    low_data_df = aggregate_low_data_summaries(exp_dirs, output_dir=outputs_root)
    if not low_data_df.empty:
        print(low_data_df.to_markdown(index=False))
    else:
        print("⚠️ No low-data summary files found across experiment folders.")

    # 4. Aggregate OOD Robustness Summaries
    print("\n" + "-" * 50)
    print("3️⃣ Aggregating Out-of-Distribution (OOD) Stress Tests...")
    ood_df = aggregate_ood_summaries(exp_dirs, output_dir=outputs_root)
    if not ood_df.empty:
        print(ood_df.to_markdown(index=False))
    else:
        print("ℹ️ No OOD summary files found (or single-experiment evaluation).")

    # 5. Export LaTeX Tables
    print("\n" + "-" * 50)
    print("4️⃣ Compiling Publication LaTeX Tables...")
    latex_path = outputs_root / "tables_latex.tex"
    export_latex_tables(
        master_df=master_df if not master_df.empty else None,
        low_data_df=low_data_df if not low_data_df.empty else None,
        ood_df=ood_df if not ood_df.empty else None,
        output_path=latex_path,
    )
    print(f"✓ Saved LaTeX tables to: {latex_path}")

    # 6. Generate Publication Figures
    if not args.no_figures:
        print("\n" + "-" * 50)
        print("5️⃣ Rendering Publication Figures (Vector PDF & 300 DPI PNG)...")
        gen_figures_py = PROJECT_ROOT / "paper" / "latex" / "figures" / "gen_figures.py"
        if gen_figures_py.exists():
            print("  🎨 Running paper/latex/figures/gen_figures.py ...")
            subprocess.run([sys.executable, str(gen_figures_py)], check=False)

        gen_figures_3d_py = PROJECT_ROOT / "scripts" / "generate_figures_3d.py"
        if gen_figures_3d_py.exists():
            print("  🎨 Running scripts/generate_figures_3d.py ...")
            subprocess.run([sys.executable, str(gen_figures_3d_py), "--output_dir", str(outputs_root / "figures"), "--metrics_dir", str(outputs_root)], check=False)

    # 7. Package Paper Artifacts
    if not args.no_zip:
        print("\n" + "-" * 50)
        print("6️⃣ Packaging All Paper Artifacts into paper_artifacts.zip ...")
        staging_dir = outputs_root / "paper_artifacts"
        zip_path = outputs_root / "paper_artifacts.zip"
        staging_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["figures", "metrics", "tables", "logs"]:
            (staging_dir / sub).mkdir(parents=True, exist_ok=True)

        copied_count = 0
        # Copy figures
        latex_figs = PROJECT_ROOT / "paper" / "latex" / "figures"
        if latex_figs.exists():
            for f in latex_figs.glob("*.*"):
                if f.suffix.lower() in [".pdf", ".png"]:
                    shutil.copy2(f, staging_dir / "figures" / f.name)
                    copied_count += 1

        out_figs = outputs_root / "figures"
        if out_figs.exists():
            for f in out_figs.glob("*.*"):
                if f.suffix.lower() in [".pdf", ".png"]:
                    shutil.copy2(f, staging_dir / "figures" / f.name)
                    copied_count += 1

        # Copy aggregated metrics & LaTeX tables
        for f in outputs_root.glob("*.csv"):
            shutil.copy2(f, staging_dir / "metrics" / f.name)
            copied_count += 1
        for f in outputs_root.glob("*.md"):
            shutil.copy2(f, staging_dir / "metrics" / f.name)
            copied_count += 1
        if latex_path.exists():
            shutil.copy2(latex_path, staging_dir / "tables" / latex_path.name)
            copied_count += 1

        # Compress to zip
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(staging_dir):
                for f in files:
                    fp = Path(root) / f
                    zf.write(fp, arcname=fp.relative_to(staging_dir))

        zip_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"✓ Archived {copied_count} files into {zip_path} ({zip_mb:.2f} MB)")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"⏱️ Master Aggregation Completed in {elapsed:.2f} seconds!")
    print(f"📄 Summary Markdown: {outputs_root / 'master_3d_benchmark.md'}")
    print(f"📈 Low-Data Markdown: {outputs_root / 'low_data_3d_summary.md'}")
    print(f"📦 Archive Download: {outputs_root / 'paper_artifacts.zip'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
