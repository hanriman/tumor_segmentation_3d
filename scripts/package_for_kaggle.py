#!/usr/bin/env python3
"""
Automated Kaggle Dataset Packaging Script for BraTS 3D Multimodal JEPA Project.
Packages processed 3D .npz volumes and metadata.csv into clean, upload-ready zip archives for Kaggle.

Usage:
    python scripts/package_for_kaggle.py
    python scripts/package_for_kaggle.py --output_dir dist_kaggle
    python scripts/package_for_kaggle.py --dataset_name brats_gli_3d
"""

import argparse
import os
import time
import zipfile
from pathlib import Path

# Root directory of thesis_3d
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_PATTERNS = {
    "__pycache__",
    ".DS_Store",
    "Thumbs.db",
}


def should_exclude(rel_path: Path) -> bool:
    """Returns True if any part of the path matches an exclusion pattern."""
    for part in rel_path.parts:
        if part in EXCLUDE_PATTERNS:
            return True
        if any(part.endswith(pat) for pat in [".pyc", ".pyo", ".swp"]):
            return True
    return False


def format_size(num_bytes: int) -> str:
    """Formats byte count into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def package_dataset(
    dataset_name: str,
    dataset_dir: Path,
    output_zip: Path,
    arc_prefix: str = "",
):
    """Packages a single processed 3D dataset directory into a zip file."""
    if not dataset_dir.exists():
        print(f"⚠️  Dataset directory not found: {dataset_dir}. Skipping.")
        return 0, 0

    print(f"\n📦 Packaging 3D Dataset: {dataset_name} ({dataset_dir}) -> {output_zip} ...")
    t0 = time.perf_counter()
    file_count = 0
    total_uncompressed = 0

    all_files = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            p = Path(root) / f
            if not should_exclude(p.relative_to(dataset_dir)):
                all_files.append(p)

    total_files = len(all_files)
    if total_files == 0:
        print(f"⚠️  No files found in {dataset_dir}!")
        return 0, 0

    print(f"   Scanning found {total_files:,} files...")

    # Ensure output directory exists
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    # Write to zip with DEFLATED compression
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, file_path in enumerate(all_files, start=1):
            rel_path = file_path.relative_to(dataset_dir)
            arc_name = f"{arc_prefix}/{rel_path}" if arc_prefix else f"{dataset_name}/{rel_path}"
            zf.write(file_path, arcname=arc_name)
            file_count += 1
            total_uncompressed += file_path.stat().st_size
            if idx % 100 == 0 or idx == total_files:
                print(
                    f"   Progress: {idx:,} / {total_files:,} files ({idx / total_files * 100:.1f}%)"
                )

    elapsed = time.perf_counter() - t0
    zip_size = output_zip.stat().st_size
    print(f"   ✓ Archived {file_count:,} files from {dataset_name}")
    print(
        f"   ✓ Completed in {elapsed:.1f}s | Uncompressed: {format_size(total_uncompressed)} | Archive: {format_size(zip_size)}"
    )
    return file_count, total_uncompressed


def parse_args():
    parser = argparse.ArgumentParser(description="Package 3D datasets for Kaggle.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dist_kaggle",
        help="Target output directory for zip packages (default: dist_kaggle)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="brats_gli_3d",
        help="Processed dataset folder name under data/processed/ (default: brats_gli_3d)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      KAGGLE 3D DATASET PACKAGING PIPELINE")
    print("=" * 80)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Output directory: {out_dir}")

    target_dir = PROJECT_ROOT / "data" / "processed" / args.dataset_name
    output_zip = out_dir / "brats_3d_datasets.zip"

    file_count, _uncompressed_bytes = package_dataset(
        args.dataset_name,
        target_dir,
        output_zip,
        arc_prefix=args.dataset_name,
    )

    if file_count > 0:
        print("\n" + "=" * 80)
        print("🎉 SUCCESS: Kaggle 3D Dataset archive is ready for upload!")
        print(f"📍 Location: {output_zip}")
        print("Next steps:")
        print("  1. Go to kaggle.com/datasets -> 'New Dataset'")
        print(f"  2. Upload {output_zip.name} with title 'brats-3d-datasets'")
        print("  3. Attach the dataset to your Kaggle GPU notebook!")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
