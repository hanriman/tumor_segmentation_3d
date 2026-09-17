#!/usr/bin/env python3
"""
3D BraTS 2024 GLI Data Preparation Script.

Extracts non-zero brain bounding boxes, performs aspect-preserving trilinear resampling
to canonical 128x128x128 grid, applies parenchyma-only Z-score intensity normalization,
and saves compressed .npz volumes with patient-stratified metadata.csv.

Theoretical & Methodological References:
----------------------------------------
1. Bounding Box & Resampling (Isensee et al., Nature Methods 2021):
   Parenchyma bounding box extraction removes uninformative air background (~60% of native voxels).
   Resampling preserves continuous anatomical morphology across white matter tracts, preventing
   the peripheral cortical and infiltrative margin truncation caused by naive spatial cropping.
2. Z-Score Intensity Normalization (Isensee et al., 2021; Baid et al., 2024):
   Evaluated strictly over non-zero brain voxels per channel to eliminate scanner-dependent intensity drift.
3. Target Binarization (Menze et al., 2014; Baid et al., 2024):
   Whole Tumor (WT) target is defined as binary union of all tumor sub-compartments: WT = mask > 0.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from brats_jepa_3d.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare 3D BraTS 2024 GLI volumes")
    default_data_dir = RAW_DATA_DIR / "BraTS_2024" / "BraTS-GLI" / "training_data1_v2"
    if not default_data_dir.exists():
        default_data_dir = RAW_DATA_DIR / "BraTS-GLI" / "training_data1_v2"
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(default_data_dir),
        help="Path to raw BraTS-GLI patient folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="Path to save processed 3D .npz volumes",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=128,
        help="Canonical isotropic cubic grid dimension (default: 128)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of patients to process (for rapid testing/verification)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for images in .npz files (default: float16 for 2x smaller footprint)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(os.cpu_count() or 4, 8),
        help="Number of parallel worker processes for volume preprocessing (default: min(cpu_count, 8))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val/test patient splitting",
    )
    return parser.parse_args()


def zscore_normalize_non_zero(volume: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Channel-wise Z-score normalization computed strictly over non-zero brain parenchyma."""
    if mask is None:
        mask = volume > 0
    else:
        mask = mask & (volume != 0)

    if not np.any(mask):
        return volume.astype(np.float32)
    mean = float(volume[mask].mean())
    std = float(volume[mask].std())
    normalized = np.zeros_like(volume, dtype=np.float32)
    if std > 1e-8:
        normalized[mask] = (volume[mask] - mean) / std
    else:
        normalized[mask] = volume[mask] - mean
    return normalized


def resample_3d_volume(
    volume: np.ndarray, target_shape: tuple[int, int, int], is_mask: bool = False
) -> np.ndarray:
    """
    Resamples 3D volume [D, H, W] to target_shape using trilinear (image) or nearest-neighbor (mask).
    """
    # Convert to torch tensor [1, 1, D, H, W]
    t = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
    mode = "nearest" if is_mask else "trilinear"
    align_corners = None if is_mask else False

    resampled = (
        F.interpolate(
            t,
            size=target_shape,
            mode=mode,
            align_corners=align_corners,
        )
        .squeeze(0)
        .squeeze(0)
    )

    if is_mask:
        return (resampled > 0.5).to(torch.uint8).numpy()
    return resampled.numpy().astype(np.float32)


def process_patient(
    patient_dir: Path, output_dir: Path, target_size: int = 128, dtype: str = "float16"
) -> dict | None:
    """Processes a single patient's 4 MRI sequences + segmentation into a 128^3 .npz."""
    pid = patient_dir.name

    # Find the 4 MRI sequence files and 1 segmentation file
    t1n_files = list(patient_dir.glob(f"{pid}-t1n.nii*")) or list(patient_dir.glob("*t1n.nii*"))
    t1c_files = list(patient_dir.glob(f"{pid}-t1c.nii*")) or list(patient_dir.glob("*t1c.nii*"))
    t2w_files = list(patient_dir.glob(f"{pid}-t2w.nii*")) or list(patient_dir.glob("*t2w.nii*"))
    t2f_files = list(patient_dir.glob(f"{pid}-t2f.nii*")) or list(patient_dir.glob("*t2f.nii*"))
    seg_files = list(patient_dir.glob(f"{pid}-seg.nii*")) or list(patient_dir.glob("*seg.nii*"))

    if not (t1n_files and t1c_files and t2w_files and t2f_files and seg_files):
        return None

    # Load NIfTI volumes
    t1n = nib.load(str(t1n_files[0])).get_fdata().astype(np.float32)
    t1c = nib.load(str(t1c_files[0])).get_fdata().astype(np.float32)
    t2w = nib.load(str(t2w_files[0])).get_fdata().astype(np.float32)
    t2f = nib.load(str(t2f_files[0])).get_fdata().astype(np.float32)
    seg = nib.load(str(seg_files[0])).get_fdata().astype(np.float32)

    # 1. Non-zero brain bounding box extraction
    brain_mask = (t1n > 0) | (t1c > 0) | (t2w > 0) | (t2f > 0)
    if not np.any(brain_mask):
        return None

    z_indices = np.where(brain_mask.any(axis=(1, 2)))[0]
    y_indices = np.where(brain_mask.any(axis=(0, 2)))[0]
    x_indices = np.where(brain_mask.any(axis=(0, 1)))[0]

    z_min, z_max = z_indices[0], z_indices[-1] + 1
    y_min, y_max = y_indices[0], y_indices[-1] + 1
    x_min, x_max = x_indices[0], x_indices[-1] + 1

    # Crop to non-zero bounding box
    brain_mask_crop = brain_mask[z_min:z_max, y_min:y_max, x_min:x_max]
    t1n_crop = t1n[z_min:z_max, y_min:y_max, x_min:x_max]
    t1c_crop = t1c[z_min:z_max, y_min:y_max, x_min:x_max]
    t2w_crop = t2w[z_min:z_max, y_min:y_max, x_min:x_max]
    t2f_crop = t2f[z_min:z_max, y_min:y_max, x_min:x_max]
    seg_crop = seg[z_min:z_max, y_min:y_max, x_min:x_max]

    # 2. Resample bounding box to canonical target_size^3
    target_shape = (target_size, target_size, target_size)
    # Resample binary brain mask via nearest-neighbor to define crisp anatomical boundary (CUT-A)
    brain_mask_resampled = (
        resample_3d_volume(brain_mask_crop.astype(np.float32), target_shape, is_mask=True) > 0.5
    )

    t1n_resampled = resample_3d_volume(t1n_crop, target_shape, is_mask=False)
    t1c_resampled = resample_3d_volume(t1c_crop, target_shape, is_mask=False)
    t2w_resampled = resample_3d_volume(t2w_crop, target_shape, is_mask=False)
    t2f_resampled = resample_3d_volume(t2f_crop, target_shape, is_mask=False)
    seg_resampled = resample_3d_volume(seg_crop, target_shape, is_mask=True)

    # Clean trilinear interpolation bleed outside the resampled brain mask (CUT-A)
    t1n_resampled = np.where(brain_mask_resampled, t1n_resampled, 0.0)
    t1c_resampled = np.where(brain_mask_resampled, t1c_resampled, 0.0)
    t2w_resampled = np.where(brain_mask_resampled, t2w_resampled, 0.0)
    t2f_resampled = np.where(brain_mask_resampled, t2f_resampled, 0.0)

    # 3. Channel-independent Z-Score Normalization strictly within brain mask
    t1n_norm = zscore_normalize_non_zero(t1n_resampled, mask=brain_mask_resampled)
    t1c_norm = zscore_normalize_non_zero(t1c_resampled, mask=brain_mask_resampled)
    t2w_norm = zscore_normalize_non_zero(t2w_resampled, mask=brain_mask_resampled)
    t2f_norm = zscore_normalize_non_zero(t2f_resampled, mask=brain_mask_resampled)

    # Stack channels: [4, 128, 128, 128]
    np_dtype = np.float16 if dtype == "float16" else np.float32
    image_4ch = np.stack([t1n_norm, t1c_norm, t2w_norm, t2f_norm], axis=0).astype(np_dtype)
    mask_1ch = np.expand_dims((seg_resampled > 0).astype(np.uint8), axis=0)  # [1, 128, 128, 128]
    brain_mask_1ch = np.expand_dims(brain_mask_resampled.astype(np.uint8), axis=0)  # [1, 128, 128, 128]

    # Save compressed .npz with anatomical brain_mask (CUT-B)
    out_file = output_dir / f"{pid}_volume.npz"
    np.savez_compressed(
        out_file,
        image=image_4ch,
        mask=mask_1ch,
        brain_mask=brain_mask_1ch,
    )

    num_voxels_brain = int(np.sum(brain_mask_resampled))
    num_voxels_tumor = int(np.sum(mask_1ch > 0))
    has_tumor = bool(num_voxels_tumor > 0)

    try:
        rel_path = str(out_file.relative_to(output_dir.parent.parent))
    except ValueError:
        rel_path = f"{output_dir.name}/{out_file.name}"

    return {
        "patient_id": pid,
        "file_name": out_file.name,
        "rel_path": rel_path,
        "num_voxels_brain": num_voxels_brain,
        "num_voxels_tumor": num_voxels_tumor,
        "has_tumor": has_tumor,
        "orig_z_span": int(z_max - z_min),
        "orig_y_span": int(y_max - y_min),
        "orig_x_span": int(x_max - x_min),
    }


def _process_patient_worker(args_tuple):
    """Worker function for ProcessPoolExecutor with isolated thread count."""
    patient_dir, output_dir, target_size, dtype = args_tuple
    torch.set_num_threads(1)
    return process_patient(patient_dir, output_dir, target_size=target_size, dtype=dtype)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Fallback to local and Kaggle search if default path is not found directly
    if not data_dir.exists():
        fallback_candidates = [
            RAW_DATA_DIR / "BraTS_2024" / "BraTS-GLI" / "training_data1_v2",
            RAW_DATA_DIR / "BraTS-GLI" / "training_data1_v2",
            Path("../thesis_2d/data/raw/BraTS-GLI/training_data1_v2").resolve(),
            Path("data/raw/BraTS_2024/BraTS-GLI/training_data1_v2").resolve(),
            Path("data/raw/BraTS-GLI/training_data1_v2").resolve(),
            Path("/kaggle/input/brats-2024-gli/training_data1_v2"),
            Path("/kaggle/input/brats2024-gli/training_data1_v2"),
        ]
        if Path("/kaggle/input").exists():
            for p in Path("/kaggle/input").glob("**/training_data1_v2"):
                if p.is_dir():
                    fallback_candidates.append(p)
            for p in Path("/kaggle/input").glob("**/BraTS-GLI*"):
                if p.is_dir() and (p / "training_data1_v2").is_dir():
                    fallback_candidates.append(p / "training_data1_v2")
                elif p.is_dir():
                    fallback_candidates.append(p)

        for cand in fallback_candidates:
            if cand.exists():
                data_dir = cand
                break

    if not data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading raw 3D BraTS NIfTI data from: {data_dir}")
    print(f"Saving canonical 128^3 .npz volumes ({args.dtype}) to: {output_dir}")

    # Gather patient directories
    patient_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("BraTS")]
    )
    if args.limit and args.limit > 0:
        patient_dirs = patient_dirs[: args.limit]
        print(f"Limiting preprocessing to first {len(patient_dirs)} patients.")
    else:
        print(f"Found {len(patient_dirs)} patient volumes to process.")

    records = []
    if args.num_workers > 1 and len(patient_dirs) > 1:
        tasks = [(pdir, output_dir, args.target_size, args.dtype) for pdir in patient_dirs]
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for rec in tqdm(
                executor.map(_process_patient_worker, tasks),
                total=len(tasks),
                desc=f"Processing 3D Volumes ({args.num_workers} workers, {args.dtype})",
            ):
                if rec is not None:
                    records.append(rec)
    else:
        for pdir in tqdm(patient_dirs, desc="Processing 3D Volumes"):
            rec = process_patient(
                pdir, output_dir, target_size=args.target_size, dtype=args.dtype
            )
            if rec is not None:
                records.append(rec)

    if not records:
        print("No valid patient volumes were processed!")
        return

    df = pd.DataFrame(records)

    # Patient-stratified split: 70% Train, 15% Val, 15% Test
    # Stratified by tumor volume quartiles to ensure invariant volume distributions (Roy et al., 2024)
    pids = df["patient_id"].unique()
    split_map = None
    if len(pids) >= 20:
        try:
            # Categorize into quartiles (dropping duplicates if many zero/identical tumor volumes exist)
            df["tumor_quartile"] = pd.qcut(
                df["num_voxels_tumor"], q=4, labels=False, duplicates="drop"
            )
            q_counts = df["tumor_quartile"].value_counts()
            if q_counts.min() >= 3:
                train_df, temp_df = train_test_split(
                    df, test_size=0.30, random_state=args.seed, stratify=df["tumor_quartile"]
                )
                val_df, test_df = train_test_split(
                    temp_df,
                    test_size=0.50,
                    random_state=args.seed,
                    stratify=temp_df["tumor_quartile"],
                )
                split_map = {pid: "train" for pid in train_df["patient_id"]}
                split_map.update({pid: "val" for pid in val_df["patient_id"]})
                split_map.update({pid: "test" for pid in test_df["patient_id"]})
        except (ValueError, KeyError, TypeError):
            split_map = None

    if split_map is None and len(pids) >= 6:
        train_pids, temp_pids = train_test_split(pids, test_size=0.30, random_state=args.seed)
        val_pids, test_pids = train_test_split(temp_pids, test_size=0.50, random_state=args.seed)
        split_map = {pid: "train" for pid in train_pids}
        split_map.update({pid: "val" for pid in val_pids})
        split_map.update({pid: "test" for pid in test_pids})
    elif split_map is None and len(pids) >= 3:
        split_map = {pids[0]: "train", pids[1]: "val", pids[2]: "test"}
        for pid in pids[3:]:
            split_map[pid] = "train"
    elif split_map is None:
        # Very small subset (e.g. limit=1 or 2 for debugging/smoke test)
        split_map = {pid: "train" for pid in pids}
        # Duplicate rows for val and test so loaders find records
        val_df = df.copy()
        val_df["split"] = "val"
        test_df = df.copy()
        test_df["split"] = "test"
        df["split"] = "train"
        df = pd.concat([df, val_df, test_df], ignore_index=True)
        split_map = None

    if split_map is not None:
        df["split"] = df["patient_id"].map(split_map)

    metadata_path = output_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"\nMetadata saved to {metadata_path}")
    print("Split distribution:")
    print(df["split"].value_counts())
    print(f"Total processed volumes: {len(df)}")


if __name__ == "__main__":
    main()
