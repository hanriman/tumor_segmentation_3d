#!/usr/bin/env python3
"""Merge staged 3D .npz pools into one dataset with grouped fresh split.

- Moves *.npz from each --part_dir into --output_dir.
- Concatenates metadata, groups rows by base patient ID (strip trailing -100/-101/-102),
  splits GROUPS 70/15/15 (seed 42, stratified by group max tumor_quartile), assigns scans.
- Recomputes rel_path, writes merged metadata.csv.

Usage:
    python scripts/merge_and_resplit_3d.py \
        --part_dir data/processed/brats_gli_3d_full_partA \
        --part_dir data/processed/brats_gli_3d_full_partB \
        --output_dir data/processed/brats_gli_3d_full [--seed 42]
"""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_ID_RE = re.compile(r"^(.*)-\d+$")


def base_id(pid: str) -> str:
    m = BASE_ID_RE.match(pid)
    return m.group(1) if m else pid


def parse_args():
    p = argparse.ArgumentParser(description="Merge staged 3D pools with grouped split")
    p.add_argument("--part_dir", action="append", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    n_files = 0
    for part in args.part_dir:
        part = Path(part).resolve()
        if not part.is_dir():
            raise FileNotFoundError(f"Staged part directory not found: {part}")
        for npz in sorted(part.glob("*_volume.npz")):
            shutil.move(str(npz), out / npz.name)
            n_files += 1
        meta = part / "metadata.csv"
        if meta.exists():
            frames.append(pd.read_csv(meta))
    if n_files == 0:
        raise FileNotFoundError(
            f"No *_volume.npz files found in {args.part_dir} "
            "(already merged? part dirs are drained by shutil.move)."
        )
    if not frames:
        raise FileNotFoundError(f"No metadata.csv found in {args.part_dir}.")
    print(f"Moved {n_files} .npz files -> {out}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="file_name")
    print(f"Merged metadata rows: {len(df)}")

    df["base_id"] = df["patient_id"].apply(base_id)
    print(f"Unique base patients: {df['base_id'].nunique()} "
          f"({df['base_id'].value_counts().to_dict() and 'multi-timepoint present' if (df['base_id'].value_counts() > 1).any() else 'all single'})")

    # Group-level quartile from max tumor volume (worst timepoint represents patient)
    g = df.groupby("base_id").agg(
        num_voxels_tumor=("num_voxels_tumor", "max"),
        n_scans=("patient_id", "count"),
    ).reset_index()
    try:
        g["tumor_quartile"] = pd.qcut(g["num_voxels_tumor"], q=4, labels=False, duplicates="drop")
    except (ValueError, TypeError):
        g["tumor_quartile"] = 0
    strat = g["tumor_quartile"] if g["tumor_quartile"].nunique() > 1 and g["tumor_quartile"].value_counts().min() >= 6 else None

    train_g, temp_g = train_test_split(g, test_size=0.30, random_state=args.seed, stratify=strat)
    strat_t = temp_g["tumor_quartile"] if strat is not None and temp_g["tumor_quartile"].nunique() > 1 and temp_g["tumor_quartile"].value_counts().min() >= 3 else None
    val_g, test_g = train_test_split(temp_g, test_size=0.50, random_state=args.seed, stratify=strat_t)
    split_map = {r.base_id: "train" for r in train_g.itertuples()}
    split_map.update({r.base_id: "val" for r in val_g.itertuples()})
    split_map.update({r.base_id: "test" for r in test_g.itertuples()})
    df["split"] = df["base_id"].map(split_map)

    # Per-scan quartile column for downstream stratified subsampling (dataset.py needs it)
    try:
        df["tumor_quartile"] = pd.qcut(df["num_voxels_tumor"], q=4, labels=False, duplicates="drop")
    except (ValueError, TypeError):
        df["tumor_quartile"] = 0

    df["rel_path"] = df["file_name"].apply(lambda f: f"{out.name}/{f}")
    df = df.drop(columns=["base_id"])
    df.to_csv(out / "metadata.csv", index=False)

    # Leakage audit: no base ID may span splits
    check = df.copy()
    check["base_id"] = check["patient_id"].apply(base_id)
    leaked = check.groupby("base_id")["split"].nunique()
    leaked = leaked[leaked > 1]
    print("Split distribution:\n", df["split"].value_counts().to_string())
    print(f"Base IDs spanning >1 split (must be 0): {len(leaked)}")
    assert len(leaked) == 0, f"LEAKAGE: {leaked.index.tolist()[:5]}"
    assert (out / "metadata.csv").exists()
    print(f"Merged metadata saved ({len(df)} rows). Done.")


if __name__ == "__main__":
    main()
