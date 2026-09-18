import re
from pathlib import Path
from typing import Any

import pandas as pd

from brats_jepa_3d.config import PROJECT_ROOT


def discover_experiment_directories(base_dir: Path | str | None = None) -> list[Path]:
    """Discovers all experiment subdirectories under base_dir that contain a 'metrics' or 'checkpoints' folder.

    If base_dir itself directly contains metrics/checkpoints (and no experiment subdirs), returns [base_dir].
    """
    root = Path(base_dir).resolve() if base_dir is not None else (PROJECT_ROOT / "outputs").resolve()
    if not root.exists():
        return []

    discovered = []
    # Check immediate subdirectories
    for item in sorted(root.iterdir()):
        if item.is_dir() and not item.name.startswith((".", "_")):
            if item.name in ["paper_artifacts", "figures"]:
                continue
            if (item / "metrics").exists() or (item / "checkpoints").exists() or (item / "logs").exists():
                discovered.append(item)

    if not discovered and ((root / "metrics").exists() or (root / "checkpoints").exists()):
        discovered.append(root)

    return discovered


def parse_dice_value(val: Any) -> float:
    """Extracts numeric mean Dice from strings like '69.62 ± 16.34' or '82.52%'."""
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip()
    match = re.search(r"(\d+(?:\.\d+)?)", val_str)
    return float(match.group(1)) if match else 0.0


def aggregate_master_benchmarks(
    experiment_dirs: list[Path],
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Combines individual benchmark_3d_summary.csv or master_3d_benchmark.csv across runs.
    Returns a unified, sorted master benchmark DataFrame.
    """
    dfs: list[pd.DataFrame] = []

    for exp_dir in experiment_dirs:
        candidates = [
            exp_dir / "metrics" / "master_3d_benchmark.csv",
            exp_dir / "metrics" / "benchmark_3d_summary.csv",
            exp_dir / "master_3d_benchmark.csv",
            exp_dir / "benchmark_3d_summary.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path:
            try:
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    if "Model" in df.columns:
                        if "Model Architecture" in df.columns:
                            df["Model Architecture"] = df["Model Architecture"].fillna(df["Model"])
                            df = df.drop(columns=["Model"])
                        else:
                            df = df.rename(columns={"Model": "Model Architecture"})
                    dfs.append(df)
            except Exception as e:
                print(f"⚠️ Warning: Could not read {csv_path}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    if "Model" in combined.columns and "Model Architecture" in combined.columns:
        combined["Model Architecture"] = combined["Model Architecture"].fillna(combined["Model"])
        combined = combined.drop(columns=["Model"])

    model_col = "Model Architecture" if "Model Architecture" in combined.columns else combined.columns[0]
    dice_col = next((c for c in combined.columns if "dice" in c.lower()), None)

    # Deduplicate keeping the best (or latest) record per model
    if dice_col:
        combined["_sort_dice"] = combined[dice_col].apply(parse_dice_value)
        combined = combined.sort_values(by="_sort_dice", ascending=False)
        combined = combined.drop_duplicates(subset=[model_col], keep="first")
        combined = combined.drop(columns=["_sort_dice"])
    else:
        combined = combined.drop_duplicates(subset=[model_col], keep="last")

    combined = combined.reset_index(drop=True)

    # Save to output_dir if provided
    if output_dir:
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path / "master_3d_benchmark.csv", index=False)
        with open(out_path / "master_3d_benchmark.md", "w", encoding="utf-8") as f:
            f.write("# 3D Volumetric BraTS 2024 GLI Master Benchmark Results\n\n")
            f.write(combined.to_markdown(index=False))
            f.write("\n")

    return combined


def aggregate_low_data_summaries(
    experiment_dirs: list[Path],
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Merges 'low_data_3d_summary.csv' from each experiment into a single table with

    columns: ['Fraction', '<Model 1>', '<Model 2>', ...].
    """
    merged_df: pd.DataFrame | None = None

    for exp_dir in experiment_dirs:
        candidates = [
            exp_dir / "metrics" / "low_data_3d_summary.csv",
            exp_dir / "low_data_3d_summary.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if not csv_path:
            continue

        try:
            df = pd.read_csv(csv_path)
            if "Fraction" not in df.columns or len(df) == 0:
                continue

            # Standardize Fraction string format (e.g. '1.0%')
            df["Fraction"] = df["Fraction"].astype(str).str.strip()

            if merged_df is None:
                merged_df = df
            else:
                new_cols = [c for c in df.columns if c not in merged_df.columns]
                overlap_cols = [c for c in df.columns if c != "Fraction" and c in merged_df.columns]
                if new_cols:
                    merged_df = pd.merge(merged_df, df[["Fraction"] + new_cols], on="Fraction", how="outer")
                else:
                    merged_df = pd.merge(merged_df, df[["Fraction"]], on="Fraction", how="outer")
                if overlap_cols:
                    df_indexed = df.set_index("Fraction")
                    for col in overlap_cols:
                        for frac, val in df_indexed[col].items():
                            if pd.notna(val) and str(val) != "N/A":
                                merged_df.loc[merged_df["Fraction"] == frac, col] = val
        except Exception as e:
            print(f"⚠️ Warning: Could not read {csv_path}: {e}")

    if merged_df is None:
        return pd.DataFrame()

    # Sort by fraction numerically if possible
    def frac_key(val: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)", val)
        return float(m.group(1)) if m else 0.0

    merged_df["_sort_frac"] = merged_df["Fraction"].apply(frac_key)
    merged_df = merged_df.sort_values(by="_sort_frac").drop(columns=["_sort_frac"]).reset_index(drop=True)
    merged_df = merged_df.fillna("N/A")

    if output_dir:
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(out_path / "low_data_3d_summary.csv", index=False)
        with open(out_path / "low_data_3d_summary.md", "w", encoding="utf-8") as f:
            f.write("# 3D Low-Data Label Efficiency Benchmark Results\n\n")
            f.write(merged_df.to_markdown(index=False))
            f.write("\n")

    return merged_df


def aggregate_ood_summaries(
    experiment_dirs: list[Path],
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Merges 'ood_3d_summary.csv' from each experiment directory by 'Regime'."""
    merged_df: pd.DataFrame | None = None

    for exp_dir in experiment_dirs:
        candidates = [
            exp_dir / "metrics" / "ood_3d_summary.csv",
            exp_dir / "ood_3d_summary.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if not csv_path:
            continue

        try:
            df = pd.read_csv(csv_path)
            if "Regime" not in df.columns or len(df) == 0:
                continue

            df["Regime"] = df["Regime"].astype(str).str.strip()

            if merged_df is None:
                merged_df = df
            else:
                new_cols = [c for c in df.columns if c not in merged_df.columns]
                overlap_cols = [c for c in df.columns if c != "Regime" and c in merged_df.columns]
                if new_cols:
                    merged_df = pd.merge(merged_df, df[["Regime"] + new_cols], on="Regime", how="outer")
                else:
                    merged_df = pd.merge(merged_df, df[["Regime"]], on="Regime", how="outer")
                if overlap_cols:
                    df_indexed = df.set_index("Regime")
                    for col in overlap_cols:
                        for r_name, val in df_indexed[col].items():
                            if pd.notna(val) and str(val) != "N/A":
                                merged_df.loc[merged_df["Regime"] == r_name, col] = val
        except Exception as e:
            print(f"⚠️ Warning: Could not read {csv_path}: {e}")

    if merged_df is None:
        return pd.DataFrame()

    merged_df = merged_df.fillna("N/A")

    if output_dir:
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(out_path / "ood_3d_summary.csv", index=False)
        with open(out_path / "ood_3d_summary.md", "w", encoding="utf-8") as f:
            f.write("# 3D Out-of-Distribution (OOD) Scanner Shift Results\n\n")
            f.write(merged_df.to_markdown(index=False))
            f.write("\n")

    return merged_df


def export_latex_tables(
    master_df: pd.DataFrame | None = None,
    low_data_df: pd.DataFrame | None = None,
    ood_df: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
) -> str:
    """Generates LaTeX table code snippets matching the paper format."""
    sections = []

    if master_df is not None and not master_df.empty:
        sections.append("% --- Table 1: Master 3D Volumetric Benchmark ---")
        sections.append("\\begin{table}[t]")
        sections.append("\\centering")
        sections.append("\\caption{\\textbf{Master 3D Volumetric Brain Glioma Segmentation Benchmark (BraTS 2024 GLI held-out test split, $N=271$).}}")
        sections.append("\\label{tab:master_benchmark}")
        sections.append("\\resizebox{\\textwidth}{!}{%")
        col_spec = "l" + "c" * (len(master_df.columns) - 1)
        sections.append(f"\\begin{{tabular}}{{{col_spec}}}")
        sections.append("\\toprule")
        header_cols = [f"\\textbf{{{c}}}" for c in master_df.columns]
        sections.append(" & ".join(header_cols) + " \\\\")
        sections.append("\\midrule")
        for _, row in master_df.iterrows():
            row_str = " & ".join(str(val) for val in row.values) + " \\\\"
            sections.append(row_str)
        sections.append("\\bottomrule")
        sections.append("\\end{tabular}%")
        sections.append("}")
        sections.append("\\end{table}\n")

    if low_data_df is not None and not low_data_df.empty:
        sections.append("% --- Table 2: Low-Data Label Efficiency Benchmark ---")
        sections.append("\\begin{table}[t]")
        sections.append("\\centering")
        sections.append("\\caption{\\textbf{Low-Data Volumetric Label Efficiency across Annotation Budgets (Test Dice \\%).}}")
        sections.append("\\label{tab:low_data}")
        col_spec = "l" + "c" * (len(low_data_df.columns) - 1)
        sections.append(f"\\begin{{tabular}}{{{col_spec}}}")
        sections.append("\\toprule")
        header_cols = [f"\\textbf{{{c}}}" for c in low_data_df.columns]
        sections.append(" & ".join(header_cols) + " \\\\")
        sections.append("\\midrule")
        for _, row in low_data_df.iterrows():
            row_str = " & ".join(str(val) for val in row.values) + " \\\\"
            sections.append(row_str)
        sections.append("\\bottomrule")
        sections.append("\\end{tabular}")
        sections.append("\\end{table}\n")

    if ood_df is not None and not ood_df.empty:
        sections.append("% --- Table 3: Out-of-Distribution and Missing Modality Robustness ---")
        sections.append("\\begin{table}[t]")
        sections.append("\\centering")
        sections.append("\\caption{\\textbf{Out-of-Distribution Scanner Shift and Missing Modality Robustness (Test Dice \\%).}}")
        sections.append("\\label{tab:ood_robustness}")
        sections.append("\\resizebox{\\textwidth}{!}{%")
        col_spec = "l" + "c" * (len(ood_df.columns) - 1)
        sections.append(f"\\begin{{tabular}}{{{col_spec}}}")
        sections.append("\\toprule")
        header_cols = [f"\\textbf{{{c}}}" for c in ood_df.columns]
        sections.append(" & ".join(header_cols) + " \\\\")
        sections.append("\\midrule")
        for _, row in ood_df.iterrows():
            row_str = " & ".join(str(val) for val in row.values) + " \\\\"
            sections.append(row_str)
        sections.append("\\bottomrule")
        sections.append("\\end{tabular}%")
        sections.append("}")
        sections.append("\\end{table}\n")

    latex_content = "\n".join(sections)

    if output_path:
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(latex_content)

    return latex_content

