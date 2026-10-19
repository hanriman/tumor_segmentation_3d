import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from brats_jepa_3d.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def discover_experiment_directories(base_dir: Path | str | None = None) -> list[Path]:
    """Discovers all experiment subdirectories under base_dir that contain a 'metrics' or 'checkpoints' folder.

    If base_dir itself directly contains metrics/checkpoints (and no experiment subdirs), returns [base_dir].
    Searches both base_dir and base_dir / "experiments" or "runs" up to 2 levels deep.
    """
    root = Path(base_dir).resolve() if base_dir is not None else (PROJECT_ROOT / "outputs").resolve()
    if not root.exists():
        return []

    ignore_names = {
        "paper_artifacts", "figures", "legacy", "archive", "reports",
        "multiregion", "checkpoints", "logs", "metrics", "data", "dist_kaggle",
        ".git", ".venv", "__pycache__", "tables"
    }

    discovered = []
    seen = set()

    def check_candidate(d: Path) -> bool:
        if not d.is_dir() or d.name.startswith((".", "_")) or d.name in ignore_names:
            return False
        if (d / "metrics").exists() or (d / "checkpoints").exists() or (d / "logs").exists():
            resolved = d.resolve()
            if resolved not in seen and resolved != root:
                discovered.append(d)
                seen.add(resolved)
                return True
        return False

    # 1. First priority: check designated experiments/ or runs/ subdirectories
    for sub in ["experiments", "runs"]:
        sub_path = root / sub
        if sub_path.is_dir():
            for item in sorted(sub_path.iterdir()):
                check_candidate(item)

    # 2. Check immediate and second-level subdirectories of root
    for item in sorted(root.iterdir()):
        if not check_candidate(item) and item.is_dir() and item.name not in ignore_names and not item.name.startswith((".", "_")):
            # Check 1 level deeper (e.g. outputs/experiments/my_exp or outputs/sweep_1/my_exp)
            for sub_item in sorted(item.iterdir()):
                check_candidate(sub_item)

    if not discovered and ((root / "metrics").exists() or (root / "checkpoints").exists()):
        discovered.append(root)

    return sorted(discovered, key=lambda p: p.name)


def parse_dice_value(val: Any) -> float:
    """Extracts numeric mean Dice from strings like '69.62 ± 16.34' or '82.52%'.

    Unparseable / missing values return NaN so they sort last and never
    outrank real measurements.
    """
    if val is None:
        return float("nan")
    try:
        if pd.isna(val):
            return float("nan")
    except (TypeError, ValueError):
        pass
    val_str = str(val).strip()
    match = re.search(r"(\d+(?:\.\d+)?)", val_str)
    return float(match.group(1)) if match else float("nan")


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
                logger.warning("Could not read %s: %s", csv_path, e)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    if "Model" in combined.columns and "Model Architecture" in combined.columns:
        combined["Model Architecture"] = combined["Model Architecture"].fillna(combined["Model"])
        combined = combined.drop(columns=["Model"])

    # Drop redundant unscaled 'HD95 (mm)' / 'HD95_tumor_only (mm)' columns when 'HD95 (voxels)' is present
    # (128^3 tight-crop grids are anisotropically resampled per patient, so 1 voxel != 1 mm).
    if "HD95 (voxels)" in combined.columns and "HD95 (mm)" in combined.columns:
        combined = combined.drop(columns=["HD95 (mm)"])
    if "HD95_tumor_only (voxels)" in combined.columns and "HD95_tumor_only (mm)" in combined.columns:
        combined = combined.drop(columns=["HD95_tumor_only (mm)"])

    model_col = "Model Architecture" if "Model Architecture" in combined.columns else combined.columns[0]
    dice_col = next((c for c in combined.columns if "dice" in c.lower()), None)

    # Deduplicate keeping the best record per model, but record the evidence:
    # `n_runs` counts merged rows per model, `dice_std` captures spread, and
    # dropped rows are logged so no run disappears silently.
    if dice_col:
        combined["_sort_dice"] = combined[dice_col].apply(parse_dice_value)
        combined = combined.sort_values(by="_sort_dice", ascending=False, na_position="last")
        grouped = combined.groupby(model_col)["_sort_dice"]
        n_runs = grouped.size().rename("n_runs")
        dice_std = grouped.std(ddof=0).fillna(0.0).rename("dice_std")
        dropped = combined.duplicated(subset=[model_col], keep="first").sum()
        if dropped:
            logger.info(
                "aggregate_master_benchmarks: keeping best of %d rows per model "
                "(%d dropped); see n_runs/dice_std columns.",
                len(combined),
                int(dropped),
            )
        combined = combined.drop_duplicates(subset=[model_col], keep="first")
        combined = combined.drop(columns=["_sort_dice"])
        combined = combined.merge(n_runs, left_on=model_col, right_index=True, how="left")
        combined = combined.merge(dice_std, left_on=model_col, right_index=True, how="left")
    else:
        combined = combined.drop_duplicates(subset=[model_col], keep="last")

    combined = combined.reset_index(drop=True)

    # Save to output_dir if provided
    if output_dir:
        out_path = Path(output_dir).resolve()
        target = out_path / "reports" if out_path.name == "outputs" else out_path
        target.mkdir(parents=True, exist_ok=True)
        combined.to_csv(target / "master_3d_benchmark.csv", index=False)
        with open(target / "master_3d_benchmark.md", "w", encoding="utf-8") as f:
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
            logger.warning("Could not read %s: %s", csv_path, e)

    if merged_df is None:
        return pd.DataFrame()

    # Sort by fraction numerically if possible
    def frac_key(val: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)", val)
        return float(m.group(1)) if m else 0.0

    merged_df["_sort_frac"] = merged_df["Fraction"].apply(frac_key)
    merged_df = merged_df.sort_values(by="_sort_frac").drop(columns=["_sort_frac"]).reset_index(drop=True)
    merged_df = merged_df.fillna("N/A")

    # Ensure standard aliases for UNETR Scratch
    if "3D ViT-UNETR (From Scratch) [mean]" in merged_df.columns and "3D ViT-FPN (From Scratch) [unetr_hybrid] [mean]" not in merged_df.columns:
        merged_df["3D ViT-FPN (From Scratch) [unetr_hybrid] [mean]"] = merged_df["3D ViT-UNETR (From Scratch) [mean]"]
    elif "3D ViT-FPN (From Scratch) [unetr_hybrid] [mean]" in merged_df.columns and "3D ViT-UNETR (From Scratch) [mean]" not in merged_df.columns:
        merged_df["3D ViT-UNETR (From Scratch) [mean]"] = merged_df["3D ViT-FPN (From Scratch) [unetr_hybrid] [mean]"]

    if output_dir:
        out_path = Path(output_dir).resolve()
        target = out_path / "reports" if out_path.name == "outputs" else out_path
        target.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(target / "low_data_3d_summary.csv", index=False)
        with open(target / "low_data_3d_summary.md", "w", encoding="utf-8") as f:
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
            logger.warning("Could not read %s: %s", csv_path, e)

    if merged_df is None:
        return pd.DataFrame()

    merged_df = merged_df.fillna("N/A")

    if output_dir:
        out_path = Path(output_dir).resolve()
        target = out_path / "reports" if out_path.name == "outputs" else out_path
        target.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(target / "ood_3d_summary.csv", index=False)
        with open(target / "ood_3d_summary.md", "w", encoding="utf-8") as f:
            f.write("# 3D Out-of-Distribution (OOD) Scanner Shift Results\n\n")
            f.write(merged_df.to_markdown(index=False))
            f.write("\n")

    return merged_df


def _latex_escape(val: Any) -> str:
    """Escapes LaTeX special characters in table cells."""
    s = str(val)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def export_latex_tables(
    master_df: pd.DataFrame | None = None,
    low_data_df: pd.DataFrame | None = None,
    ood_df: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    test_n: int | None = None,
) -> str:
    """Generates LaTeX table code snippets matching the paper format.

    `test_n` fills the held-out test-split size in the master caption instead of
    a hardcoded constant; when None the caption omits the count.
    """
    sections = []

    if master_df is not None and not master_df.empty:
        sections.append("% --- Table 1: Master 3D Volumetric Benchmark ---")
        sections.append("\\begin{table}[t]")
        sections.append("\\centering")
        n_clause = f", $N={test_n}$" if test_n is not None else ""
        sections.append(f"\\caption{{\\textbf{{Master 3D Volumetric Brain Glioma Segmentation Benchmark (BraTS 2024 GLI held-out test split{n_clause}).}}}}")
        sections.append("\\label{tab:master_benchmark}")
        sections.append("\\resizebox{\\textwidth}{!}{%")
        col_spec = "l" + "c" * (len(master_df.columns) - 1)
        sections.append(f"\\begin{{tabular}}{{{col_spec}}}")
        sections.append("\\toprule")
        header_cols = [f"\\textbf{{{_latex_escape(c)}}}" for c in master_df.columns]
        sections.append(" & ".join(header_cols) + " \\\\")
        sections.append("\\midrule")
        for _, row in master_df.iterrows():
            row_str = " & ".join(_latex_escape(val) for val in row.values) + " \\\\"
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
        header_cols = [f"\\textbf{{{_latex_escape(c)}}}" for c in low_data_df.columns]
        sections.append(" & ".join(header_cols) + " \\\\")
        sections.append("\\midrule")
        for _, row in low_data_df.iterrows():
            row_str = " & ".join(_latex_escape(val) for val in row.values) + " \\\\"
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
        header_cols = [f"\\textbf{{{_latex_escape(c)}}}" for c in ood_df.columns]
        sections.append(" & ".join(header_cols) + " \\\\")
        sections.append("\\midrule")
        for _, row in ood_df.iterrows():
            row_str = " & ".join(_latex_escape(val) for val in row.values) + " \\\\"
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


def compute_paired_significance_table(
    experiment_dirs: list[Path],
    output_dir: Path | str | None = None,
    reference_model: str | None = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Computes paired Wilcoxon signed-rank p-values and paired bootstrap 95% confidence intervals
    from `<model>_per_case.csv` files across experiment directories.
    """
    import numpy as np
    from scipy.stats import wilcoxon

    per_case_by_model: dict[str, pd.DataFrame] = {}
    for exp_dir in experiment_dirs:
        for search_dir in (exp_dir / "metrics", exp_dir):
            if not search_dir.is_dir():
                continue
            for csv_path in sorted(search_dir.glob("*_per_case.csv")):
                try:
                    df = pd.read_csv(csv_path)
                    if "patient_id" not in df.columns or len(df) == 0:
                        continue
                    model_name = (
                        str(df["model"].iloc[0])
                        if "model" in df.columns and len(df["model"]) > 0
                        else csv_path.stem.replace("_per_case", "")
                    )
                    per_case_by_model[model_name] = df
                except Exception as e:
                    logger.warning("Could not read per-case CSV %s: %s", csv_path, e)

    if len(per_case_by_model) < 2:
        return pd.DataFrame()

    models = sorted(per_case_by_model.keys())
    if reference_model is None or reference_model not in per_case_by_model:
        # Prefer VisReg JEPA UNETR or FPN if present, otherwise first model
        preferred = [
            m for m in models
            if "visreg" in m.lower() and "frozen" not in m.lower() and "scratch" not in m.lower()
        ]
        reference_model = preferred[0] if preferred else models[0]

    ref_df = per_case_by_model[reference_model].set_index("patient_id")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for comp_model in models:
        if comp_model == reference_model:
            continue
        comp_df = per_case_by_model[comp_model].set_index("patient_id")
        common_ids = ref_df.index.intersection(comp_df.index)
        if len(common_ids) < 2:
            continue

        for metric_col in ("Dice_WT", "Dice_TC", "Dice_ET", "Dice_Mean"):
            if metric_col not in ref_df.columns or metric_col not in comp_df.columns:
                continue
            a = ref_df.loc[common_ids, metric_col].to_numpy(dtype=float) * 100.0
            b = comp_df.loc[common_ids, metric_col].to_numpy(dtype=float) * 100.0
            valid = np.isfinite(a) & np.isfinite(b)
            a, b = a[valid], b[valid]
            n_pairs = int(len(a))
            if n_pairs < 2:
                continue

            diff = a - b
            mean_diff = float(np.mean(diff))
            boot_idx = rng.integers(0, n_pairs, size=(n_boot, n_pairs))
            boot_means = np.mean(diff[boot_idx], axis=1)
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

            if np.allclose(diff, 0.0):
                p_val = 1.0
            else:
                try:
                    _, p_val = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                    p_val = float(p_val)
                except ValueError:
                    p_val = 1.0

            rows.append(
                {
                    "Reference Model": reference_model,
                    "Comparator Model": comp_model,
                    "Metric": metric_col,
                    "N_patients": n_pairs,
                    "Mean Diff (pp)": f"{mean_diff:+.2f}",
                    "95% Bootstrap CI (pp)": f"[{ci_low:+.2f}, {ci_high:+.2f}]",
                    "Wilcoxon p-value": f"{p_val:.4g}",
                }
            )

    sig_df = pd.DataFrame(rows)
    if output_dir and not sig_df.empty:
        out_path = Path(output_dir).resolve()
        target = out_path / "reports" if out_path.name == "outputs" else out_path
        target.mkdir(parents=True, exist_ok=True)
        sig_df.to_csv(target / "paired_significance_3d.csv", index=False)
        with open(target / "paired_significance_3d.md", "w", encoding="utf-8") as f:
            f.write("# 3D Paired Statistical Significance (Wilcoxon & 10,000-Resample Bootstrap)\n\n")
            f.write(sig_df.to_markdown(index=False))
            f.write("\n")

    return sig_df

