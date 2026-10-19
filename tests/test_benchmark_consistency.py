"""
test_benchmark_consistency.py
-----------------------------
Validates consistency between empirical benchmark evaluation files (CSV),
the LaTeX paper manuscript (paper/latex/main.tex), and documentation.

Ensures no protocol drift, transcription errors, or claim mismatches occur.
"""

from pathlib import Path
import re
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "outputs" / "multiregion" / "low_data_3d_summary.csv"
MASTER_CSV_PATH = REPO_ROOT / "outputs" / "multiregion" / "master_3d_benchmark.csv"
OOD_CSV_PATH = REPO_ROOT / "outputs" / "multiregion" / "ood_3d_summary.csv"
TEX_PATH = REPO_ROOT / "paper" / "latex" / "main.tex"
DOCS_PATH = REPO_ROOT / "docs" / "03_scientific_rationales_and_benchmarks.md"


def test_low_data_csv_exists_and_valid():
    """Verify low_data_3d_summary.csv exists and contains all 6 fractions and models."""
    assert CSV_PATH.exists(), f"Benchmark summary not found at {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    assert len(df) == 6, f"Expected 6 fractions, found {len(df)}"

    expected_fractions = ["1.0%", "5.0%", "10.0%", "25.0%", "50.0%", "100.0%"]
    actual_fractions = df["Fraction"].astype(str).str.strip().tolist()
    assert actual_fractions == expected_fractions

    # Ensure critical mean columns exist
    critical_cols = [
        "3D nnU-Net [mean]",
        "3D UNet [mean]",
        "3D VisReg JEPA (FPN) [mean]",
        "3D VisReg JEPA (FPN) [unetr_hybrid] [mean]",
        "3D ViT-FPN (From Scratch) [mean]",
        "3D ViT-FPN (From Scratch) [unetr_hybrid] [mean]",
    ]
    for col in critical_cols:
        assert col in df.columns, f"Missing critical column {col}"


def test_latex_table2_matches_csv():
    """Verify Table 2 in paper/latex/main.tex matches low_data_3d_summary.csv exactly."""
    assert TEX_PATH.exists(), f"LaTeX file not found at {TEX_PATH}"
    df = pd.read_csv(CSV_PATH).set_index("Fraction")

    tex_content = TEX_PATH.read_text(encoding="utf-8")

    # Isolate Table 2 block (tab:low_data_results)
    table_match = re.search(
        r"\\label\{tab:low_data_results\}(.*?)\\end\{table\}",
        tex_content,
        re.DOTALL,
    )
    assert table_match is not None, "Could not locate Table 2 block (tab:low_data_results) in LaTeX"
    table_text = table_match.group(1)

    # Map LaTeX table model rows to CSV column names
    model_mapping = {
        "3D nnU-Net Baseline (DynUNet)": "3D nnU-Net [mean]",
        "3D Residual UNet Baseline": "3D UNet [mean]",
        "3D ViT-UNETR Hybrid (Scratch)": "3D ViT-FPN (From Scratch) [unetr_hybrid] [mean]",
        "3D VisReg JEPA (UNETR Hybrid)": "3D VisReg JEPA (FPN) [unetr_hybrid] [mean]",
        "3D ViT-FPN (From Scratch)": "3D ViT-FPN (From Scratch) [mean]",
        "3D VisReg JEPA (Multiscale FPN)": "3D VisReg JEPA (FPN) [mean]",
    }

    fractions = ["1.0%", "5.0%", "10.0%", "25.0%", "50.0%", "100.0%"]

    for tex_row_name, csv_col in model_mapping.items():
        pattern = rf"^[ \t]*{re.escape(tex_row_name)}\s*&(.*?)\\\\\s*$"
        match = re.search(pattern, table_text, re.MULTILINE)
        assert match is not None, f"Could not find row '{tex_row_name}' in Table 2"

        row_cells = match.group(1).split("&")
        assert len(row_cells) == 6, f"Expected 6 values in row '{tex_row_name}', got {len(row_cells)}"

        for frac, cell in zip(fractions, row_cells):
            num_match = re.search(r"(\d+\.\d+)", cell)
            assert num_match is not None, f"Could not parse numeric score from cell '{cell}'"
            tex_val = float(num_match.group(1))

            csv_cell = df.loc[frac, csv_col]
            csv_val = float(str(csv_cell).replace("%", "").strip())

            assert abs(tex_val - csv_val) < 0.05, (
                f"Mismatch for '{tex_row_name}' at {frac}: LaTeX={tex_val}%, CSV={csv_val}%"
            )


def test_docs03_matches_benchmark():
    """Verify docs/03_scientific_rationales_and_benchmarks.md does not report stale scratch scores."""
    assert DOCS_PATH.exists(), f"Docs file not found at {DOCS_PATH}"
    docs_content = DOCS_PATH.read_text(encoding="utf-8")

    # Assert that old conflated numbers (6.88% scratch, 5.12% scratch, +25.21pp) are absent
    assert "6.88% *(Collapse)*" not in docs_content, "Old 6.88% scratch score still present in docs/03"
    assert "5.12% *(Collapse)*" not in docs_content, "Old 5.12% scratch score still present in docs/03"
    assert "+25.21" not in docs_content, "Old +25.21pp delta claim still present in docs/03"

    # Assert correct canonical numbers are present
    assert "63.65%" in docs_content, "VisReg UNETR 5% score 63.65% missing from docs/03"
    assert "56.57%" in docs_content, "ViT-UNETR Scratch 5% score 56.57% missing from docs/03"
    assert "+7.08" in docs_content, "+7.08pp JEPA gain missing from docs/03"
    assert "75.13%" in docs_content, "Frozen UNETR score 75.13% missing from docs/03"
    assert "79.50%" in docs_content, "3D nnU-Net score 79.50% missing from docs/03"


def test_low_data_volume_counts_consistent():
    """Verify that paper Table 2 and abstract report ceil-based volumes (12, 58, 115)."""
    assert TEX_PATH.exists()
    tex_content = TEX_PATH.read_text(encoding="utf-8")
    assert r"1\% (12 Vols)" in tex_content, "Table 2 header should report '1\\% (12 Vols)'"
    assert r"5\% (58 Vols)" in tex_content, "Table 2 header should report '5\\% (58 Vols)'"
    assert r"10\% (115 Vols)" in tex_content, "Table 2 header should report '10\\% (115 Vols)'"
    assert r"($1\%$ labels, $12$ volumes)" in tex_content, "Abstract should report 12 volumes"


def test_latex_table1_matches_master_csv():
    """Verify Table 1 in paper/latex/main.tex matches master_3d_benchmark.csv."""
    assert MASTER_CSV_PATH.exists()
    assert TEX_PATH.exists()

    df = pd.read_csv(MASTER_CSV_PATH)
    df_indexed = df.set_index("Model Architecture")
    tex_content = TEX_PATH.read_text(encoding="utf-8")

    table_match = re.search(
        r"\\label\{tab:benchmark_results\}(.*?)\\end\{table\}",
        tex_content,
        re.DOTALL,
    )
    assert table_match is not None, "Could not locate Table 1 block in LaTeX"
    table_text = table_match.group(1)

    models = [
        ("3D nnU-Net (DynUNet)", "3D nnU-Net (DynUNet)"),
        ("3D Residual UNet", "3D Residual UNet"),
        ("3D ViT-UNETR Hybrid (Scratch)", "3D ViT-FPN (From Scratch) [unetr_hybrid]"),
        ("3D VisReg JEPA (UNETR Hybrid)", "3D VisReg JEPA (FPN) [unetr_hybrid]"),
        ("3D ViT-FPN (From Scratch)", "3D ViT-FPN (From Scratch)"),
        ("3D VisReg JEPA (Multiscale FPN)", "3D VisReg JEPA (FPN)"),
    ]

    for tex_name, csv_prefix in models:
        pattern = rf"^[ \t]*{re.escape(tex_name)}\s*&(.*?)\\\\\s*$"
        match = re.search(pattern, table_text, re.MULTILINE)
        assert match is not None, f"Could not find row '{tex_name}' in Table 1"
        row_cells = match.group(1).split("&")
        assert len(row_cells) == 8, f"Expected 8 cells in row '{tex_name}', got {len(row_cells)}"

        # WT Dice
        wt_cell = row_cells[0]
        wt_num = float(re.search(r"(\d+\.\d+)", wt_cell).group(1))
        wt_csv_val = float(df_indexed.loc[f"{csv_prefix} [WT]", "Dice (%)"].split("±")[0].strip())
        assert abs(wt_num - wt_csv_val) < 0.05, f"WT mismatch for {tex_name}: {wt_num} vs {wt_csv_val}"

        # TC Dice
        tc_cell = row_cells[1]
        tc_num = float(re.search(r"(\d+\.\d+)", tc_cell).group(1))
        tc_csv_val = float(df_indexed.loc[f"{csv_prefix} [TC]", "Dice (%)"].split("±")[0].strip())
        assert abs(tc_num - tc_csv_val) < 0.05, f"TC mismatch for {tex_name}: {tc_num} vs {tc_csv_val}"

        # ET Dice
        et_cell = row_cells[2]
        et_num = float(re.search(r"(\d+\.\d+)", et_cell).group(1))
        et_csv_val = float(df_indexed.loc[f"{csv_prefix} [ET]", "Dice (%)"].split("±")[0].strip())
        assert abs(et_num - et_csv_val) < 0.05, f"ET mismatch for {tex_name}: {et_num} vs {et_csv_val}"


def test_latex_table3_and_4_match_ood_csv():
    """Verify Table 3 & Table 4 in paper/latex/main.tex match ood_3d_summary.csv."""
    assert OOD_CSV_PATH.exists()
    assert TEX_PATH.exists()

    df = pd.read_csv(OOD_CSV_PATH).set_index("Regime")
    tex_content = TEX_PATH.read_text(encoding="utf-8")

    # Table 3: Synthetic Scanner Shifts (tab:ood_synthetic_results)
    table3_match = re.search(
        r"\\label\{tab:ood_synthetic_results\}(.*?)\\end\{table\}",
        tex_content,
        re.DOTALL,
    )
    assert table3_match is not None, "Could not locate Table 3 block in LaTeX"
    table3_text = table3_match.group(1)

    # Clean Baseline nnU-Net
    match3 = re.search(r"Clean Baseline\s*&\s*3D nnU-Net Baseline\s*&(.*?)\\\\\s*$", table3_text, re.MULTILINE)
    assert match3 is not None
    cells3 = match3.group(1).split("&")
    mean_val = float(re.search(r"(\d+\.\d+)", cells3[3]).group(1))
    csv_mean = float(str(df.loc["Clean Baseline", "3D nnU-Net [mean]"]).replace("%", "").strip())
    assert abs(mean_val - csv_mean) < 0.05, f"Mismatch on Clean Baseline nnU-Net: {mean_val} vs {csv_mean}"

    # Table 4: Missing Modality Emergency Triage (tab:missing_modality_results)
    table4_match = re.search(
        r"\\label\{tab:missing_modality_results\}(.*?)\\end\{table\}",
        tex_content,
        re.DOTALL,
    )
    assert table4_match is not None, "Could not locate Table 4 block in LaTeX"
    table4_text = table4_match.group(1)

    # FLAIR Only nnU-Net
    match4 = re.search(r"FLAIR Only.*?&\s*3D nnU-Net Baseline\s*&(.*?)\\\\\s*$", table4_text, re.MULTILINE)
    assert match4 is not None
    cells4 = match4.group(1).split("&")
    flair_wt = float(re.search(r"(\d+\.\d+)", cells4[0]).group(1))
    csv_flair_wt = float(str(df.loc["Missing Modalities: FLAIR Only", "3D nnU-Net [WT]"]).replace("%", "").strip())
    assert abs(flair_wt - csv_flair_wt) < 0.05, f"Mismatch on FLAIR Only nnU-Net WT: {flair_wt} vs {csv_flair_wt}"

