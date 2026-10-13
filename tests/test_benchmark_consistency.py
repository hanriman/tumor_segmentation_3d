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

    # Assert correct numbers are present
    assert "29.61%" in docs_content, "ViT-UNETR Scratch score 29.61% missing from docs/03"
    assert "32.09%" in docs_content, "VisReg UNETR score 32.09% missing from docs/03"
    assert "7.03% *(Collapse)*" in docs_content, "ResUNet collapse 7.03% missing from docs/03"
    assert "+2.48" in docs_content, "+2.48pp JEPA gain missing from docs/03"
