from pathlib import Path

import pytest

from brats_jepa_3d.utils.aggregation import (
    aggregate_low_data_summaries,
    aggregate_master_benchmarks,
    aggregate_ood_summaries,
    discover_experiment_directories,
    export_latex_tables,
)


@pytest.fixture
def mock_experiment_dirs(tmp_path: Path):
    """Creates mock experiment directories with benchmark, low-data, and OOD metrics."""
    exp1 = tmp_path / "exp_visreg"
    exp2 = tmp_path / "exp_nnunet"
    exp3 = tmp_path / "exp_unet"

    for exp in [exp1, exp2, exp3]:
        (exp / "metrics").mkdir(parents=True)
        (exp / "checkpoints").mkdir(parents=True)

    # Master benchmark CSVs
    (exp1 / "metrics" / "master_3d_benchmark.csv").write_text(
        "Model Architecture,Dice (%),IoU (%),HD95 (mm),Latency (ms)\n"
        "3D VisReg JEPA (FPN),69.62 ± 16.34,55.40 ± 16.28,16.83 ± 17.17,73.14\n"
    )
    (exp2 / "metrics" / "master_3d_benchmark.csv").write_text(
        "Model Architecture,Dice (%),IoU (%),HD95 (mm),Latency (ms)\n"
        "3D nnU-Net (DynUNet),82.52 ± 14.14,72.05 ± 15.32,9.05 ± 14.41,126.57\n"
    )
    (exp3 / "metrics" / "master_3d_benchmark.csv").write_text(
        "Model Architecture,Dice (%),IoU (%),HD95 (mm),Latency (ms)\n"
        "3D Residual UNet,50.83 ± 25.03,37.80 ± 22.45,54.35 ± 29.28,43.11\n"
    )

    # Low-data summary CSVs
    (exp1 / "metrics" / "low_data_3d_summary.csv").write_text(
        "Fraction,3D VisReg JEPA (FPN)\n"
        "1.0%,9.90%\n"
        "5.0%,45.66%\n"
        "10.0%,52.69%\n"
        "25.0%,60.24%\n"
        "50.0%,62.78%\n"
        "100.0%,67.58%\n"
    )
    (exp2 / "metrics" / "low_data_3d_summary.csv").write_text(
        "Fraction,3D nnU-Net\n"
        "1.0%,16.40%\n"
        "5.0%,56.03%\n"
        "10.0%,64.23%\n"
        "25.0%,74.47%\n"
        "50.0%,81.18%\n"
        "100.0%,81.27%\n"
    )
    (exp3 / "metrics" / "low_data_3d_summary.csv").write_text(
        "Fraction,3D UNet\n"
        "1.0%,3.83%\n"
        "5.0%,13.35%\n"
        "10.0%,24.49%\n"
        "25.0%,44.33%\n"
        "50.0%,56.69%\n"
        "100.0%,64.76%\n"
    )

    # OOD summary CSVs
    (exp1 / "metrics" / "ood_3d_summary.csv").write_text(
        "Regime,3D VisReg JEPA (FPN)\n"
        "Clean Baseline,69.62%\n"
        "Rician Noise (sigma=0.08),66.15%\n"
        "B1 Bias Field Inhomogeneity,67.80%\n"
        "Missing Modalities: T1c Only,58.40%\n"
        "Missing Modalities: FLAIR Only,54.10%\n"
    )
    (exp2 / "metrics" / "ood_3d_summary.csv").write_text(
        "Regime,3D nnU-Net\n"
        "Clean Baseline,82.52%\n"
        "Rician Noise (sigma=0.08),74.30%\n"
        "B1 Bias Field Inhomogeneity,79.10%\n"
        "Missing Modalities: T1c Only,42.10%\n"
        "Missing Modalities: FLAIR Only,39.50%\n"
    )
    (exp3 / "metrics" / "ood_3d_summary.csv").write_text(
        "Regime,3D UNet\n"
        "Clean Baseline,50.83%\n"
        "Rician Noise (sigma=0.08),38.20%\n"
        "B1 Bias Field Inhomogeneity,45.60%\n"
        "Missing Modalities: T1c Only,21.30%\n"
        "Missing Modalities: FLAIR Only,18.70%\n"
    )

    return [exp1, exp2, exp3], tmp_path


def test_discover_experiment_directories(mock_experiment_dirs):
    _exp_dirs, tmp_path = mock_experiment_dirs
    discovered = discover_experiment_directories(tmp_path)
    assert len(discovered) == 3
    names = [d.name for d in discovered]
    assert "exp_visreg" in names
    assert "exp_nnunet" in names
    assert "exp_unet" in names


def test_aggregate_master_benchmarks(mock_experiment_dirs):
    exp_dirs, tmp_path = mock_experiment_dirs
    out_dir = tmp_path / "aggregated"
    df = aggregate_master_benchmarks(exp_dirs, output_dir=out_dir)

    assert len(df) == 3
    # Sorted by Dice descending: nnUNet (82.5%) -> VisReg (69.6%) -> UNet (50.8%)
    assert "nnU-Net" in df.iloc[0]["Model Architecture"]
    assert "VisReg" in df.iloc[1]["Model Architecture"]
    assert "UNet" in df.iloc[2]["Model Architecture"]

    assert (out_dir / "master_3d_benchmark.csv").exists()
    assert (out_dir / "master_3d_benchmark.md").exists()


def test_aggregate_low_data_summaries(mock_experiment_dirs):
    exp_dirs, tmp_path = mock_experiment_dirs
    out_dir = tmp_path / "aggregated"
    df = aggregate_low_data_summaries(exp_dirs, output_dir=out_dir)

    assert len(df) == 6
    assert "Fraction" in df.columns
    assert "3D VisReg JEPA (FPN)" in df.columns
    assert "3D nnU-Net" in df.columns
    assert "3D UNet" in df.columns

    # Check that 100% row has correct values
    row_100 = df[df["Fraction"] == "100.0%"].iloc[0]
    assert row_100["3D VisReg JEPA (FPN)"] == "67.58%"
    assert row_100["3D nnU-Net"] == "81.27%"
    assert row_100["3D UNet"] == "64.76%"

    assert (out_dir / "low_data_3d_summary.csv").exists()
    assert (out_dir / "low_data_3d_summary.md").exists()


def test_aggregate_ood_summaries(mock_experiment_dirs):
    exp_dirs, tmp_path = mock_experiment_dirs
    out_dir = tmp_path / "aggregated"
    df = aggregate_ood_summaries(exp_dirs, output_dir=out_dir)

    assert len(df) == 5
    assert "Regime" in df.columns
    assert "3D VisReg JEPA (FPN)" in df.columns
    assert "3D nnU-Net" in df.columns
    assert "3D UNet" in df.columns

    clean_row = df[df["Regime"] == "Clean Baseline"].iloc[0]
    assert clean_row["3D VisReg JEPA (FPN)"] == "69.62%"
    assert clean_row["3D nnU-Net"] == "82.52%"
    assert clean_row["3D UNet"] == "50.83%"

    assert (out_dir / "ood_3d_summary.csv").exists()
    assert (out_dir / "ood_3d_summary.md").exists()


def test_export_latex_tables(mock_experiment_dirs):
    exp_dirs, tmp_path = mock_experiment_dirs
    out_dir = tmp_path / "aggregated"
    master_df = aggregate_master_benchmarks(exp_dirs, output_dir=out_dir)
    low_data_df = aggregate_low_data_summaries(exp_dirs, output_dir=out_dir)
    ood_df = aggregate_ood_summaries(exp_dirs, output_dir=out_dir)

    tex_out = out_dir / "tables_latex.tex"
    latex_code = export_latex_tables(
        master_df=master_df, low_data_df=low_data_df, ood_df=ood_df, output_path=tex_out
    )

    assert "\\begin{table}" in latex_code
    assert "\\label{tab:master_benchmark}" in latex_code
    assert "\\label{tab:low_data}" in latex_code
    assert "\\label{tab:ood_robustness}" in latex_code
    assert tex_out.exists()

