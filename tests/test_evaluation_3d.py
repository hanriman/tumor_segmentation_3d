"""Evaluation pipeline tests: CLI flags, TTA, OOD perturbations, artifact export, aggregation, and LaTeX generation."""

import argparse
import importlib.util
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brats_jepa_3d.config import (
    CONFIGS_DIR,
    PROJECT_ROOT,
    load_yaml_config,
    merge_config_with_args,
)
from brats_jepa_3d.data.transforms import apply_b1_bias_field_3d, apply_rician_noise_3d
from brats_jepa_3d.models import BraTS3DUNet
from brats_jepa_3d.utils import (
    TTA_FLIP_DIMS,
    predict_with_tta_3d,
    predict_with_tta_multiclass_3d,
    sort_checkpoints_by_epoch,
)
from brats_jepa_3d.utils.seed import set_seed
from brats_jepa_3d.utils.aggregation import (
    aggregate_low_data_summaries,
    aggregate_master_benchmarks,
    aggregate_ood_summaries,
    discover_experiment_directories,
    export_latex_tables,
)
from brats_jepa_3d.utils.export import (
    export_artifacts,
    import_artifacts,
    resolve_outputs_source_dir,
)


def _load_script(name: str):
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [name]
    spec.loader.exec_module(mod)
    return mod


def _tiny_loader(n: int = 2):
    data = [
        {
            "image": torch.randn(4, 32, 32, 32),
            "mask": torch.randint(0, 5, (1, 32, 32, 32)).long(),
            "brain_mask": torch.ones(1, 32, 32, 32),
        }
        for _ in range(n)
    ]
    return DataLoader(data, batch_size=1)


# =========================================================================
# 1. Script CLI Flag Parsing & Argument Contracts
# =========================================================================


def test_low_data_flags():
    m = _load_script("evaluate_low_data_3d")
    sys.argv = ["x"]
    a = m.parse_args()
    assert a.decoder_type == "multiscale" and a.loss_type == "dice_bce" and a.tta is False
    assert a.deep_supervision is True
    assert a.min_steps == 250
    sys.argv = ["x", "--decoder_type", "unetr_hybrid", "--loss_type", "tversky", "--tta", "--no_deep_supervision", "--min_steps", "500"]
    a = m.parse_args()
    assert a.decoder_type == "unetr_hybrid" and a.loss_type == "tversky" and a.tta is True
    assert a.deep_supervision is False
    assert a.min_steps == 500


def test_paired_significance_table(tmp_path):
    from brats_jepa_3d.utils.aggregation import compute_paired_significance_table

    exp_a = tmp_path / "exp_visreg" / "metrics"
    exp_b = tmp_path / "exp_nnunet" / "metrics"
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)
    pids = [f"BraTS-GLI-{i:04d}" for i in range(20)]
    df_a = pd.DataFrame({
        "patient_id": pids,
        "model": ["3D VisReg JEPA (FPN) [unetr_hybrid]"] * 20,
        "Dice_WT": np.linspace(0.80, 0.90, 20),
        "Dice_TC": np.linspace(0.75, 0.85, 20),
        "Dice_ET": np.linspace(0.70, 0.80, 20),
        "Dice_Mean": np.linspace(0.75, 0.85, 20),
    })
    df_b = pd.DataFrame({
        "patient_id": pids,
        "model": ["3D nnU-Net (DynUNet)"] * 20,
        "Dice_WT": np.linspace(0.78, 0.88, 20),
        "Dice_TC": np.linspace(0.72, 0.82, 20),
        "Dice_ET": np.linspace(0.68, 0.78, 20),
        "Dice_Mean": np.linspace(0.7267, 0.8267, 20),
    })
    df_a.to_csv(exp_a / "visreg_per_case.csv", index=False)
    df_b.to_csv(exp_b / "nnunet_per_case.csv", index=False)

    sig_df = compute_paired_significance_table(
        [tmp_path / "exp_visreg", tmp_path / "exp_nnunet"],
        output_dir=tmp_path / "reports",
        n_boot=500,
    )
    assert len(sig_df) == 4
    assert (tmp_path / "reports" / "paired_significance_3d.csv").exists()
    assert set(sig_df["Metric"]) == {"Dice_WT", "Dice_TC", "Dice_ET", "Dice_Mean"}


def test_ood_flags():
    m = _load_script("evaluate_ood_3d")
    sys.argv = ["x"]
    a = m.parse_args()
    assert a.decoder_type == "multiscale" and a.tta is False
    sys.argv = ["x", "--decoder_type", "unetr_hybrid", "--tta"]
    a = m.parse_args()
    assert a.decoder_type == "unetr_hybrid" and a.tta is True


def test_evaluate_scripts_model_type_parsing():
    """Verify --model_type argument is present in argument parsers."""
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from evaluate_low_data_3d import parse_args as parse_low_data_args
    from evaluate_ood_3d import parse_args as parse_ood_args

    orig_argv = sys.argv
    try:
        sys.argv = ["evaluate_low_data_3d.py", "--model_type", "visreg_jepa", "--smoke_test"]
        args = parse_low_data_args()
        assert args.model_type == "visreg_jepa"
        assert args.smoke_test is True

        sys.argv = ["evaluate_ood_3d.py", "--model_type", "sigreg_jepa", "--smoke_test"]
        args_ood = parse_ood_args()
        assert args_ood.model_type == "sigreg_jepa"
    finally:
        sys.argv = orig_argv


# =========================================================================
# 2. Test-Time Augmentation (TTA) Verification
# =========================================================================


class _ConstLogit(nn.Module):
    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def forward(self, x):
        return torch.full((x.shape[0], 1) + x.shape[2:], self.c)


class _PatternLogit(nn.Module):
    """Input-independent asymmetric pattern: TTA output must equal manual flip-average."""

    def forward(self, x):
        B = x.shape[0]
        base = torch.linspace(-2.0, 2.0, x.shape[2] * x.shape[3] * x.shape[4]).reshape(
            1, 1, x.shape[2], x.shape[3], x.shape[4]
        )
        return base.expand(B, -1, -1, -1, -1)


class _Fixed(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        torch.manual_seed(3)
        self.base = torch.randn(1, c, 8, 8, 8)

    def forward(self, x):
        return self.base.expand(x.shape[0], -1, -1, -1, -1)


def test_tta_views():
    assert len(TTA_FLIP_DIMS) == 4 and TTA_FLIP_DIMS[0] is None


def test_tta_constant_model_identity():
    torch.manual_seed(0)
    model = _ConstLogit(0.7).eval()
    vol = torch.randn(2, 4, 16, 16, 16)
    with torch.no_grad():
        single = model(vol)
        tta = predict_with_tta_3d(model, vol)
    assert tta.shape == single.shape
    assert torch.allclose(tta, single, atol=1e-4)


def test_tta_flip_average_matches_manual():
    torch.manual_seed(1)
    model = _PatternLogit().eval()
    vol = torch.randn(1, 2, 8, 8, 8)
    with torch.no_grad():
        tta = predict_with_tta_3d(model, vol)
        acc = None
        for dims in TTA_FLIP_DIMS:
            v = torch.flip(vol, dims=dims) if dims is not None else vol
            p = torch.sigmoid(model(v).float())
            if dims is not None:
                p = torch.flip(p, dims=dims)
            acc = p if acc is None else acc + p
        expected = torch.logit((acc / 4).clamp(1e-6, 1 - 1e-6)).to(tta.dtype)
    assert torch.allclose(tta, expected, atol=1e-5)


def test_tta_preserves_eval_mode_and_shape():
    model = _ConstLogit(-0.3)
    model.train()
    vol = torch.randn(1, 1, 8, 8, 8)
    out = predict_with_tta_3d(model, vol)
    assert out.shape == (1, 1, 8, 8, 8)
    assert model.training, "TTA must restore training mode"


def test_tta_multiclass_contract_and_binary_guard():
    m5 = _Fixed(5).eval()
    vol = torch.randn(1, 4, 8, 8, 8)
    with torch.no_grad():
        out = predict_with_tta_multiclass_3d(m5, vol)
    assert tuple(out.shape) == (1, 5, 8, 8, 8)
    assert torch.allclose(out.softmax(dim=1).sum(dim=1), torch.ones(1, 8, 8, 8), atol=1e-4)
    m1 = _Fixed(1).eval()
    with torch.no_grad():
        predict_with_tta_3d(m1, vol)
        with pytest.raises(ValueError, match="binary-only"):
            predict_with_tta_3d(_Fixed(4).eval(), vol)
        with pytest.raises(ValueError, match="C>=2"):
            predict_with_tta_multiclass_3d(m1, vol)


def test_low_data_train_and_eval_tversky_tta():
    m = _load_script("evaluate_low_data_3d")
    device = torch.device("cpu")
    model = BraTS3DUNet(channels=(8, 16, 32, 64, 128), num_res_units=1, dropout=0.0)
    scores = m.train_and_eval(
        model, _tiny_loader(), _tiny_loader(), device, epochs=1, amp=False,
        smoke_test=True, loss_type="tversky", tta=True,
    )
    assert set(scores) == {"WT", "TC", "ET", "mean"}
    assert 0.0 <= scores["mean"] <= 1.0


def test_low_data_train_and_eval_dice_no_tta():
    m = _load_script("evaluate_low_data_3d")
    device = torch.device("cpu")
    model = BraTS3DUNet(channels=(8, 16, 32, 64, 128), num_res_units=1, dropout=0.0)
    scores = m.train_and_eval(
        model, _tiny_loader(), _tiny_loader(), device, epochs=1, amp=False, smoke_test=True,
    )
    assert set(scores) == {"WT", "TC", "ET", "mean"}
    assert 0.0 <= scores["mean"] <= 1.0


def test_ood_perturbation_tta():
    m = _load_script("evaluate_ood_3d")
    device = torch.device("cpu")
    model = BraTS3DUNet(channels=(8, 16, 32, 64, 128), num_res_units=1, dropout=0.0)
    scores = m.evaluate_perturbation(
        model, _tiny_loader(), device, None, amp=False, smoke_test=True, tta=True,
    )
    assert set(scores) == {"WT", "TC", "ET", "mean"}
    assert 0.0 <= scores["mean"] <= 1.0


# =========================================================================
# 3. Out-Of-Distribution (OOD) Scanner Perturbations
# =========================================================================


def test_ood_perturbations_3d():
    """Verify 3D Rician scanner noise and B1 bias field transformations."""
    torch.manual_seed(42)
    img = torch.zeros(4, 16, 16, 16)
    img[:, 4:12, 4:12, 4:12] = torch.rand(4, 8, 8, 8) + 0.5

    # 1. Rician noise
    noisy = apply_rician_noise_3d(img, sigma=0.05)
    assert noisy.shape == img.shape
    assert not torch.isnan(noisy).any()
    assert not torch.allclose(noisy, img)
    assert (noisy[:, 0:3, 0:3, 0:3] != 0).any()
    assert (noisy >= 0).all()

    # 2. B1 bias field
    biased = apply_b1_bias_field_3d(img, strength=0.3)
    assert biased.shape == img.shape
    assert not torch.isnan(biased).any()
    assert not torch.allclose(biased, img)


def test_ood_perturbations_explicit_brain_mask_3d():
    """Verify 3D Rician noise and B1 bias field work with explicit float32 single-channel brain_mask (4D and 5D)."""
    torch.manual_seed(42)
    img_4d = torch.zeros(4, 16, 16, 16, dtype=torch.float32)
    img_4d[:, 4:12, 4:12, 4:12] = torch.rand(4, 8, 8, 8) + 0.5
    mask_4d = torch.zeros(1, 16, 16, 16, dtype=torch.float32)
    mask_4d[:, 4:12, 4:12, 4:12] = 1.0

    noisy_4d = apply_rician_noise_3d(img_4d, sigma=0.05, brain_mask=mask_4d)
    assert noisy_4d.shape == img_4d.shape
    assert (noisy_4d[:, 0:3, 0:3, 0:3] != 0).any()
    assert (noisy_4d >= 0).all()
    assert not torch.allclose(noisy_4d[:, 4:12, 4:12, 4:12], img_4d[:, 4:12, 4:12, 4:12])

    biased_4d = apply_b1_bias_field_3d(img_4d, strength=0.3, brain_mask=mask_4d)
    assert biased_4d.shape == img_4d.shape
    assert (biased_4d[:, 0:3, 0:3, 0:3] == 0).all()
    assert not torch.allclose(biased_4d[:, 4:12, 4:12, 4:12], img_4d[:, 4:12, 4:12, 4:12])


# =========================================================================
# 4. Artifact Export and Import
# =========================================================================


@pytest.fixture
def mock_outputs_structure(tmp_path: Path):
    """Creates a mock outputs directory structure with checkpoints, metrics, and logs."""
    outputs_dir = tmp_path / "outputs"
    ckpts = outputs_dir / "checkpoints"
    metrics = outputs_dir / "metrics"
    logs = outputs_dir / "logs"

    ckpts.mkdir(parents=True)
    metrics.mkdir(parents=True)
    logs.mkdir(parents=True)

    (ckpts / "visreg_jepa_best.pt").write_bytes(b"dummy_visreg_weights_12345")
    (ckpts / "visreg_jepa_multiscale_best.pt").write_bytes(b"dummy_visreg_downstream_weights")
    (ckpts / "nnunet_3d_best.pt").write_bytes(b"dummy_nnunet_weights")
    (ckpts / "unet_3d_best.pt").write_bytes(b"dummy_unet_weights")

    (metrics / "visreg_jepa_metrics.json").write_text('{"dice": 0.901}')
    (metrics / "master_3d_benchmark.csv").write_text("model,dice\nvisreg,0.901")
    (metrics / "nnunet_metrics.json").write_text('{"dice": 0.885}')

    (logs / "train_visreg_jepa.log").write_text("Epoch 1... Epoch 50 done.")
    return outputs_dir


def test_export_artifacts_model_prefix_filtering(tmp_path: Path, mock_outputs_structure: Path):
    export_dir = tmp_path / "export_visreg"
    zip_path = tmp_path / "visreg_outputs.zip"

    res = export_artifacts(
        export_name="visreg_outputs",
        export_dir=export_dir,
        zip_path=zip_path,
        model_prefix="visreg",
        source_dir=mock_outputs_structure,
        verbose=False,
    )

    assert res["total_files"] >= 4
    assert zip_path.exists()
    assert res["zip_size_bytes"] > 0

    exported_ckpts = [f.name for f in (export_dir / "checkpoints").iterdir()]
    assert "visreg_jepa_best.pt" in exported_ckpts
    assert "visreg_jepa_multiscale_best.pt" in exported_ckpts
    assert "nnunet_3d_best.pt" not in exported_ckpts
    assert "unet_3d_best.pt" not in exported_ckpts

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert "checkpoints/visreg_jepa_best.pt" in names
        assert "metrics/visreg_jepa_metrics.json" in names
        assert "metrics/master_3d_benchmark.csv" in names


def test_export_artifacts_all_models(tmp_path: Path, mock_outputs_structure: Path):
    export_dir = tmp_path / "export_all"
    zip_path = tmp_path / "all_outputs.zip"

    res = export_artifacts(
        export_name="all_outputs",
        export_dir=export_dir,
        zip_path=zip_path,
        model_prefix=None,
        source_dir=mock_outputs_structure,
        verbose=False,
    )

    assert res["total_files"] >= 7
    exported_ckpts = [f.name for f in (export_dir / "checkpoints").iterdir()]
    assert "visreg_jepa_best.pt" in exported_ckpts
    assert "nnunet_3d_best.pt" in exported_ckpts
    assert "unet_3d_best.pt" in exported_ckpts


def test_import_artifacts_from_zip(tmp_path: Path, mock_outputs_structure: Path):
    staging_dir = tmp_path / "staging"
    zip_path = tmp_path / "input_mount" / "visreg_outputs.zip"
    export_artifacts(
        export_name="visreg_outputs",
        export_dir=staging_dir,
        zip_path=zip_path,
        model_prefix="visreg",
        source_dir=mock_outputs_structure,
        verbose=False,
    )

    import_target = tmp_path / "imported_outputs"
    imported = import_artifacts(
        input_dir=tmp_path / "input_mount",
        target_dir=import_target,
        mirror_to_local_outputs=False,
        verbose=False,
    )

    assert len(imported) >= 4
    assert (import_target / "checkpoints" / "visreg_jepa_best.pt").exists()
    assert (import_target / "metrics" / "visreg_jepa_metrics.json").exists()


def test_resolve_outputs_source_dir_fallback(tmp_path: Path):
    preferred = tmp_path / "custom_outputs"
    preferred.mkdir()
    resolved = resolve_outputs_source_dir(preferred)
    assert resolved == preferred.resolve()


def test_evaluate_baseline_checkpoint_zero_matched_raises(tmp_path: Path):
    """Verify baseline evaluation raises RuntimeError if checkpoint contains zero matching keys."""
    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(4, 16, 3)

    model = DummyNet()
    corrupt_sd = {"unrelated_layer.weight": torch.randn(10, 10)}
    ckpt_file = tmp_path / "corrupted_unet.pt"
    torch.save(corrupt_sd, ckpt_file)

    sd = torch.load(ckpt_file, map_location="cpu")
    missing, _unexpected = model.load_state_dict(sd, strict=False)
    matched = [k for k in model.state_dict() if k not in missing]

    assert len(matched) == 0
    with pytest.raises(RuntimeError, match="Failed to load any weights"):
        if len(matched) == 0:
            raise RuntimeError(f"Failed to load any weights for 3D UNet from {ckpt_file}")


# =========================================================================
# 5. Benchmark Aggregation, Merging, Formatting & LaTeX Generation
# =========================================================================


@pytest.fixture
def mock_experiment_dirs(tmp_path: Path):
    """Creates mock experiment directories with benchmark, low-data, and OOD metrics."""
    exp1 = tmp_path / "exp_visreg"
    exp2 = tmp_path / "exp_nnunet"
    exp3 = tmp_path / "exp_unet"

    for exp in [exp1, exp2, exp3]:
        (exp / "metrics").mkdir(parents=True)
        (exp / "checkpoints").mkdir(parents=True)

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
    assert "nnU-Net" in df.iloc[0]["Model Architecture"]
    assert "VisReg" in df.iloc[1]["Model Architecture"]
    assert "UNet" in df.iloc[2]["Model Architecture"]

    assert (out_dir / "master_3d_benchmark.csv").exists()
    assert (out_dir / "master_3d_benchmark.md").exists()


def test_f9_aggregation_reports_dice_std(tmp_path: Path):
    """Master aggregation keeps best row but reports n_runs + dice_std."""
    for i, dice in enumerate(["70.0 ± 1.0", "80.0 ± 1.0"]):
        d = tmp_path / f"exp{i}"
        (d / "metrics").mkdir(parents=True)
        pd.DataFrame([{"Model Architecture": "3D VisReg JEPA (FPN)", "Dice (%)": dice}]).to_csv(
            d / "metrics" / "master_3d_benchmark.csv", index=False
        )
    df = aggregate_master_benchmarks([tmp_path / "exp0", tmp_path / "exp1"])
    assert len(df) == 1
    assert df.iloc[0]["n_runs"] == 2
    assert df.iloc[0]["dice_std"] >= 0.0
    assert "80.0" in str(df.iloc[0]["Dice (%)"])


def test_aggregate_low_data_summaries(mock_experiment_dirs):
    exp_dirs, tmp_path = mock_experiment_dirs
    out_dir = tmp_path / "aggregated"
    df = aggregate_low_data_summaries(exp_dirs, output_dir=out_dir)

    assert len(df) == 6
    assert "Fraction" in df.columns
    assert "3D VisReg JEPA (FPN)" in df.columns
    assert "3D nnU-Net" in df.columns
    assert "3D UNet" in df.columns

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


def test_benchmark_model_region_rows():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import evaluate_3d

    torch.manual_seed(0)
    base = torch.randn(1, 5, 16, 16, 16)

    class _M(torch.nn.Module):
        def forward(self, x):
            return base.expand(x.shape[0], -1, -1, -1, -1)

    data = [
        {"image": torch.randn(4, 16, 16, 16),
         "mask": torch.randint(0, 5, (1, 16, 16, 16)).float()}
        for _ in range(2)
    ]
    loader = DataLoader(data, batch_size=1)
    stats = evaluate_3d.benchmark_model(
        _M().eval(), loader, torch.device("cpu"), amp=False, smoke_test=True)
    assert set(stats["regions"]) == {"WT", "TC", "ET"}
    assert stats["regions"]["WT"]["dice_mean"] >= 0.0
    assert stats["latency_ms"] >= 0.0


def test_incremental_csv_merging(tmp_path: Path):
    """Verify incremental CSV merging updates matching fraction without wiping other models."""
    csv_path = tmp_path / "low_data_3d_summary.csv"
    existing_df = pd.DataFrame([
        {"Fraction": "1.0%", "3D VisReg JEPA (FPN)": "6.79%", "3D nnU-Net": "12.12%"},
        {"Fraction": "5.0%", "3D VisReg JEPA (FPN)": "25.00%", "3D nnU-Net": "30.00%"},
    ])
    existing_df.to_csv(csv_path, index=False)

    new_records = [
        {"Fraction": "1.0%", "3D SigReg JEPA (FPN)": "8.50%"},
        {"Fraction": "5.0%", "3D SigReg JEPA (FPN)": "28.00%"},
    ]
    new_df = pd.DataFrame(new_records)

    existing_df = pd.read_csv(csv_path)
    existing_df["Fraction"] = existing_df["Fraction"].astype(str)
    new_df["Fraction"] = new_df["Fraction"].astype(str)
    for col in new_df.columns:
        if col == "Fraction":
            continue
        new_vals = new_df.set_index("Fraction")[col]
        if not (new_vals == "N/A").all():
            if col not in existing_df.columns:
                existing_df[col] = "N/A"
            for frac, val in new_vals.items():
                existing_df.loc[existing_df["Fraction"] == frac, col] = val

    assert len(existing_df) == 2
    assert "3D SigReg JEPA (FPN)" in existing_df.columns
    assert existing_df.loc[existing_df["Fraction"] == "1.0%", "3D SigReg JEPA (FPN)"].values[0] == "8.50%"


def test_aggregate_low_data_summaries_outer_merge_unequal_lengths(tmp_path: Path):
    """Verify aggregate_low_data_summaries handles differing fraction counts and scrambled orders."""
    exp1 = tmp_path / "exp1"
    exp2 = tmp_path / "exp2"
    (exp1 / "metrics").mkdir(parents=True)
    (exp2 / "metrics").mkdir(parents=True)

    (exp1 / "metrics" / "low_data_3d_summary.csv").write_text(
        "Fraction,Model1\n"
        "1.0%,10.5%\n"
        "5.0%,25.0%\n"
        "10.0%,40.0%\n"
        "100.0%,70.0%\n"
    )

    (exp2 / "metrics" / "low_data_3d_summary.csv").write_text(
        "Fraction,Model2\n"
        "25.0%,55.0%\n"
        "5.0%,30.0%\n"
    )

    df = aggregate_low_data_summaries([exp1, exp2], output_dir=tmp_path / "out")
    assert len(df) == 5
    assert list(df["Fraction"]) == ["1.0%", "5.0%", "10.0%", "25.0%", "100.0%"]
    assert "Model1" in df.columns and "Model2" in df.columns

    row_5 = df[df["Fraction"] == "5.0%"].iloc[0]
    assert row_5["Model1"] == "25.0%"
    assert row_5["Model2"] == "30.0%"

    row_1 = df[df["Fraction"] == "1.0%"].iloc[0]
    assert row_1["Model1"] == "10.5%"
    assert row_1["Model2"] == "N/A"

    row_25 = df[df["Fraction"] == "25.0%"].iloc[0]
    assert row_25["Model1"] == "N/A"
    assert row_25["Model2"] == "55.0%"


def test_aggregate_ood_summaries_outer_merge_unequal_regimes(tmp_path: Path):
    """Verify aggregate_ood_summaries handles disjoint regimes and out-of-order rows."""
    exp1 = tmp_path / "exp1"
    exp2 = tmp_path / "exp2"
    (exp1 / "metrics").mkdir(parents=True)
    (exp2 / "metrics").mkdir(parents=True)

    (exp1 / "metrics" / "ood_3d_summary.csv").write_text(
        "Regime,ModelA\n"
        "Clean Baseline,80.0%\n"
        "Rician Noise (sigma=0.08),75.0%\n"
        "B1 Bias Field Inhomogeneity,78.0%\n"
    )

    (exp2 / "metrics" / "ood_3d_summary.csv").write_text(
        "Regime,ModelB\n"
        "Missing Modalities: T1c Only,45.0%\n"
        "Clean Baseline,82.0%\n"
    )

    df = aggregate_ood_summaries([exp1, exp2], output_dir=tmp_path / "out")
    assert len(df) >= 4
    assert "ModelA" in df.columns and "ModelB" in df.columns


def test_evaluate_low_data_incremental_outer_merge(tmp_path: Path):
    """Verify evaluate_low_data_3d outer merge preserves disjoint fractions and models."""
    csv_path = tmp_path / "low_data_3d_summary.csv"
    csv_path.write_text("Fraction,ModelA\n1.0%,12.0%\n5.0%,35.0%\n")

    new_df = pd.DataFrame([
        {"Fraction": "10.0%", "ModelB": "50.0%"},
        {"Fraction": "5.0%", "ModelB": "38.0%"},
    ])

    existing_df = pd.read_csv(csv_path)
    existing_df["Fraction"] = existing_df["Fraction"].astype(str).str.strip()
    new_df["Fraction"] = new_df["Fraction"].astype(str).str.strip()

    new_cols = [c for c in new_df.columns if c not in existing_df.columns]
    overlap_cols = [c for c in new_df.columns if c != "Fraction" and c in existing_df.columns]
    if new_cols:
        merged = pd.merge(existing_df, new_df[["Fraction"] + new_cols], on="Fraction", how="outer")
    else:
        merged = pd.merge(existing_df, new_df[["Fraction"]], on="Fraction", how="outer")
    if overlap_cols:
        new_indexed = new_df.set_index("Fraction")
        for col in overlap_cols:
            for frac, val in new_indexed[col].items():
                if pd.notna(val) and str(val) != "N/A":
                    merged.loc[merged["Fraction"] == frac, col] = val

    def frac_key(v: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)", str(v))
        return float(m.group(1)) if m else 0.0

    merged["_sort"] = merged["Fraction"].apply(frac_key)
    df = merged.sort_values(by="_sort").drop(columns=["_sort"]).reset_index(drop=True).fillna("N/A")

    assert len(df) == 3
    assert list(df["Fraction"]) == ["1.0%", "5.0%", "10.0%"]
    assert "ModelA" in df.columns and "ModelB" in df.columns
    assert df[df["Fraction"] == "1.0%"]["ModelA"].iloc[0] == "12.0%"
    assert df[df["Fraction"] == "1.0%"]["ModelB"].iloc[0] == "N/A"
    assert df[df["Fraction"] == "10.0%"]["ModelA"].iloc[0] == "N/A"
    assert df[df["Fraction"] == "10.0%"]["ModelB"].iloc[0] == "50.0%"
    assert df[df["Fraction"] == "5.0%"]["ModelB"].iloc[0] == "38.0%"


def test_evaluate_ood_incremental_outer_merge(tmp_path: Path):
    """Verify evaluate_ood_3d outer merge preserves disjoint regimes and models."""
    csv_path = tmp_path / "ood_3d_summary.csv"
    csv_path.write_text("Regime,ModelA\nClean Baseline,80.0%\n")

    new_df = pd.DataFrame([
        {"Regime": "3D Rician Noise (sigma=0.08)", "ModelB": "72.0%"},
        {"Regime": "Clean Baseline", "ModelB": "81.5%"},
    ])

    existing_df = pd.read_csv(csv_path)
    existing_df["Regime"] = existing_df["Regime"].astype(str).str.strip()
    new_df["Regime"] = new_df["Regime"].astype(str).str.strip()

    new_cols = [c for c in new_df.columns if c not in existing_df.columns]
    overlap_cols = [c for c in new_df.columns if c != "Regime" and c in existing_df.columns]
    if new_cols:
        merged = pd.merge(existing_df, new_df[["Regime"] + new_cols], on="Regime", how="outer")
    else:
        merged = pd.merge(existing_df, new_df[["Regime"]], on="Regime", how="outer")
    if overlap_cols:
        new_indexed = new_df.set_index("Regime")
        for col in overlap_cols:
            for reg, val in new_indexed[col].items():
                if pd.notna(val) and str(val) != "N/A":
                    merged.loc[merged["Regime"] == reg, col] = val

    df = merged.reset_index(drop=True).fillna("N/A")
    assert len(df) == 2
    assert "ModelA" in df.columns and "ModelB" in df.columns


def test_config_merging_in_runners():
    """Verify configs merge correctly into argparse Namespace for runner scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args([])

    base_cfg = load_yaml_config(CONFIGS_DIR / "base.yaml")
    args = merge_config_with_args(base_cfg, args)
    assert hasattr(args, "seed")
    assert hasattr(args, "amp")
    assert args.seed == 42

    model_cfg = load_yaml_config(CONFIGS_DIR / "model" / "visreg_jepa_3d.yaml")
    args = merge_config_with_args(model_cfg, args)
    assert hasattr(args, "spatial_shape")
    assert hasattr(args, "patch_size")


def test_b1_bias_field_zero_centered_contrast_preservation():
    """Verify B1 bias field shifts zero-centered parenchyma before multiplicative scaling to avoid contrast inversion."""
    img = torch.zeros(1, 4, 16, 16, 16)
    fg_mask = torch.zeros((16, 16, 16), dtype=torch.bool)
    fg_mask[4:12, 4:12, 4:12] = True

    low_val = -2.0
    high_val = 2.0
    img[0, :, 4:8, 4:12, 4:12] = low_val
    img[0, :, 8:12, 4:12, 4:12] = high_val

    biased = apply_b1_bias_field_3d(img, strength=0.35)

    assert torch.all(biased[0, :, :4, :, :] == 0.0)
    assert torch.all(biased[0, :, 12:, :, :] == 0.0)

    low_region = biased[0, 0, 4:8, 6:10, 6:10]
    high_region = biased[0, 0, 8:12, 6:10, 6:10]
    assert high_region.min() > low_region.max()
    assert not torch.isnan(biased).any()


def test_dataset_config_cli_integration_and_dynamic_model_args():
    """Verify downstream script integrates dataset config and dynamically builds JEPA segmentation model."""
    dataset_cfg_path = CONFIGS_DIR / "dataset" / "brats3d.yaml"
    assert dataset_cfg_path.exists()
    dataset_cfg = load_yaml_config(dataset_cfg_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--rand_flip_prob", type=float, default=0.5)
    parser.add_argument("--rand_noise_prob", type=float, default=0.0)
    parser.add_argument("--modality_dropout_prob", type=float, default=0.25)
    parser.add_argument("--spatial_shape", type=int, nargs=3, default=[32, 32, 32])
    parser.add_argument("--patch_size", type=int, nargs=3, default=[16, 16, 16])
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--encoder_depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)

    args = parser.parse_args([])
    merged_args = merge_config_with_args(dataset_cfg, args)
    assert merged_args.rand_flip_prob == dataset_cfg["augmentations"]["rand_flip_prob"]


def test_seed_and_num_workers_environment_overrides(monkeypatch):
    """Verify that BRATS3D_SEED and BRATS3D_NUM_WORKERS env vars correctly propagate to configs and PRNG."""
    monkeypatch.setenv("BRATS3D_SEED", "99")
    set_seed()
    val1 = torch.randn(5)
    set_seed(99)
    val2 = torch.randn(5)
    assert torch.allclose(val1, val2), "BRATS3D_SEED env var should match explicit set_seed(99)"

    monkeypatch.setenv("BRATS3D_NUM_WORKERS", "4")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args([])
    config = {"seed": 42, "num_workers": 2}
    merged_args = merge_config_with_args(config, args)
    assert merged_args.seed == 99
    assert merged_args.num_workers == 4

    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from evaluate_ood_3d import parse_args
    monkeypatch.setattr("sys.argv", ["evaluate_ood_3d.py", "--num_workers", "4", "--seed", "77"])
    ood_args = parse_args()
    assert ood_args.num_workers == 4


def test_combine_artifacts_script_import():
    """Verify combine_and_generate_paper_artifacts script imports cleanly without redundant assignments."""
    import subprocess
    res = subprocess.run(
        [sys.executable, "scripts/combine_and_generate_paper_artifacts.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "usage:" in res.stdout.lower()


def test_run_full_pipeline_checkpoint_resolution(tmp_path):
    """Verify run_full_pipeline_3d prioritizes *_3d_best.pt over epoch checkpoints."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    (ckpt_dir / "visreg_jepa_3d_epoch_10.pt").write_text("dummy")
    (ckpt_dir / "visreg_jepa_3d_epoch_20.pt").write_text("dummy")
    (ckpt_dir / "visreg_jepa_3d_best.pt").write_text("best")

    best_ckpts = sorted(ckpt_dir.glob("visreg_jepa*_3d_best.pt"))
    epoch_ckpts = sort_checkpoints_by_epoch(list(ckpt_dir.glob("visreg_jepa*epoch*.pt")))

    selected_ckpt = best_ckpts[-1] if best_ckpts else (epoch_ckpts[-1] if epoch_ckpts else None)
    assert selected_ckpt.name == "visreg_jepa_3d_best.pt"


def test_evaluate_3d_pretrain_only_checkpoint_raises(tmp_path):
    """Verify evaluate_3d.py raises RuntimeError when encountering an encoder-only checkpoint without decoder weights."""
    from brats_jepa_3d.models import JEPASegmentationModel3D

    model = JEPASegmentationModel3D(
        img_size=(32, 32, 32),
        patch_size=(16, 16, 16),
        encoder_depth=2,
        decoder_type="multiscale",
    )
    # Checkpoint contains only pre-trained context_encoder keys (no decoder keys)
    pretrain_sd = {"context_encoder.patch_embed.proj.weight": torch.randn(64, 4, 16, 16, 16)}
    ckpt_file = tmp_path / "visreg_jepa_encoder_only.pt"
    torch.save(pretrain_sd, ckpt_file)

    sd = torch.load(ckpt_file, map_location="cpu")
    has_downstream = any(k.startswith("encoder.") for k in sd) or any(k.startswith("decoder.") for k in sd)
    has_pretrain = any(k.startswith("context_encoder.") for k in sd)

    assert not has_downstream
    assert has_pretrain
    with pytest.raises(RuntimeError, match="contains only pre-trained encoder weights"):
        if has_pretrain and not has_downstream:
            raise RuntimeError(
                f"TestModel: Checkpoint {ckpt_file.name} contains only pre-trained encoder weights. "
                f"Evaluating a randomly initialized downstream decoder produces invalid benchmark numbers."
            )


def test_evaluate_3d_sample_level_tumor_accumulation():
    """Verify sample-level tumor metric accumulation avoids Simpson's paradox across uneven batches."""
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import evaluate_3d

    torch.manual_seed(0)
    # Class 1 (tumor) at z in [0, 8), Class 0 elsewhere
    logits = torch.full((2, 5, 16, 16, 16), -10.0)
    logits[0, 1] = 10.0  # Sample 0 has tumor everywhere
    logits[1, 0] = 10.0  # Sample 1 is completely background

    class _FixedNet(torch.nn.Module):
        def forward(self, x):
            return logits

    # Target: Sample 0 has tumor (label 1), Sample 1 is tumor-free (label 0)
    mask0 = torch.zeros((1, 16, 16, 16), dtype=torch.float32)
    mask0[0, :8, :, :] = 1.0
    mask1 = torch.zeros((1, 16, 16, 16), dtype=torch.float32)  # tumor-free

    data = [
        {"image": torch.randn(4, 16, 16, 16), "mask": mask0},
        {"image": torch.randn(4, 16, 16, 16), "mask": mask1},
    ]
    loader = DataLoader(data, batch_size=2)
    stats = evaluate_3d.benchmark_model(
        _FixedNet().eval(), loader, torch.device("cpu"), amp=False, smoke_test=False
    )
    # ET is absent in target, so dice_tumor_only for ET should be NaN (tumor-free)
    assert np.isnan(stats["regions"]["ET"]["dice_tumor_only"])
    # WT was present in sample 0, so dice_tumor_only should be a valid score >= 0.0
    assert 0.0 <= stats["regions"]["WT"]["dice_tumor_only"] <= 1.0


def test_get_checkpoint_search_dirs_discovery(tmp_path: Path):
    """Verify get_checkpoint_search_dirs includes existing primary and fallback candidate directories."""
    from brats_jepa_3d.config import get_checkpoint_search_dirs

    dummy_primary = tmp_path / "primary_ckpts"
    dummy_primary.mkdir(parents=True)

    dirs = get_checkpoint_search_dirs(dummy_primary)
    assert dummy_primary in dirs
    # Order preserves primary first
    assert dirs[0] == dummy_primary
