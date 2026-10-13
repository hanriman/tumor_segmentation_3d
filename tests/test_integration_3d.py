"""End-to-end integration and smoke tests for 3D pretraining, baseline training, from-scratch evaluation, and multi-region battery runs."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader

from brats_jepa_3d.config import PROJECT_ROOT


# =========================================================================
# 1. Baseline Training, Checkpointing & Resumption
# =========================================================================


def test_train_nnunet_smoke_and_checkpoints(tmp_path):
    cmd = [
        sys.executable,
        "scripts/train_nnunet_3d.py",
        "--smoke_test",
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--no_amp",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Val HD95: nan mm" not in result.stderr
    assert "Val HD95: nan mm" not in result.stdout

    best_ckpt = Path("outputs/checkpoints/nnunet_3d_best.pt")
    latest_ckpt = Path("outputs/checkpoints/nnunet_3d_latest.pt")
    assert best_ckpt.exists(), "nnunet_3d_best.pt was not saved"
    assert latest_ckpt.exists(), "nnunet_3d_latest.pt was not saved"

    ckpt_data = torch.load(best_ckpt, map_location="cpu")
    assert "model_state_dict" in ckpt_data
    assert "optimizer_state_dict" in ckpt_data
    assert "val_dice" in ckpt_data


def test_train_unet_smoke_and_checkpoints(tmp_path):
    cmd = [
        sys.executable,
        "scripts/train_unet_3d.py",
        "--smoke_test",
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--no_amp",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Val HD95: nan mm" not in result.stderr
    assert "Val HD95: nan mm" not in result.stdout

    best_ckpt = Path("outputs/checkpoints/unet_3d_best.pt")
    latest_ckpt = Path("outputs/checkpoints/unet_3d_latest.pt")
    assert best_ckpt.exists(), "unet_3d_best.pt was not saved"
    assert latest_ckpt.exists(), "unet_3d_latest.pt was not saved"

    ckpt_data = torch.load(best_ckpt, map_location="cpu")
    assert "model_state_dict" in ckpt_data
    assert "optimizer_state_dict" in ckpt_data
    assert "val_dice" in ckpt_data


def test_train_unet_resume(tmp_path):
    latest_ckpt = Path("outputs/checkpoints/unet_3d_latest.pt")
    assert latest_ckpt.exists()

    cmd = [
        sys.executable,
        "scripts/train_unet_3d.py",
        "--smoke_test",
        "--epochs",
        "1",
        "--resume",
        str(latest_ckpt),
        "--no_amp",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Resumed successfully" in result.stderr or "Resumed successfully" in result.stdout


# =========================================================================
# 2. From-Scratch Evaluation & Auto-Discovery Suite
# =========================================================================


@pytest.fixture(scope="module")
def ensure_scratch_checkpoint():
    """Ensures a synthetic visreg_jepa_multiscale_scratch_best.pt exists for smoke tests."""
    ckpt_path = Path("outputs/checkpoints/visreg_jepa_multiscale_scratch_best.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if not ckpt_path.exists():
        from brats_jepa_3d.models import JEPASegmentationModel3D

        model = JEPASegmentationModel3D(
            img_size=(128, 128, 128),
            patch_size=(16, 16, 16),
            in_channels=4,
            embed_dim=384,
            encoder_depth=8,
            num_heads=6,
            mlp_ratio=4.0,
            decoder_type="multiscale",
            deep_supervision=True,
        )
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "val_dice": 0.50,
            },
            ckpt_path,
        )
    return ckpt_path


def test_evaluate_scratch_auto_discovery(ensure_scratch_checkpoint):
    """Verifies evaluate_3d.py resolves the scratch checkpoint without --checkpoint argument."""
    cmd = [
        sys.executable,
        "scripts/evaluate_3d.py",
        "--model_type",
        "visreg_jepa",
        "--decoder_type",
        "multiscale",
        "--from_scratch",
        "--smoke_test",
        "--no_amp",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "3D ViT-FPN (From Scratch)" in result.stderr or "3D ViT-FPN (From Scratch)" in result.stdout
    assert "visreg_jepa_multiscale_scratch_best.pt" in result.stderr or "visreg_jepa_multiscale_scratch_best.pt" in result.stdout


def test_evaluate_low_data_from_scratch():
    """Verifies evaluate_low_data_3d.py supports --from_scratch."""
    cmd = [
        sys.executable,
        "scripts/evaluate_low_data_3d.py",
        "--model_type",
        "visreg_jepa",
        "--from_scratch",
        "--fractions",
        "0.10",
        "--epochs",
        "1",
        "--smoke_test",
        "--no_amp",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "3D ViT-FPN (From Scratch)" in result.stderr or "3D ViT-FPN (From Scratch)" in result.stdout
    assert "Initializing 3D ViT-FPN from SCRATCH" in result.stderr or "Initializing 3D ViT-FPN from SCRATCH" in result.stdout

    summary_csv = Path("outputs/metrics/low_data_3d_summary.csv")
    assert summary_csv.exists()


def test_evaluate_ood_from_scratch(ensure_scratch_checkpoint):
    """Verifies evaluate_ood_3d.py supports --from_scratch."""
    cmd = [
        sys.executable,
        "scripts/evaluate_ood_3d.py",
        "--model_type",
        "visreg_jepa",
        "--from_scratch",
        "--smoke_test",
        "--no_amp",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "3D ViT-FPN (From Scratch)" in result.stderr or "3D ViT-FPN (From Scratch)" in result.stdout

    ood_csv = Path("outputs/metrics/ood_3d_summary.csv")
    assert ood_csv.exists()


def test_notebook_05_schema_and_uniformity():
    """Verifies Notebook 05 (ViT From Scratch) keeps the shared train/eval/low-data/OOD cell layout."""
    nb_path = Path("notebooks/05_train_vit_from_scratch_ablation_3d.ipynb")
    assert nb_path.exists()
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 21

    cell_eval = nb["cells"][13]
    eval_code = "".join(cell_eval["source"])
    assert "--checkpoint" not in eval_code
    assert "--from_scratch" in eval_code

    cell_low_data = nb["cells"][15]
    assert "evaluate_low_data_3d.py" in "".join(cell_low_data["source"])

    cell_ood = nb["cells"][17]
    assert "evaluate_ood_3d.py" in "".join(cell_ood["source"])


def test_notebooks_use_no_removed_cli_flags_or_methods():
    """Notebooks must not reference CLI flags/methods removed in the multiregion-only cleanup."""
    removed = ["--label_protocol", "ijepa", "use_token_queue", "validation_dice_iou_wt"]
    nb_paths = sorted(Path("notebooks").rglob("*.ipynb"))
    assert nb_paths, "no notebooks found"
    for nb_path in nb_paths:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for token in removed:
            assert token not in code, f"{nb_path} still references removed '{token}'"


# =========================================================================
# 3. Multi-Region Evaluation Batteries
# =========================================================================


class _Fixed5(torch.nn.Module):
    """Fixed C=5 logits: argmax paints WT everywhere, ET/TC on known voxels."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        B = x.shape[0]
        base = torch.full((1, 5, 16, 16, 16), -10.0)
        base[0, 2] = 10.0
        base[0, 4, :4, :4, :4] = 20.0
        return base.expand(B, -1, -1, -1, -1) + self.dummy


def _battery_loader(n: int = 2, et_free: bool = False):
    data = []
    for _ in range(n):
        m = torch.full((1, 16, 16, 16), 2.0)
        if not et_free:
            m[0, :4, :4, :4] = 4.0
        data.append({
            "image": torch.randn(4, 16, 16, 16),
            "mask": m,
            "brain_mask": torch.ones(1, 16, 16, 16),
        })
    return DataLoader(data, batch_size=1)


def test_low_data_returns_regions_multiclass():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import evaluate_low_data_3d as lowmod

    torch.manual_seed(0)
    loader = _battery_loader()
    out = lowmod.train_and_eval(
        _Fixed5().eval(), loader, loader, torch.device("cpu"),
        epochs=1, amp=False, smoke_test=True, num_classes=5)
    assert set(out) == {"WT", "TC", "ET", "mean"}, out.keys()
    assert abs(out["mean"] - (out["WT"] + out["TC"] + out["ET"]) / 3.0) < 1e-6
    assert out["WT"] == 1.0 and out["TC"] == 1.0 and out["ET"] == 1.0


def test_ood_returns_regions_and_et_free_graceful():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import evaluate_ood_3d as oodmod

    torch.manual_seed(0)
    out = oodmod.evaluate_perturbation(
        _Fixed5().eval(), _battery_loader(et_free=False), torch.device("cpu"),
        None, amp=False, smoke_test=True)
    assert set(out) == {"WT", "TC", "ET", "mean"}
    assert out["ET"] == 1.0

    out2 = oodmod.evaluate_perturbation(
        _Fixed5().eval(), _battery_loader(et_free=True), torch.device("cpu"),
        None, amp=False, smoke_test=True)
    assert out2["ET"] >= 0.0
