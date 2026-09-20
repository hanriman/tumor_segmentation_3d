"""
Tests for ViT From-Scratch evaluation across all benchmark dimensions:
- Held-out test split evaluation with auto-checkpoint discovery (no --checkpoint argument required)
- Low-data volumetric label efficiency evaluation (--from_scratch)
- Out-of-distribution (OOD) scanner shift robustness evaluation (--from_scratch)
- Notebook 04 structure and JSON validity
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


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


def test_notebook_04_schema_and_uniformity():
    """Verifies Notebook 04 follows the exact 20-cell pipeline of Notebooks 01, 02, and 03."""
    nb_path = Path("notebooks/04_train_vit_from_scratch_ablation_3d.ipynb")
    assert nb_path.exists()
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 20

    # Ensure no hardcoded --checkpoint flag in evaluation cell
    cell_eval = nb["cells"][13]
    eval_code = "".join(cell_eval["source"])
    assert "--checkpoint" not in eval_code
    assert "--from_scratch" in eval_code

    # Ensure low data and ood cells exist
    cell_low_data = nb["cells"][15]
    assert "evaluate_low_data_3d.py" in "".join(cell_low_data["source"])

    cell_ood = nb["cells"][17]
    assert "evaluate_ood_3d.py" in "".join(cell_ood["source"])
