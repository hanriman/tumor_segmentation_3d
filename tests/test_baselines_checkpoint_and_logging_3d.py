"""
Tests for 3D UNet and nnU-Net baseline runners:
- Fast validation without HD95 computation during training epochs
- Clean checkpointing saving _best.pt and _latest.pt with optimizer/scheduler states
- Resume capability from checkpoint
"""

import subprocess
import sys
from pathlib import Path

import torch


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
