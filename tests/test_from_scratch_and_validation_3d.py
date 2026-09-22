import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from brats_jepa_3d.config import CHECKPOINTS_DIR, LOGS_DIR, METRICS_DIR, PROJECT_ROOT
from brats_jepa_3d.metrics.volumetric_metrics import (
    compute_hd95_3d,
    compute_volumetric_metrics_3d,
)


def test_volumetric_metrics_fast_validation_bypass():
    """Verify compute_volumetric_metrics_3d with compute_hd95=False executes in milliseconds and returns NaN for HD95."""
    pred = torch.randn(2, 1, 128, 128, 128)
    target = (torch.rand(2, 1, 128, 128, 128) > 0.95).float()

    import time
    t0 = time.perf_counter()
    metrics = compute_volumetric_metrics_3d(pred, target, compute_hd95=False)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.2, f"Validation took {elapsed:.4f}s, expected < 0.2s"
    assert "dice" in metrics
    assert "iou" in metrics
    assert np.isnan(metrics["hd95"]), f"Expected NaN for HD95 when compute_hd95=False, got {metrics['hd95']}"
    assert np.isnan(metrics["hd95_per_sample"][0])


def test_hd95_point_cap_subsampling():
    """Verify compute_hd95_3d handles noisy outlier predictions with 200,000 points without freezing."""
    # Synthetic noisy volume with 200k+ surface points
    vol_p = (np.random.rand(128, 128, 128) > 0.85).astype(np.uint8)
    vol_t = (np.random.rand(128, 128, 128) > 0.85).astype(np.uint8)

    import time
    t0 = time.perf_counter()
    hd95 = compute_hd95_3d(vol_p, vol_t, max_points=10000)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, f"Subsampled HD95 took {elapsed:.4f}s, expected < 0.5s"
    assert isinstance(hd95, float)
    assert hd95 > 0.0


def test_from_scratch_smoke_training(tmp_path):
    """Verify train_downstream_3d.py runs with --from_scratch, creates isolated checkpoint and metrics."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_downstream_3d.py"),
        "--smoke_test",
        "--from_scratch",
        "--model_type", "visreg_jepa",
        "--decoder_type", "multiscale",
        "--epochs", "1",
        "--batch_size", "2",
        "--seed", "42",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert res.returncode == 0, f"Training failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"

    scratch_ckpt = CHECKPOINTS_DIR / "visreg_jepa_multiscale_scratch_best.pt"
    assert scratch_ckpt.exists(), f"Scratch checkpoint not found at {scratch_ckpt}"

    scratch_csv = LOGS_DIR / "visreg_jepa_multiscale_scratch_downstream_metrics.csv"
    assert scratch_csv.exists(), f"Scratch metrics CSV not found at {scratch_csv}"


def test_from_scratch_evaluation():
    """Verify evaluate_3d.py successfully evaluates the --from_scratch model and labels it cleanly."""
    scratch_ckpt = CHECKPOINTS_DIR / "visreg_jepa_multiscale_scratch_best.pt"
    assert scratch_ckpt.exists(), "Scratch checkpoint must exist before evaluation test"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_3d.py"),
        "--smoke_test",
        "--from_scratch",
        "--model_type", "visreg_jepa",
        "--decoder_type", "multiscale",
        "--checkpoint", str(scratch_ckpt),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert res.returncode == 0, f"Evaluation failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    # Smoke runs skip CSV writes by design (--no_write/smoke_test guard), so assert
    # on the printed table instead of the summary file.
    assert "From Scratch" in res.stdout, f"Expected 'From Scratch' in eval output:\n{res.stdout}"


def test_notebook_json_validity():
    """Verify notebooks/04_train_vit_from_scratch_ablation_3d.ipynb is a valid Jupyter notebook."""
    nb_path = PROJECT_ROOT / "notebooks" / "04_train_vit_from_scratch_ablation_3d.ipynb"
    assert nb_path.exists()
    with open(nb_path) as f:
        data = json.load(f)
    assert data["nbformat"] == 4
    assert len(data["cells"]) >= 10
