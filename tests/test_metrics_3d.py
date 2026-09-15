import numpy as np
import pytest
import torch
from brats_jepa_3d.metrics import (
    compute_dice_score_3d,
    compute_effective_rank,
    compute_hd95_3d,
    compute_iou_score_3d,
    compute_representation_collapse_metrics,
    compute_volumetric_metrics_3d,
)


def test_compute_dice_and_iou_3d():
    # 1. Identical foreground
    m1 = torch.zeros(1, 1, 32, 32, 32)
    m1[:, :, 10:20, 10:20, 10:20] = 1.0
    dice_id = compute_dice_score_3d(m1, m1, from_logits=False)
    iou_id = compute_iou_score_3d(m1, m1, from_logits=False)
    assert dice_id == 1.0
    assert iou_id == 1.0

    # 2. Completely disjoint
    m2 = torch.zeros_like(m1)
    m2[:, :, 0:5, 0:5, 0:5] = 1.0
    dice_dis = compute_dice_score_3d(m1, m2, from_logits=False)
    assert dice_dis == 0.0

    # 3. Both empty volumes (Guarded zero division per Powers 2011)
    empty = torch.zeros_like(m1)
    dice_empty = compute_dice_score_3d(empty, empty, from_logits=False)
    assert dice_empty == 1.0

    # 4. One empty, other non-empty
    dice_miss = compute_dice_score_3d(empty, m1, from_logits=False)
    assert dice_miss == 0.0


def test_compute_hd95_3d():
    # 1. Identical volumes
    vol_a = np.zeros((32, 32, 32), dtype=np.uint8)
    vol_a[10:20, 10:20, 10:20] = 1
    assert compute_hd95_3d(vol_a, vol_a) == 0.0

    # 2. Both empty
    vol_empty = np.zeros((32, 32, 32), dtype=np.uint8)
    assert compute_hd95_3d(vol_empty, vol_empty) == 0.0

    # 3. Shifted cubes: 10:20 shifted by 3 voxels along z -> 13:23
    vol_b = np.zeros((32, 32, 32), dtype=np.uint8)
    vol_b[13:23, 10:20, 10:20] = 1
    hd95 = compute_hd95_3d(vol_a, vol_b, voxel_spacing=(1.0, 1.0, 1.0))
    assert 2.5 <= hd95 <= 3.5, f"Expected HD95 approx 3.0 mm, got {hd95}"


def test_effective_rank():
    # 1. High rank isotropic matrix
    torch.manual_seed(42)
    N, D = 1000, 64
    z_isotropic = torch.randn(N, D)
    erank_iso = compute_effective_rank(z_isotropic)
    # Effective rank of standard normal N(0, I) should approach D (>= 90% of D)
    assert erank_iso >= 0.85 * D, f"Expected erank >= {0.85*D}, got {erank_iso}"

    # 2. Rank-1 collapsed matrix (all vectors collinear)
    u = torch.randn(1, D)
    scalars = torch.randn(N, 1)
    z_collapsed = scalars @ u
    erank_col = compute_effective_rank(z_collapsed)
    assert erank_col <= 1.2, f"Expected collapsed rank approx 1.0, got {erank_col}"


def test_representation_collapse_metrics():
    torch.manual_seed(42)
    N, D = 200, 32
    z_rand = torch.randn(N, D)
    metrics = compute_representation_collapse_metrics(z_rand)
    assert "effective_rank" in metrics
    assert "avg_cosine_sim_centered" in metrics
    # Centered cosine similarity of random Gaussian vectors should be near 0
    assert abs(metrics["avg_cosine_sim_centered"]) < 0.15


def test_compute_volumetric_metrics_3d():
    preds = torch.randn(2, 1, 32, 32, 32)
    targets = (torch.rand(2, 1, 32, 32, 32) > 0.9).float()

    res = compute_volumetric_metrics_3d(preds, targets)
    assert "dice" in res
    assert "iou" in res
    assert "hd95" in res
    assert len(res["dice_per_sample"]) == 2
