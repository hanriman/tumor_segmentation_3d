"""Evaluation metrics tests: Dice, IoU, HD95, multi-region BraTS evaluation, effective rank, collapse metrics, and tracking."""

import numpy as np
import pytest
import torch

from brats_jepa_3d.metrics import (
    compute_brats_regions_3d,
    compute_dice_score_3d,
    compute_effective_rank,
    compute_hd95_3d,
    compute_iou_score_3d,
    compute_representation_collapse_metrics,
    compute_volumetric_metrics_3d,
)
from brats_jepa_3d.utils.logging import MetricTracker


# =========================================================================
# 1. Volumetric Segmentation Metrics (Dice, IoU, HD95)
# =========================================================================


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

    # 2. Empty target (BraTS challenge protocol: undefined -> NaN)
    vol_empty = np.zeros((32, 32, 32), dtype=np.uint8)
    assert np.isnan(compute_hd95_3d(vol_empty, vol_empty))
    assert np.isnan(compute_hd95_3d(vol_a, vol_empty))

    # 3. Empty prediction with non-empty target -> maximum diagonal penalty
    diag_expected = np.sqrt(32**2 + 32**2 + 32**2)
    assert np.isclose(compute_hd95_3d(vol_empty, vol_a), diag_expected)

    # 4. Shifted cubes: 10:20 shifted by 3 voxels along z -> 13:23
    vol_b = np.zeros((32, 32, 32), dtype=np.uint8)
    vol_b[13:23, 10:20, 10:20] = 1
    hd95 = compute_hd95_3d(vol_a, vol_b, voxel_spacing=(1.0, 1.0, 1.0))
    assert 2.5 <= hd95 <= 3.5, f"Expected HD95 approx 3.0 mm, got {hd95}"


def test_hd95_point_cap_subsampling():
    """Verify compute_hd95_3d handles noisy outlier predictions with 200,000 points without freezing."""
    vol_p = (np.random.rand(128, 128, 128) > 0.85).astype(np.uint8)
    vol_t = (np.random.rand(128, 128, 128) > 0.85).astype(np.uint8)

    import time
    t0 = time.perf_counter()
    hd95 = compute_hd95_3d(vol_p, vol_t, max_points=10000)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, f"Subsampled HD95 took {elapsed:.4f}s, expected < 0.5s"


def test_f5_hd95_per_volume_determinism():
    """Same pair reproducible; different volumes valid; cap keeps it fast."""
    import time

    rng = np.random.default_rng(0)
    a = (rng.random((32, 32, 32)) > 0.7).astype(np.float32)
    b = (rng.random((32, 32, 32)) > 0.7).astype(np.float32)
    h1 = compute_hd95_3d(a, b)
    h2 = compute_hd95_3d(a, b)
    assert h1 == h2
    c = (rng.random((48, 48, 48)) > 0.7).astype(np.float32)
    d = (rng.random((48, 48, 48)) > 0.7).astype(np.float32)
    t0 = time.perf_counter()
    h3 = compute_hd95_3d(c, d)
    assert time.perf_counter() - t0 < 10.0
    assert np.isfinite(h3)


def test_compute_volumetric_metrics_3d():
    preds = torch.randn(2, 1, 32, 32, 32)
    targets = (torch.rand(2, 1, 32, 32, 32) > 0.9).float()

    res = compute_volumetric_metrics_3d(preds, targets)
    assert "dice" in res
    assert "iou" in res
    assert "hd95" in res
    assert len(res["dice_per_sample"]) == 2


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


def test_compute_volumetric_metrics_shape_alignment():
    """Verify compute_volumetric_metrics_3d accepts 4D target with 5D pred and vice versa."""
    pred_5d = torch.randn(2, 1, 16, 16, 16)
    target_4d = (torch.rand(2, 16, 16, 16) > 0.8).float()

    res1 = compute_volumetric_metrics_3d(pred_5d, target_4d)
    assert "dice" in res1
    assert "hd95" in res1
    assert 0.0 <= res1["dice"] <= 1.0

    pred_4d = torch.randn(2, 16, 16, 16)
    target_5d = (torch.rand(2, 1, 16, 16, 16) > 0.8).float()

    res2 = compute_volumetric_metrics_3d(pred_4d, target_5d)
    assert "dice" in res2
    assert "hd95" in res2
    assert 0.0 <= res2["dice"] <= 1.0


def test_compute_volumetric_metrics_validation_and_from_logits():
    """Verify compute_volumetric_metrics_3d enforces single channel input and respects from_logits."""
    pred_multichannel = torch.randn(2, 4, 16, 16, 16)
    target = (torch.rand(2, 1, 16, 16, 16) > 0.5).float()
    with pytest.raises(ValueError, match="expects a binary channel"):
        compute_volumetric_metrics_3d(pred_multichannel, target)

    prob = torch.zeros(1, 1, 8, 8, 8)
    prob[:, :, :4, :, :] = 0.9
    gt = torch.zeros(1, 1, 8, 8, 8)
    gt[:, :, :4, :, :] = 1.0

    res_prob = compute_volumetric_metrics_3d(prob, gt, from_logits=False)
    assert res_prob["dice"] == pytest.approx(1.0, abs=1e-4)

    res_logits = compute_volumetric_metrics_3d(prob, gt, from_logits=True)
    assert res_logits["dice"] == pytest.approx(1.0, abs=1e-4)


def test_volumetric_metrics_unbatched_3d():
    """Verify compute_volumetric_metrics_3d accepts 3D tensors [D, H, W] without treating D as batch size."""
    pred = torch.randn(16, 16, 16)
    target = (torch.rand(16, 16, 16) > 0.8).float()

    metrics = compute_volumetric_metrics_3d(pred, target)
    assert "dice" in metrics
    assert "hd95" in metrics
    assert len(metrics["dice_per_sample"]) == 1
    assert 0.0 <= metrics["dice"] <= 1.0


def test_p1_step8_hd95_tumor_only_nan_semantics():
    """Tumor-only aggregates are nan (not 1.0/0.0) when HD95 is off or cohort tumor-free."""
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16, 16)
    masks = torch.zeros(2, 1, 16, 16, 16)
    res = compute_volumetric_metrics_3d(logits, masks, compute_hd95=False)
    assert res["hd95_tumor_only"] != res["hd95_tumor_only"], "must be nan when HD95 off"
    assert res["dice_tumor_only"] != res["dice_tumor_only"], "must be nan when cohort tumor-free"
    assert res["iou_tumor_only"] != res["iou_tumor_only"], "must be nan when cohort tumor-free"
    res2 = compute_volumetric_metrics_3d(logits, masks, compute_hd95=True)
    assert res2["hd95_tumor_only"] != res2["hd95_tumor_only"], "must be nan on tumor-free cohort"
    assert res2["dice_tumor_only"] != res2["dice_tumor_only"], "must be nan on tumor-free cohort"
    assert res2["iou_tumor_only"] != res2["iou_tumor_only"], "must be nan on tumor-free cohort"


# =========================================================================
# 2. Multi-Region Evaluation (WT / TC / ET)
# =========================================================================


def _hand_built_multiregion():
    logits = torch.full((1, 5, 8, 8, 8), -10.0)
    # class 1 (NEC/NCR) at z in [0, 2)
    logits[0, 1, 0:2, :, :] = 10.0
    # class 2 (ED) at z in [2, 4)
    logits[0, 2, 2:4, :, :] = 10.0
    # class 4 (ET) at z in [4, 6)
    logits[0, 4, 4:6, :, :] = 10.0
    # background elsewhere (class 0)
    logits[0, 0, 6:8, :, :] = 10.0

    target = torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32)
    target[0, 0, 0:2, :, :] = 1.0
    target[0, 0, 2:4, :, :] = 2.0
    target[0, 0, 4:6, :, :] = 4.0
    return logits, target


def test_region_metrics_perfect_and_et_free_nan():
    logits, target = _hand_built_multiregion()
    res = compute_brats_regions_3d(logits, target, compute_hd95=False)
    assert res["WT"]["dice"] == 1.0 and res["TC"]["dice"] == 1.0 and res["ET"]["dice"] == 1.0
    # ET-free cohort: ET dice falls back gracefully, ET HD95 is NaN (never 0.0).
    tgt_noet = target.clone()
    tgt_noet[tgt_noet == 4] = 2.0
    res2 = compute_brats_regions_3d(logits, tgt_noet, compute_hd95=True)
    assert res2["ET"]["hd95_tumor_only"] != res2["ET"]["hd95_tumor_only"]
    assert res2["WT"]["dice"] > 0.0


# =========================================================================
# 3. Representation Collapse & Effective Rank Metrics
# =========================================================================


def test_effective_rank():
    # 1. High rank isotropic matrix
    torch.manual_seed(42)
    N, D = 1000, 64
    z_isotropic = torch.randn(N, D)
    erank_iso = compute_effective_rank(z_isotropic)
    assert erank_iso >= 0.85 * D, f"Expected erank >= {0.85 * D}, got {erank_iso}"

    # 2. Rank-1 collapsed matrix (all vectors collinear)
    u = torch.randn(1, D)
    scalars = torch.randn(N, 1)
    z_collapsed = scalars @ u
    erank_col = compute_effective_rank(z_collapsed)
    assert erank_col <= 1.2, f"Expected collapsed rank approx 1.0, got {erank_col}"


def test_effective_rank_identity_covariance():
    """Effective rank of N(0, I_D) should approach D."""
    torch.manual_seed(0)
    D = 32
    z = torch.randn(2000, D)
    erank = compute_effective_rank(z)
    assert erank >= 0.85 * D, f"Expected erank >= {0.85 * D} for identity cov, got {erank}"


def test_effective_rank_known_rank():
    """Effective rank of a rank-k matrix should be close to k."""
    torch.manual_seed(0)
    N, D, k = 500, 64, 5
    U = torch.randn(N, k)
    V = torch.randn(k, D)
    z = U @ V
    erank = compute_effective_rank(z)
    assert erank <= k + 1.5, f"Expected erank close to {k}, got {erank}"
    assert erank >= k - 1.5, f"Expected erank close to {k}, got {erank}"


def test_effective_rank_3d_tensor_bound():
    """Effective rank of a 3D batched token tensor must respect erank <= D."""
    torch.manual_seed(42)
    B, T, D = 2, 512, 384
    z_3d = torch.randn(B, T, D)

    erank_3d = compute_effective_rank(z_3d)
    erank_2d = compute_effective_rank(z_3d.reshape(-1, D))

    assert erank_3d <= D, f"Effective rank {erank_3d} exceeds feature dimension D={D}!"
    assert abs(erank_3d - erank_2d) < 1e-4, f"Mismatch: {erank_3d} vs {erank_2d}"
    assert erank_3d >= 0.80 * D, f"Expected isotropic tokens to have erank >= {0.80 * D}, got {erank_3d}"


def test_effective_rank_half_precision():
    """Verify compute_effective_rank and collapse metrics do not fail with float16 or bfloat16 inputs."""
    z_half = torch.randn(2, 64, 128, dtype=torch.float16)
    erank = compute_effective_rank(z_half)
    assert isinstance(erank, float)
    assert erank > 0.0

    metrics = compute_representation_collapse_metrics(z_half)
    assert "effective_rank" in metrics
    assert metrics["effective_rank"] > 0.0

    z_bf16 = torch.randn(2, 64, 128, dtype=torch.bfloat16)
    erank_bf = compute_effective_rank(z_bf16)
    assert isinstance(erank_bf, float)
    assert erank_bf > 0.0


def test_representation_collapse_metrics():
    torch.manual_seed(42)
    N, D = 200, 32
    z_rand = torch.randn(N, D)
    metrics = compute_representation_collapse_metrics(z_rand)
    assert "effective_rank" in metrics
    assert "avg_cosine_sim_centered" in metrics
    assert abs(metrics["avg_cosine_sim_centered"]) < 0.15


def test_representation_collapse_detection():
    """Collapsed representations should show high uncentered cosine similarity and low effective rank."""
    torch.manual_seed(42)
    u = torch.randn(1, 32)
    u = u / torch.norm(u)

    # 1. Directional collapse: representations lie along a single 1D ray
    scalars = torch.randn(100, 1) + 10.0
    z_directional = scalars @ u
    metrics_dir = compute_representation_collapse_metrics(z_directional)
    assert metrics_dir["avg_cosine_sim"] > 0.95
    assert metrics_dir["effective_rank"] <= 1.5

    # 2. Point collapse: all representations identical
    z_point = u.repeat(100, 1)
    metrics_pt = compute_representation_collapse_metrics(z_point)
    assert metrics_pt["feature_variance"] < 1e-6
    assert metrics_pt["effective_rank"] <= 1.5


# =========================================================================
# 4. Metric Tracking & CSV Serialization
# =========================================================================


def test_metric_tracker_ragged_epoch_metrics_csv_serialization(tmp_path):
    """Verify MetricTracker handles uneven/ragged keys across epochs and serializes to CSV without ValueError."""
    tracker = MetricTracker()

    tracker.update({
        "epoch": 1,
        "loss": 0.2086,
        "val_loss": 0.1843,
        "train_jepa_loss": 0.0881,
        "train_center_loss": 0.0166,
        "epoch_duration_sec": 53.04,
    })

    tracker.update({
        "epoch": 2,
        "loss": 0.1315,
        "val_loss": 0.1520,
        "train_jepa_loss": 0.0750,
        "train_center_loss": 0.0142,
        "epoch_duration_sec": 51.20,
    })

    tracker.update({
        "epoch": 3,
        "loss": 0.1240,
        "val_loss": 0.1410,
        "train_jepa_loss": 0.0690,
        "train_center_loss": 0.0125,
        "epoch_duration_sec": 52.10,
        "effective_rank": 53.05,
        "avg_cosine_sim_centered": 0.012,
        "feature_variance": 0.985,
    })

    df = tracker.to_dataframe()
    assert len(df) == 3
    assert "effective_rank" in df.columns
    assert df["effective_rank"].isna().sum() == 2
    assert df.loc[df["epoch"] == 3, "effective_rank"].values[0] == pytest.approx(53.05)

    csv_file = tmp_path / "metrics.csv"
    tracker.save_csv(csv_file)
    assert csv_file.exists()

    import pandas as pd
    loaded_df = pd.read_csv(csv_file)
    assert len(loaded_df) == 3
    assert "train_center_loss" in loaded_df.columns
    assert "effective_rank" in loaded_df.columns
