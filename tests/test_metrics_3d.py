import numpy as np
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
    assert erank_iso >= 0.85 * D, f"Expected erank >= {0.85 * D}, got {erank_iso}"

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


def test_effective_rank_identity_covariance():
    """Effective rank of N(0, I_D) should approach D."""
    torch.manual_seed(0)
    D = 32
    z = torch.randn(2000, D)  # Large sample from N(0, I_D)
    erank = compute_effective_rank(z)
    assert erank >= 0.85 * D, f"Expected erank >= {0.85 * D} for identity cov, got {erank}"


def test_effective_rank_known_rank():
    """Effective rank of a rank-k matrix should be close to k."""
    torch.manual_seed(0)
    N, D, k = 500, 64, 5
    # Create rank-k data: N samples in a k-dimensional subspace
    U = torch.randn(N, k)
    V = torch.randn(k, D)
    z = U @ V
    erank = compute_effective_rank(z)
    assert erank <= k + 1.5, f"Expected erank close to {k}, got {erank}"
    assert erank >= k - 1.5, f"Expected erank close to {k}, got {erank}"


def test_representation_collapse_detection():
    """Collapsed representations should show high uncentered cosine similarity and low effective rank."""
    torch.manual_seed(42)
    u = torch.randn(1, 32)
    u = u / torch.norm(u)

    # 1. Directional collapse: representations lie along a single 1D ray
    scalars = torch.randn(100, 1) + 10.0
    z_directional = scalars @ u
    metrics_dir = compute_representation_collapse_metrics(z_directional)
    assert metrics_dir["avg_cosine_sim"] > 0.95, (
        f"Expected high uncentered cosine sim for directional collapse, got {metrics_dir['avg_cosine_sim']}"
    )
    assert metrics_dir["effective_rank"] <= 1.5, (
        f"Expected effective rank <= 1.5 for 1D collapse, got {metrics_dir['effective_rank']}"
    )

    # 2. Point collapse: all representations identical
    z_point = u.repeat(100, 1)
    metrics_pt = compute_representation_collapse_metrics(z_point)
    assert metrics_pt["feature_variance"] < 1e-6
    assert metrics_pt["effective_rank"] <= 1.5


def test_effective_rank_3d_tensor_bound():
    """Effective rank of a 3D batched token tensor must respect erank <= D."""
    torch.manual_seed(42)
    B, T, D = 2, 512, 384
    z_3d = torch.randn(B, T, D)

    erank_3d = compute_effective_rank(z_3d)
    erank_2d = compute_effective_rank(z_3d.reshape(-1, D))

    # 1. Theoretical bound: erank <= min(N, D) = 384
    assert erank_3d <= D, f"Effective rank {erank_3d} exceeds feature dimension D={D}!"

    # 2. Consistency: 3D input should equal flattened 2D input
    assert abs(erank_3d - erank_2d) < 1e-4, f"Mismatch: {erank_3d} vs {erank_2d}"

    # 3. For standard normal isotropic tokens, erank should be large (>= 0.80 * D for D/N=0.375)
    assert erank_3d >= 0.80 * D, f"Expected isotropic tokens to have erank >= {0.80 * D}, got {erank_3d}"

