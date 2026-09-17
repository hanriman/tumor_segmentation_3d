from typing import Any

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def compute_dice_score_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 0.0,
    from_logits: bool = True,
) -> float:
    r"""
    Sørensen-Dice Similarity Coefficient (DSC) for Volumetric 3D Binary Segmentation.

    Mathematical Rationale & Defense Context (Dice, 1945; Powers, 2011):
    --------------------------------------------------------------------
    Evaluated globally across all voxels in the 3D volume [D, H, W]:
        \text{DSC} = \begin{cases}
            1.0, & \text{if } |P| + |T| = 0 \text{ (both prediction and ground truth empty)} \\
            \frac{2 |P \cap T|}{|P| + |T|}, & \text{otherwise}
        \end{cases}
    Guarded zero-division eliminates artificial epsilon score inflation on empty non-tumor volumes.
    """
    target_bin = (target > 0).float()
    if from_logits:
        pred_bin = (torch.sigmoid(pred) > threshold).float()
    else:
        pred_bin = (pred > threshold).float()

    intersection = (pred_bin * target_bin).sum().item()
    cardinality = pred_bin.sum().item() + target_bin.sum().item()

    if cardinality == 0:
        return 1.0
    if smooth > 0.0:
        return float((2.0 * intersection + smooth) / (cardinality + smooth))
    return float((2.0 * intersection) / cardinality)


def compute_iou_score_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 0.0,
    from_logits: bool = True,
) -> float:
    r"""Volumetric 3D Intersection over Union (Jaccard Index) with Zero-Division Guard."""
    target_bin = (target > 0).float()
    if from_logits:
        pred_bin = (torch.sigmoid(pred) > threshold).float()
    else:
        pred_bin = (pred > threshold).float()

    intersection = (pred_bin * target_bin).sum().item()
    union = (pred_bin + target_bin - pred_bin * target_bin).sum().item()

    if union == 0:
        return 1.0
    if smooth > 0.0:
        return float((intersection + smooth) / (union + smooth))
    return float(intersection / union)


def compute_hd95_3d(
    pred_vol_3d: np.ndarray,
    target_vol_3d: np.ndarray,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    r"""
    Exact 95th Percentile Symmetric Hausdorff Distance (HD95) for 3D binary volumes [D, H, W].

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Physical Metric Measurement (Huttenlocher et al., 1993; Taha & Hanbury, 2015):
       Evaluated in true physical millimeters using voxel spacing (s_z, s_y, s_x):
           d_H(P, T) = \max \left\{ P_{95\%} \min_{t \in \partial T} \|p - t\|_2, \; P_{95\%} \min_{p \in \partial P} \|t - p\|_2 \right\}
       Surface boundaries \partial P, \partial T are extracted via 3D 26-connectivity morphological erosion.
       Nearest-neighbor Euclidean queries run in O(K log K) time via scipy.spatial.cKDTree.
    2. Guarded Failure Edge Cases:
       - Identical volumes (both empty or identical mask) -> HD95 = 0.0 mm.
       - Complete failure (one empty, other non-empty) -> penalized with the maximum spatial 3D diagonal:
         \sqrt{(D s_z)^2 + (H s_y)^2 + (W s_x)^2}.
    """
    p_mask = (pred_vol_3d > 0).astype(bool)
    t_mask = (target_vol_3d > 0).astype(bool)

    if np.array_equal(p_mask, t_mask):
        return 0.0

    d, h, w = p_mask.shape
    scale = np.array(voxel_spacing, dtype=np.float64)
    diag = float(np.sqrt((d * scale[0]) ** 2 + (h * scale[1]) ** 2 + (w * scale[2]) ** 2))

    # 3D 26-connectivity morphological erosion
    struct_26 = np.ones((3, 3, 3), dtype=bool)

    eroded_p = binary_erosion(p_mask, structure=struct_26)
    pts_p = np.argwhere(p_mask ^ eroded_p)
    if len(pts_p) == 0:
        pts_p = np.argwhere(p_mask)

    eroded_t = binary_erosion(t_mask, structure=struct_26)
    pts_t = np.argwhere(t_mask ^ eroded_t)
    if len(pts_t) == 0:
        pts_t = np.argwhere(t_mask)

    if len(pts_p) == 0 or len(pts_t) == 0:
        # One volume is empty while other is non-empty -> maximum penalty
        return diag

    # Scale to physical millimeters
    scaled_pts_p = pts_p.astype(np.float64) * scale
    scaled_pts_t = pts_t.astype(np.float64) * scale

    tree_t = cKDTree(scaled_pts_t)
    d_p2t, _ = tree_t.query(scaled_pts_p)

    tree_p = cKDTree(scaled_pts_p)
    d_t2p, _ = tree_p.query(scaled_pts_t)

    hd95 = float(max(np.percentile(d_p2t, 95), np.percentile(d_t2p, 95)))
    return hd95


def compute_volumetric_metrics_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 0.0,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    from_logits: bool = True,
) -> dict[str, Any]:
    r"""
    Comprehensive Macro-Averaged 3D Volumetric Segmentation Benchmark Suite.

    Technical Design & Verification Guarantees:
    -------------------------------------------
    1. Single-Channel Whole Tumor Validation:
       Strictly enforces binary single-channel predictions (`C=1`). In multi-class tensors
       (`C > 1`), background channel 0 accounts for >98.5% of total intracranial voxels.
       Evaluating multi-class tensors without explicit channel extraction would silently
       macro-average over the dominant background, yielding artificially inflated Dice scores.
       An informative ValueError is raised if `pred.shape[1] > 1`.
    2. Probability vs. Logit Decoupling (`from_logits`):
       When `from_logits=True` (default for raw model outputs), `torch.sigmoid` maps unconstrained
       logits to [0, 1] probabilities before thresholding. When `from_logits=False` (e.g. for
       pre-softmax probabilities or ensembled masks), direct thresholding is applied, preventing
       double-sigmoid distortion where sigmoid(0.9) = 0.71.
    3. Anisotropic Voxel Spacing:
       HD95 query points are scaled to physical millimeters using `voxel_spacing` (default 1.0mm^3).
    4. Guarded Zero-Division Protocol:
       Follows Powers (2011) and Taha & Hanbury (2015): if both prediction and target are empty,
       Dice=1.0; if one is empty while the other is non-empty, Dice=0.0 and HD95=bounding box diagonal.
    """
    # Normalize unbatched 3D volumes [D, H, W] and unchannelled 4D [B, D, H, W] to 5D [B, 1, D, H, W]
    if pred.dim() == 3:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 4:
        pred = pred.unsqueeze(1)

    if target.dim() == 3:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.dim() == 4:
        target = target.unsqueeze(1)

    # Technical Guard: Multi-class tensors must be binarized/sub-indexed prior to metric computation
    if pred.dim() == 5 and pred.shape[1] > 1:
        raise ValueError(
            f"compute_volumetric_metrics_3d expects binary Whole Tumor channel (C=1). "
            f"Got pred with shape {pred.shape}. For multi-class evaluation, extract or binarize target channel."
        )

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    # Technical Decision: Apply sigmoid only when from_logits=True
    if from_logits:
        pred_bin = (torch.sigmoid(pred) > threshold).float()
    else:
        pred_bin = (pred > threshold).float()
    target_bin = (target > 0).float()

    dice_vals = []
    iou_vals = []
    precision_vals = []
    recall_vals = []
    hd95_vals = []
    has_tumor_vals = []

    p_np = pred_bin.detach().cpu().numpy()
    t_np = target_bin.detach().cpu().numpy()

    B = pred.shape[0]
    for b in range(B):
        p_b = pred_bin[b]
        t_b = target_bin[b]

        tp = (p_b * t_b).sum().item()
        fp = (p_b * (1.0 - t_b)).sum().item()
        fn = ((1.0 - p_b) * t_b).sum().item()

        has_tumor = bool(tp + fn > 0)
        has_tumor_vals.append(has_tumor)

        # Guarded 3D Dice
        if 2.0 * tp + fp + fn > 0:
            dice_vals.append((2.0 * tp) / (2.0 * tp + fp + fn))
        else:
            dice_vals.append(1.0)

        # Guarded 3D IoU
        if tp + fp + fn > 0:
            iou_vals.append(tp / (tp + fp + fn))
        else:
            iou_vals.append(1.0)

        # Guarded 3D Precision (Powers, 2011)
        if tp + fp > 0:
            precision_vals.append(tp / (tp + fp))
        else:
            precision_vals.append(1.0 if fn == 0 else 0.0)

        # Guarded 3D Recall (Powers, 2011)
        if tp + fn > 0:
            recall_vals.append(tp / (tp + fn))
        else:
            recall_vals.append(1.0 if fp == 0 else 0.0)

        # Exact 3D HD95 in physical millimeters
        p_vol = p_np[b, 0] if p_np.ndim == 5 else p_np[b]
        t_vol = t_np[b, 0] if t_np.ndim == 5 else t_np[b]
        hd95_val = compute_hd95_3d(p_vol, t_vol, voxel_spacing=voxel_spacing)
        hd95_vals.append(hd95_val)

    tumor_indices = [i for i, h in enumerate(has_tumor_vals) if h]
    dice_tumor = float(np.mean([dice_vals[i] for i in tumor_indices])) if tumor_indices else 1.0
    iou_tumor = float(np.mean([iou_vals[i] for i in tumor_indices])) if tumor_indices else 1.0
    hd95_tumor = float(np.mean([hd95_vals[i] for i in tumor_indices])) if tumor_indices else 0.0

    return {
        "dice": float(np.mean(dice_vals)),
        "iou": float(np.mean(iou_vals)),
        "precision": float(np.mean(precision_vals)),
        "recall": float(np.mean(recall_vals)),
        "hd95": float(np.mean(hd95_vals)),
        "dice_tumor_only": dice_tumor,
        "iou_tumor_only": iou_tumor,
        "hd95_tumor_only": hd95_tumor,
        "dice_per_sample": dice_vals,
        "iou_per_sample": iou_vals,
        "hd95_per_sample": hd95_vals,
    }
