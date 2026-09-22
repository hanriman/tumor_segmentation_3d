"""BraTS 2024 region evaluation (WT/TC/ET) derived from multi-class predictions.

Label convention (verified on the raw pool): 0=background, 1=NCR, 2=ED, 3=NET,
4=ET. Regions are overlapping unions evaluated with the exact same guarded
binary machinery as the WT-only protocol — including NaN semantics on
region-missing cohorts (e.g. ET-free volumes, ~15% of the pool).
"""
import torch

from .volumetric_metrics import compute_volumetric_metrics_3d

#: BraTS 2024 region definitions as raw label sets.
BRATS_REGIONS: dict[str, tuple[int, ...]] = {
    "WT": (1, 2, 3, 4),
    "TC": (1, 3, 4),
    "ET": (4,),
}

REGION_ORDER = ("WT", "TC", "ET")


def brats_region_masks_from_logits(
    logits: torch.Tensor,
    regions: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, torch.Tensor]:
    """Argmax → per-region binary masks [B, 1, D, H, W] (float)."""
    regions = regions or BRATS_REGIONS
    pred_labels = logits.argmax(dim=1, keepdim=True)
    out = {}
    for name, labels in regions.items():
        m = torch.zeros_like(pred_labels, dtype=torch.float32)
        for lab in labels:
            m = m + (pred_labels == lab).float()
        out[name] = (m > 0).float()
    return out


def brats_region_masks_from_target(
    target: torch.Tensor,
    regions: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, torch.Tensor]:
    """Integer target [B, 1, D, H, W] → per-region binary masks (float)."""
    regions = regions or BRATS_REGIONS
    t = target.long()
    if t.dim() == 4:
        t = t.unsqueeze(1)
    out = {}
    for name, labels in regions.items():
        m = torch.zeros_like(t, dtype=torch.float32)
        for lab in labels:
            m = m + (t == lab).float()
        out[name] = (m > 0).float()
    return out


def compute_brats_regions_3d(
    logits: torch.Tensor,
    target: torch.Tensor,
    regions: dict[str, tuple[int, ...]] | None = None,
    compute_hd95: bool = True,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, dict]:
    """Evaluates each BraTS region by calling the guarded binary metric suite
    once per region — inheriting NaN semantics, HD95, and nanmean aggregation
    for free, including ET-missing cohorts."""
    regions = regions or BRATS_REGIONS
    pred_masks = brats_region_masks_from_logits(logits, regions)
    tgt_masks = brats_region_masks_from_target(target, regions)
    return {
        name: compute_volumetric_metrics_3d(
            pred_masks[name], tgt_masks[name], from_logits=False,
            compute_hd95=compute_hd95, voxel_spacing=voxel_spacing,
        )
        for name in regions
    }


def validation_dice_iou(
    logits: torch.Tensor, masks: torch.Tensor
) -> tuple[float, float]:
    """Model-selection score honoring output channels: binary path as before,
    multi-class path reports the mean over WT/TC/ET region Dice/IoU.

    WT-only selection is blind to ET collapse (WT is ED-dominated, so WT can
    stay ~87% while ET=0). Averaging all three regions keeps the gate honest;
    per-region scores remain available via compute_brats_regions_3d."""
    if logits.shape[1] == 1:
        m = compute_volumetric_metrics_3d(logits, masks, compute_hd95=False)
        return float(m["dice"]), float(m["iou"])
    res = compute_brats_regions_3d(logits, masks, compute_hd95=False)
    dice = float(sum(float(res[r]["dice"]) for r in ("WT", "TC", "ET")) / 3.0)
    iou = float(sum(float(res[r]["iou"]) for r in ("WT", "TC", "ET")) / 3.0)
    return dice, iou


def validation_dice_iou_wt(
    logits: torch.Tensor, masks: torch.Tensor
) -> tuple[float, float]:
    """Legacy WT-only score (protocol-comparability logging only, NOT for
    checkpoint selection on v2)."""
    if logits.shape[1] == 1:
        m = compute_volumetric_metrics_3d(logits, masks, compute_hd95=False)
        return float(m["dice"]), float(m["iou"])
    res = compute_brats_regions_3d(logits, masks, compute_hd95=False)["WT"]
    return float(res["dice"]), float(res["iou"])
