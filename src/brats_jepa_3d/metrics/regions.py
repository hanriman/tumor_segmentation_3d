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
) -> dict[str, dict]:
    """Evaluates each BraTS region by calling the guarded binary metric suite
    once per region — inheriting NaN semantics, HD95, and nanmean aggregation
    for free, including ET-missing cohorts."""
    regions = regions or BRATS_REGIONS
    pred_masks = brats_region_masks_from_logits(logits, regions)
    tgt_masks = brats_region_masks_from_target(target, regions)
    return {
        name: compute_volumetric_metrics_3d(
            pred_masks[name], tgt_masks[name], from_logits=False, compute_hd95=compute_hd95
        )
        for name in regions
    }
