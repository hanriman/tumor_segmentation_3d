"""BraTS 2024 Post-Treatment Glioma (GLI) region evaluation (WT/TC/ET) derived from multi-class predictions.

Official BraTS 2024 Post-Treatment GLI label convention (de Verdier et al., 2024, arXiv:2405.18368, Section 2.2):
  0 = Background
  1 = NETC (Non-Enhancing Tumor Core)
  2 = SNFH (Surrounding Non-Enhancing FLAIR Hyperintensity)
  3 = ET   (Enhancing Tissue)
  4 = RC   (Resection Cavity — post-surgical fluid cavity, excluded from WT/TC/ET)

Regions are overlapping unions evaluated with the guarded binary metric suite,
including NaN semantics on region-missing cohorts (e.g. ET-free volumes).
"""
import torch

from .volumetric_metrics import compute_volumetric_metrics_3d

#: Official BraTS 2024 Post-Treatment GLI region definitions as raw label sets.
BRATS_REGIONS: dict[str, tuple[int, ...]] = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
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
    """Model-selection score: reports the mean over WT/TC/ET region Dice/IoU.

    Averaging all three regions keeps checkpoint selection honest;
    per-region scores remain available via compute_brats_regions_3d."""
    res = compute_brats_regions_3d(logits, masks, compute_hd95=False)
    dice = float(sum(float(res[r]["dice"]) for r in ("WT", "TC", "ET")) / 3.0)
    iou = float(sum(float(res[r]["iou"]) for r in ("WT", "TC", "ET")) / 3.0)
    return dice, iou

