import torch
import torch.nn.functional as F
from torch import nn

from .deep_supervision_loss_3d import DeepSupervisionLoss3D
from .dice_bce_loss_3d import CombinedDiceBCELoss3D


class VolumetricTverskyLoss(nn.Module):
    r"""
    Asymmetric Volumetric 3D Tversky Loss (Salehi et al., DLMIA 2017).

    Generalizes Dice by weighting False Negatives (beta) and False Positives
    (alpha) asymmetrically:

        TI = TP / (TP + alpha * FP + beta * FN),  L = 1 - TI

    With alpha = beta = 0.5 this reduces exactly to Dice. In intracranial MRI,
    tumors occupy <1.5% of scan volume: missing thin infiltrative margins (FN)
    barely moves Dice yet heavily inflates HD95. beta = 0.7 / alpha = 0.3
    penalizes FN 2.33x more than FP, forcing the decoder to track peripheral
    margins. Binary (C=1) formulation matches our whole-tumor task; multi-class
    inputs fall back to per-class averaging mirroring VolumetricDiceLoss.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        include_background: bool = True,
    ):
        super().__init__()
        if alpha < 0 or beta < 0 or alpha + beta <= 0:
            raise ValueError(f"Invalid Tversky weights alpha={alpha}, beta={beta}")
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = logits.shape[1]
        if C == 1:
            probs = torch.sigmoid(logits)
            targets_bin = (targets > 0).float()
            if targets_bin.dim() == 4:
                targets_bin = targets_bin.unsqueeze(1)
            spatial = (2, 3, 4)
            tp = (probs * targets_bin).sum(dim=spatial)
            fp = (probs * (1.0 - targets_bin)).sum(dim=spatial)
            fn = ((1.0 - probs) * targets_bin).sum(dim=spatial)
            ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            return 1.0 - ti.mean()
        probs = torch.softmax(logits, dim=1)
        if targets.dim() == 5 and targets.shape[1] == 1:
            targets_long = targets[:, 0].long()
        else:
            targets_long = targets.long()
        if targets_long.min() < 0 or targets_long.max() >= C:
            raise ValueError(
                f"Target labels out of range [0, {C - 1}]: "
                f"got min={int(targets_long.min())}, max={int(targets_long.max())}."
            )
        targets_bin = F.one_hot(targets_long, num_classes=C).permute(0, 4, 1, 2, 3).float()
        spatial = (2, 3, 4)
        tp = (probs * targets_bin).sum(dim=spatial)
        fp = (probs * (1.0 - targets_bin)).sum(dim=spatial)
        fn = ((1.0 - probs) * targets_bin).sum(dim=spatial)
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        if not self.include_background:
            ti = ti[:, 1:]
        return 1.0 - ti.mean()


class CombinedTverskyBCEWithLogitsLoss3D(nn.Module):
    r"""
    Combined Volumetric 3D Tversky + BCE Loss.

    Returns the same key contract as CombinedDiceBCELoss3D
    ({"loss", "dice_loss", "bce_loss"}) with the Tversky term reported under
    "dice_loss" (plus an explicit "tversky_loss" alias), so DeepSupervisionLoss3D
    aggregation and training-log formatting work unchanged.

    Note: the BCE term is binary-only (C=1 whole-tumor task); multi-class logits
    raise an explicit ValueError rather than silently miscomputing.
    """

    def __init__(
        self,
        tversky_weight: float = 1.0,
        bce_weight: float = 1.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        self.tversky_loss = VolumetricTverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits.shape[1] != 1:
            raise ValueError(
                f"CombinedTverskyBCEWithLogitsLoss3D is binary-only (C=1); got C={logits.shape[1]}. "
                "Use VolumetricTverskyLoss directly for multi-class logits."
            )
        tversky = self.tversky_loss(logits, targets)
        targets_bin = (targets > 0).float()
        if targets_bin.dim() == 4:
            targets_bin = targets_bin.unsqueeze(1)
        ce = F.binary_cross_entropy_with_logits(logits, targets_bin)
        total = self.tversky_weight * tversky + self.bce_weight * ce
        return {
            "loss": total,
            "dice_loss": tversky,
            "bce_loss": ce,
            "tversky_loss": tversky,
        }


def resolve_seg_loss_type(args: object, cli_args: list[str] | None = None) -> str:
    """Resolves the segmentation loss_type against config-merge collisions.

    JEPA model YAMLs reuse the `loss_type` key for the latent prediction loss
    ("smooth_l1"/"l1"/"mse"), and merge_config_with_args copies it into the
    shared argparse namespace. Precedence: (1) explicitly passed CLI flags —
    read from `args._explicit_cli_flags` when present (set by
    merge_config_with_args), else the `cli_args` parameter; (2) accepted
    segmentation values; (3) "dice_bce" fallback.

    No `sys.argv` sniffing: legacy direct calls without either record log a
    deprecation warning and fall back to the accepted-value check.
    """
    import logging

    explicit: set[str] = set()
    tracked = getattr(args, "_explicit_cli_flags", None)
    if tracked is not None:
        explicit = set(tracked)
    elif cli_args is not None:
        cli = cli_args
        explicit = {
            a.lstrip("-").split("=")[0].replace("-", "_")
            for a in cli
            if a.startswith("-")
        }
    else:
        logging.getLogger(__name__).warning(
            "resolve_seg_loss_type: no explicit-flag record and no cli_args; "
            "falling back to accepted-value check "
            "(deprecated — route trainers through merge_config_with_args)."
        )
    val = getattr(args, "loss_type", "dice_bce")
    if "loss_type" in explicit:
        return val
    return val if val in ("dice_bce", "tversky") else "dice_bce"


def build_segmentation_criterion(
    loss_type: str = "dice_bce",
    deep_supervision: bool = True,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> nn.Module:
    """Shared criterion factory for all supervised 3D trainers (fair-benchmark parity).

    loss_type="dice_bce" (default) preserves the exact pre-existing behavior;
    loss_type="tversky" swaps the overlap term for the asymmetric Tversky loss
    while keeping BCE, DS weighting, and return-key contracts identical.
    """
    if loss_type not in ("dice_bce", "tversky"):
        raise ValueError(f"Unknown loss_type: {loss_type}")
    if loss_type == "tversky":
        base: nn.Module = CombinedTverskyBCEWithLogitsLoss3D(
            alpha=tversky_alpha, beta=tversky_beta
        )
    else:
        base = CombinedDiceBCELoss3D()
    if deep_supervision:
        return DeepSupervisionLoss3D(base_loss=base)
    return base
