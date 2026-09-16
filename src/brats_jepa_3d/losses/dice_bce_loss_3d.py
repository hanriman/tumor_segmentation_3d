import torch
import torch.nn.functional as F
from torch import nn


class VolumetricDiceLoss(nn.Module):
    r"""
    Soft Volumetric 3D Dice Loss (Milletari et al., 3DV 2016; V-Net).
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets_bin = (targets > 0).float()

        dims = (1, 2, 3, 4) if probs.dim() == 5 else (1, 2, 3)
        intersection = (probs * targets_bin).sum(dim=dims)
        cardinality = (probs.square() + targets_bin.square()).sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedDiceBCELoss3D(nn.Module):
    r"""
    Combined Volumetric 3D Dice + BCE Loss.
    """

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, smooth: float = 1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = VolumetricDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        targets_bin = (targets > 0).float()
        dice = self.dice_loss(logits, targets_bin)
        bce = F.binary_cross_entropy_with_logits(logits, targets_bin)
        total = self.dice_weight * dice + self.bce_weight * bce
        return {
            "loss": total,
            "dice_loss": dice,
            "bce_loss": bce,
        }
