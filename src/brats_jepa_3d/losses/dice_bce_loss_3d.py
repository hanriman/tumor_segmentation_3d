import torch
import torch.nn.functional as F
from torch import nn


class VolumetricDiceLoss(nn.Module):
    r"""
    Soft Volumetric 3D Dice Loss (Milletari et al., 3DV 2016; V-Net).

    Supports both binary (C=1) and multi-class (C>1) segmentation.
    For multi-class, computes per-class Dice and averages across classes.
    Sums over spatial dimensions only, preserving per-class computation.
    """

    def __init__(self, smooth: float = 1e-5, num_classes: int = 4, squared_pred: bool = True):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.squared_pred = squared_pred

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = logits.shape[1]

        if C == 1:
            # Binary segmentation
            probs = torch.sigmoid(logits)
            targets_bin = (targets > 0).float()
            if targets_bin.dim() == 4:  # [B, D, H, W] -> [B, 1, D, H, W]
                targets_bin = targets_bin.unsqueeze(1)
        else:
            # Multi-class segmentation: softmax over channel dim
            probs = torch.softmax(logits, dim=1)  # [B, C, D, H, W]
            # One-hot encode targets
            if targets.dim() == 5 and targets.shape[1] == 1:
                targets_long = targets[:, 0].long()  # [B, D, H, W]
            elif targets.dim() == 4:
                targets_long = targets.long()  # [B, D, H, W]
            else:
                targets_long = targets.long()
            # Clamp to valid range
            targets_long = targets_long.clamp(0, C - 1)
            targets_bin = F.one_hot(targets_long, num_classes=C)  # [B, D, H, W, C]
            targets_bin = targets_bin.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]

        # Sum over SPATIAL dimensions only (2, 3, 4), preserving batch and class dims
        spatial_dims = (2, 3, 4)
        intersection = (probs * targets_bin).sum(dim=spatial_dims)  # [B, C]
        if self.squared_pred:
            cardinality = (probs.square() + targets_bin.square()).sum(dim=spatial_dims)  # [B, C]
        else:
            cardinality = (probs + targets_bin).sum(dim=spatial_dims)  # [B, C]

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)  # [B, C]
        return 1.0 - dice.mean()  # Mean over batch AND classes


class CombinedDiceBCELoss3D(nn.Module):
    r"""
    Combined Volumetric 3D Dice + Cross-Entropy Loss.
    Supports binary (C=1) and multi-class (C>1) segmentation.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = 1e-5,
        num_classes: int = 4,
        squared_pred: bool = True,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = VolumetricDiceLoss(
            smooth=smooth, num_classes=num_classes, squared_pred=squared_pred
        )
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        dice = self.dice_loss(logits, targets)

        C = logits.shape[1]
        if C == 1:
            targets_bin = (targets > 0).float()
            if targets_bin.dim() == 4:
                targets_bin = targets_bin.unsqueeze(1)
            ce = F.binary_cross_entropy_with_logits(logits, targets_bin)
        else:
            # Cross-entropy expects targets as [B, D, H, W] with class indices
            if targets.dim() == 5 and targets.shape[1] == 1:
                targets_long = targets[:, 0].long()
            elif targets.dim() == 4:
                targets_long = targets.long()
            else:
                targets_long = targets.long()
            targets_long = targets_long.clamp(0, C - 1)
            ce = F.cross_entropy(logits, targets_long)

        total = self.dice_weight * dice + self.bce_weight * ce
        return {
            "loss": total,
            "dice_loss": dice,
            "bce_loss": ce,
        }
