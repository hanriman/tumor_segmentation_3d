from typing import Any
import torch
import torch.nn.functional as F
from torch import nn
from .dice_bce_loss_3d import CombinedDiceBCELoss3D


class DeepSupervisionLoss3D(nn.Module):
    r"""
    Multi-Scale Normalized Deep Supervision Loss (Isensee et al., Nature Methods 2021).

    Theoretical Justification:
    --------------------------
    Deep 3D networks suffer from vanishing gradient flow to intermediate feature extraction layers.
    Attaching auxiliary loss heads at downsampled resolutions (e.g. 64^3, 32^3, 16^3) provides direct
    gradient highways to early convolutional/transformer blocks. Multi-scale heads are weighted using
    normalized exponential decay w_s = 2^{-s} / \sum 2^{-j} so that higher-resolution outputs dominate.
    """
    def __init__(
        self,
        weights: list[float] | None = None,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
    ):
        super().__init__()
        if weights is None:
            # Default normalized weights for 4 multi-scale heads: 128^3, 64^3, 32^3, 16^3
            raw = [1.0 / (2 ** i) for i in range(4)]
            total = sum(raw)
            weights = [w / total for w in raw]
        self.weights = weights
        self.base_loss = CombinedDiceBCELoss3D(dice_weight=dice_weight, bce_weight=bce_weight)

    def forward(
        self,
        multi_scale_logits: list[torch.Tensor] | torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(multi_scale_logits, (list, tuple)):
            return self.base_loss(multi_scale_logits, targets)

        total_loss = torch.tensor(0.0, device=targets.device, dtype=targets.dtype)
        total_dice = torch.tensor(0.0, device=targets.device, dtype=targets.dtype)
        total_bce = torch.tensor(0.0, device=targets.device, dtype=targets.dtype)

        num_heads = len(multi_scale_logits)
        weights = self.weights[:num_heads]
        weight_sum = sum(weights)
        norm_weights = [w / weight_sum for w in weights]

        for logits_s, w in zip(multi_scale_logits, norm_weights):
            target_shape = logits_s.shape[2:]
            if targets.shape[2:] != target_shape:
                # Downsample binary target with nearest-neighbor interpolation
                targets_s = F.interpolate(targets.float(), size=target_shape, mode="nearest")
            else:
                targets_s = targets

            res = self.base_loss(logits_s, targets_s)
            total_loss = total_loss + w * res["loss"]
            total_dice = total_dice + w * res["dice_loss"]
            total_bce = total_bce + w * res["bce_loss"]

        return {
            "loss": total_loss,
            "dice_loss": total_dice,
            "bce_loss": total_bce,
        }
