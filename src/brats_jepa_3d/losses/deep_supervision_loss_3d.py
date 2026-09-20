import torch
import torch.nn.functional as F
from torch import nn

from .dice_bce_loss_3d import CombinedDiceBCELoss3D


class DeepSupervisionLoss3D(nn.Module):
    r"""
    Multi-Scale Normalized Deep Supervision Loss (Isensee et al., Nature Methods 2021).

    Theoretical Justification & Design Decisions:
    ---------------------------------------------
    1. Gradient Highways in 3D Dense Architectures:
       Deep 3D volumetric networks suffer from vanishing gradient flow to intermediate feature extraction
       layers due to extensive depth and volumetric downsampling. Attaching auxiliary loss heads at
       downsampled resolutions (e.g. 64^3, 32^3, 16^3) provides direct gradient highways to early
       convolutional and transformer blocks.
    2. Dynamic Normalized Exponential Decay:
       Multi-scale heads are weighted using normalized exponential decay:
           w_s = \frac{2^{-s}}{\sum_{j=0}^{S-1} 2^{-j}}
       so that higher-resolution outputs dominate the loss while ensuring total loss scale invariance
       (\sum w_s = 1.0) regardless of the number of active heads S. Rather than fixing a static array,
       `get_weights(num_heads)` dynamically adapts to any head count S >= 1 (e.g., 4-head FPN or
       variable-head DynUNet), preventing silent loss truncation of deeper auxiliary heads.
    3. Spatial Nearest-Neighbor Downsampling:
       Ground truth volumetric masks are adaptively resampled to intermediate head spatial shapes
       using nearest-neighbor interpolation, preserving discrete binary/integer label semantics.
    """

    def __init__(
        self,
        weights: list[float] | None = None,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        squared_pred: bool = True,
        include_background: bool = True,
        base_loss: nn.Module | None = None,
    ):
        super().__init__()
        self.weights = weights
        # Custom overlap loss (e.g. CombinedTverskyBCEWithLogitsLoss3D) must honor
        # the {"loss", "dice_loss", "bce_loss"} return-key contract.
        self.base_loss = (
            base_loss
            if base_loss is not None
            else CombinedDiceBCELoss3D(
                dice_weight=dice_weight,
                bce_weight=bce_weight,
                squared_pred=squared_pred,
                include_background=include_background,
            )
        )

    def get_weights(self, num_heads: int) -> list[float]:
        r"""
        Computes dynamically normalized exponential decay weights w_s = 2^{-s} / \sum 2^{-j}
        for any number of multi-scale heads (Isensee et al., Nature Methods 2021).
        If explicit custom weights were supplied at initialization, they are utilized
        and normalized over the active heads.
        """
        if num_heads <= 0:
            return []
        if self.weights is not None:
            if len(self.weights) >= num_heads:
                w = list(self.weights[:num_heads])
            else:
                w = list(self.weights) + [1.0 / (2**i) for i in range(len(self.weights), num_heads)]
        else:
            w = [1.0 / (2**i) for i in range(num_heads)]
        total = sum(w)
        return [x / total for x in w]

    def forward(
        self,
        multi_scale_logits: list[torch.Tensor] | torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(multi_scale_logits, (list, tuple)):
            return self.base_loss(multi_scale_logits, targets)

        if targets.dim() == 4:
            targets = targets.unsqueeze(1)

        dtype = multi_scale_logits[0].dtype if len(multi_scale_logits) > 0 else torch.float32
        total_loss = torch.tensor(0.0, device=targets.device, dtype=dtype)
        total_dice = torch.tensor(0.0, device=targets.device, dtype=dtype)
        total_bce = torch.tensor(0.0, device=targets.device, dtype=dtype)

        num_heads = len(multi_scale_logits)
        if num_heads == 0:
            return {
                "loss": total_loss,
                "dice_loss": total_dice,
                "bce_loss": total_bce,
            }

        norm_weights = self.get_weights(num_heads)

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
