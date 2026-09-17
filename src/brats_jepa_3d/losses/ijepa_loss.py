import torch
import torch.nn.functional as F
from torch import nn


class IJEPALoss(nn.Module):
    r"""
    Latent Space Prediction Loss for 3D I-JEPA.

    Mathematical Rationale (Assran et al., CVPR 2023; Huber, 1964):
    ---------------------------------------------------------------
    Computes Smooth L1 (Huber) loss between predicted target representations \hat{y}_{\text{tgt}}
    and target representations y_{\text{tgt}} normalized via LayerNorm:
        \mathcal{L} = \frac{1}{M} \sum_{m=1}^M \text{SmoothL1}\left(\hat{y}_{\text{tgt}}^{(m)}, \, \text{LayerNorm}(y_{\text{tgt}}^{(m)})\right)
    Smooth L1 provides quadratic L2 convergence near zero while maintaining robust L1 gradient
    bounds on large target deviations, preventing outlier representations from destabilizing training.
    """

    def __init__(self, loss_type: str = "smooth_l1", beta: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> torch.Tensor:
        if len(predictions) != len(targets):
            raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(targets)} targets")

        total_loss = torch.tensor(0.0, device=predictions[0].device, dtype=predictions[0].dtype)

        for pred, tgt in zip(predictions, targets):
            # Apply LayerNorm to target representations to stabilize target scale
            tgt_norm = F.layer_norm(tgt.detach(), (tgt.shape[-1],))

            if self.loss_type == "smooth_l1":
                block_loss = F.smooth_l1_loss(pred, tgt_norm, beta=self.beta)
            elif self.loss_type == "l1":
                block_loss = F.l1_loss(pred, tgt_norm)
            elif self.loss_type == "mse":
                block_loss = F.mse_loss(pred, tgt_norm)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            total_loss = total_loss + block_loss

        return total_loss / len(predictions)
