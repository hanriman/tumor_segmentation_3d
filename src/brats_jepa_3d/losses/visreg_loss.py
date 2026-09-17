import math

import torch
import torch.nn.functional as F
from torch import nn

from .ijepa_loss import IJEPALoss


class VisRegLoss(nn.Module):
    r"""
    VISReg Loss: JEPA Prediction Loss + Decoupled Scale & Shape Regularization.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Decoupled Center, Scale, and Shape Regularization (Wu et al., 2026):
       VISReg regularizes latent representations via three principled statistical objectives:
       - **Center Regularization** (\mathcal{L}_{\text{center}}): Forces empirical mean
         representations across the batch to be centered at the origin:
             \mathcal{L}_{\text{center}} = \frac{1}{D} \|\mu_z\|_2^2
       - **Scale Regularization** (\mathcal{L}_{\text{scale}}): Dimension-wise squared penalty
         enforcing standard deviation to match \gamma = 1.0, preventing point collapse:
             \mathcal{L}_{\text{scale}} = \frac{1}{D} \sum_{d=1}^D (1 - \sigma_d)^2
       - **Shape Regularization** (\mathcal{L}_{\text{SWD}}): Sliced Wasserstein Distance against
         analytical Gaussian quantiles. Features are centered and normalized with stop-gradient on
         standard deviation \tilde{z} = (z - \mu) / (\text{sg}(\sigma) + \epsilon), then projected
         onto random hypersphere rays u \sim \mathbb{S}^{D-1}. Crucially, projections are NOT
         standardized along each 1D slice with autograd, which prevents dimensional collapse onto
         oblique hyperplanes and restores high effective representation rank.

    2. AMP Numerical Stability in Quantile Evaluation:
       Gaussian quantiles \Phi^{-1}((i - 0.5) / N) are computed strictly in `torch.float32` before
       being cast to the token dtype, preventing numerical saturation and `inf`/`nan` explosion in
       `torch.erfinv` at distribution tails under AMP `float16`.

    3. Closed-Form 1D Wasserstein Computation:
       The 1D Wasserstein distance between sorted empirical samples and target quantiles has
       a closed-form exact solution. For W_2^2 (squared 2-Wasserstein, Wu et al., 2026):
           W_2^2(P_N, Q) = \frac{1}{N} \sum_{i=1}^N \left(x_{(i)} - \Phi^{-1}\left(\frac{i - 0.5}{N}\right)\right)^2
       and for W_1 (1-Wasserstein):
           W_1(P_N, Q) = \frac{1}{N} \sum_{i=1}^N \left|x_{(i)} - \Phi^{-1}\left(\frac{i - 0.5}{N}\right)\right|
       both computed in O(N \log N) time via sorting without iterative optimization.

    References:
    -----------
    - Wu, Z., Balestriero, R., & Levine, S. (2026). "Visual Representation Learning via Regularization."
      arXiv:2606.02572 (VISReg).
    - Bonneel, N., et al. (2015). "Sliced and Radon transform Wasserstein metrics of distributions."
      Journal of Mathematical Imaging and Vision, 51(1), 22-45.
    - Villani, C. (2009). Optimal Transport: Old and New. Springer.
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        center_weight: float = 1.0,
        scale_weight: float = 1.0,
        swd_weight: float = 1.0,
        num_projections: int = 256,
        target_std: float = 1.0,
        swd_metric: str = "mse",
        scale_loss_type: str = "squared",
    ):
        super().__init__()
        self.jepa_loss = IJEPALoss(loss_type=loss_type)
        self.center_weight = center_weight
        self.scale_weight = scale_weight
        self.swd_weight = swd_weight
        self.num_projections = num_projections
        self.target_std = target_std
        if swd_metric not in ("mse", "l1"):
            raise ValueError(f"Unknown swd_metric: {swd_metric}. Must be 'mse' or 'l1'.")
        self.swd_metric = swd_metric
        if scale_loss_type not in ("squared", "hinge"):
            raise ValueError(
                f"Unknown scale_loss_type: {scale_loss_type}. Must be 'squared' or 'hinge'."
            )
        self.scale_loss_type = scale_loss_type

    def _center_loss(self, z: torch.Tensor) -> torch.Tensor:
        r"""Center regularization: (1/D) * ||mu_z||_2^2."""
        mu = z.mean(dim=0)
        return torch.mean(mu**2)

    def _scale_loss(self, z: torch.Tensor) -> torch.Tensor:
        r"""Scale regularization: enforces coordinate-wise target standard deviation."""
        std_z = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6)
        if self.scale_loss_type == "hinge":
            return torch.mean(F.relu(self.target_std - std_z))
        return torch.mean((self.target_std - std_z) ** 2)

    def _sliced_wasserstein_distance(self, z: torch.Tensor) -> torch.Tensor:
        """Shape regularization: 1D Sliced-Wasserstein distance against standard normal quantiles."""
        N, D = z.shape

        # Center features and scale-normalize with stop-gradient to decouple shape from scale
        mu = z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0, unbiased=False, keepdim=True) + 1e-6)
        z_norm = (z - mu) / (std.detach() + 1e-6)

        # Sample random projection vectors on unit hypersphere in float32
        u = torch.randn(D, self.num_projections, device=z.device, dtype=torch.float32)
        u = F.normalize(u, p=2, dim=0)  # [D, M]

        # 1D projected slices: [N, M] in float32 to prevent FP16 underflow under AMP
        proj = z_norm.float() @ u
        sorted_proj, _ = torch.sort(proj, dim=0)  # [N, M]

        # Analytical standard normal N(0, 1) quantiles: Phi^{-1}((i - 0.5) / N)
        # Evaluated strictly in float32 to prevent float16 erfinv tail saturation under AMP
        probs = (torch.arange(1, N + 1, device=z.device, dtype=torch.float32) - 0.5) / N
        gaussian_quantiles = torch.erfinv(2.0 * probs - 1.0) * math.sqrt(2.0)  # [N] in float32
        target_quantiles = gaussian_quantiles.unsqueeze(-1).expand_as(sorted_proj)

        if self.swd_metric == "mse":
            swd = F.mse_loss(sorted_proj, target_quantiles)
        else:
            swd = F.l1_loss(sorted_proj, target_quantiles)
        return swd.to(dtype=z.dtype)

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
        context_tokens: torch.Tensor | None = None,
        tokens: torch.Tensor | None = None,
        projected_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        j_loss = self.jepa_loss(predictions, targets)

        reg_tokens = (
            tokens
            if tokens is not None
            else (projected_tokens if projected_tokens is not None else context_tokens)
        )
        if reg_tokens is None:
            raise ValueError("VisRegLoss requires regularized token representations.")

        # Flatten across batch and patch dimensions: [N, D]
        z = reg_tokens.reshape(-1, reg_tokens.shape[-1])

        center_loss = self._center_loss(z)
        scale_loss = self._scale_loss(z)
        swd_loss = self._sliced_wasserstein_distance(z)

        total_loss = (
            j_loss
            + self.center_weight * center_loss
            + self.scale_weight * scale_loss
            + self.swd_weight * swd_loss
        )
        return {
            "loss": total_loss,
            "jepa_loss": j_loss,
            "center_loss": center_loss,
            "scale_loss": scale_loss,
            "shape_loss": swd_loss,
            "swd_loss": swd_loss,
        }
