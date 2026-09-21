import math

import torch
import torch.nn.functional as F
from torch import nn

from .ijepa_loss import IJEPALoss


class EppsPulleyGaussianityTest(nn.Module):
    r"""
    Epps-Pulley Goodness-of-Fit Test Statistic for 1D Gaussianity.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Cramér-Wold Theorem (1936):
       A d-dimensional multivariate probability distribution P on R^d is uniquely determined
       by the family of its 1D marginal distributions under all 1D linear projections
       u \in S^{d-1}. Testing multivariate standard normal N(0, I_d) is equivalent to testing
       that projected scalars u^T z ~ N(0, 1) for all u.

    2. Epps-Pulley Test (1983):
       Compares the 1D Empirical Characteristic Function (ECF):
           \hat{\phi}_N(t) = \frac{1}{N} \sum_{n=1}^N \exp(i t p_n)
       against the analytical standard Gaussian characteristic function:
           \phi_0(t) = \exp(-t^2 / 2)
       under the Gaussian-weighted L2 metric:
           T_{EP} = N \int_{-\infty}^{\infty} |\hat{\phi}_N(t) - \phi_0(t)|^2 d\mu(t)
       where d\mu(t) = \frac{1}{\sqrt{2\pi}} \exp(-t^2 / 2) dt is the standard Gaussian measure.

    3. Gradient Scaling (Why multiply by N?):
       The derivative of the empirical characteristic function with respect to projection p_n is:
           \frac{\partial \hat{\phi}_N(t)}{\partial p_n} = \frac{i t}{N} \exp(i t p_n)
       Without multiplying the test statistic by sample count N, \frac{\partial \mathcal{L}}{\partial p_n}
       would carry an extraneous 1/N factor (~1/768), leading to vanishing gradients on representations.
       Multiplying by N cancels this factor, producing an O(1) per-sample gradient that balances with
       the primary JEPA prediction loss, matching the asymptotic chi-squared distribution under H0.

    References:
    -----------
    - Epps, T. W., & Pulley, L. B. (1983). "A test for normality based on the empirical
      characteristic function." Biometrika, 70(3), 723-726.
    - Balestriero, R., & LeCun, Y. (2025). "LeJEPA: Provable and Scalable Self-Supervised Learning
      Without the Heuristics." arXiv:2511.08544 (SigReg).
    - Cramér, H., & Wold, H. (1936). "Some theorems on distribution functions."
      Journal of the London Mathematical Society, 1(4), 290-294.
    """

    def __init__(self, t_max: float = 3.0, n_knots: int = 17, normalize_measure: bool = True):
        super().__init__()
        self.t_max = t_max
        self.n_knots = n_knots
        self.normalize_measure = normalize_measure
        t = torch.linspace(0.0, t_max, n_knots, dtype=torch.float32)
        dt = t_max / (n_knots - 1)
        weights = torch.full((n_knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t.square())
        norm_const = 1.0 / math.sqrt(2.0 * math.pi) if normalize_measure else 1.0

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi * norm_const)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        proj: [N, K] where N is number of sample tokens, K is number of random 1D projections.
        """
        # Ensure calculations run in float32 to prevent numerical underflow under FP16/AMP
        orig_dtype = proj.dtype
        proj_f32 = proj.float()
        t = self.t.to(device=proj.device, dtype=torch.float32)
        phi = self.phi.to(device=proj.device, dtype=torch.float32)
        weights = self.weights.to(device=proj.device, dtype=torch.float32)

        x_t = proj_f32.unsqueeze(-1) * t  # [N, K, Q]
        ecf_real = x_t.cos().mean(dim=0)  # [K, Q]
        ecf_imag = x_t.sin().mean(dim=0)  # [K, Q]
        err = (ecf_real - phi).square() + ecf_imag.square()  # [K, Q]

        # Multiply by sample count N = proj.size(0) to cancel the 1/N factor in d(ecf)/dz
        statistic = (err @ weights) * proj.shape[0]  # [K]
        return statistic.mean().to(dtype=orig_dtype)


class SigRegLoss(nn.Module):
    r"""
    SigReg / LeJEPA Loss: Prediction Loss + Sketched Isotropic Gaussian Regularization.

    Tissue parity: the regularized tokens are tissue-filtered upstream
    (`filter_tissue_tokens`, same rule as VisReg), so the Epps-Pulley match fits
    the tissue manifold rather than the air-padding spike.

    Calibration status (uncalibrated): `sigreg_weight=1.0` is a default, not a
    tuned balance between JEPA and EP scales — sweep ~[0.1, 10] with collapse
    curves before citing. EP quadrature `t_max=3.0, n_knots=17` truncates the
    (-inf, inf) integral; tail mass beyond |t|>3 is assumed negligible.
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        sigreg_weight: float = 1.0,
        num_projections: int = 256,
        t_max: float = 3.0,
        n_knots: int = 17,
        normalize_measure: bool = True,
    ):
        super().__init__()
        self.jepa_loss = IJEPALoss(loss_type=loss_type)
        self.sigreg_weight = sigreg_weight
        self.num_projections = num_projections
        self.ep_test = EppsPulleyGaussianityTest(
            t_max=t_max,
            n_knots=n_knots,
            normalize_measure=normalize_measure,
        )

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
        context_tokens: torch.Tensor | None = None,
        tokens: torch.Tensor | None = None,
        projected_tokens: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        j_loss = self.jepa_loss(predictions, targets)

        reg_tokens = (
            tokens
            if tokens is not None
            else (projected_tokens if projected_tokens is not None else context_tokens)
        )
        if reg_tokens is None:
            raise ValueError("SigRegLoss requires regularized token representations.")

        # Flatten tokens across batch and patch dimensions: [N, D]
        z = reg_tokens.reshape(-1, reg_tokens.shape[-1])
        if z.dim() != 2 or z.shape[0] < 1 or z.shape[1] < 1:
            raise ValueError(
                f"SigRegLoss expects non-empty [N, D] tokens after flatten, got {tuple(z.shape)}."
            )
        D = z.shape[-1]

        # Sample M random projection directions on unit hypersphere in float32
        # to ensure isotropic distribution without half-precision artifacts.
        # Sampled on CPU when an explicit generator is given (CUDA generators
        # cannot back CPU-callers and vice versa), then moved to device so
        # seeded runs stay bitwise reproducible across devices.
        if generator is None:
            A = torch.randn(D, self.num_projections, device=z.device, dtype=torch.float32)
        else:
            A = torch.randn(D, self.num_projections, generator=generator, dtype=torch.float32).to(
                z.device
            )
        A = F.normalize(A, p=2, dim=0)  # [D, M]

        # 1D projections: [N, M] in float32
        proj = z.float() @ A

        sigreg_val = self.ep_test(proj)
        total_loss = j_loss + self.sigreg_weight * sigreg_val.to(dtype=j_loss.dtype)

        return {
            "loss": total_loss,
            "jepa_loss": j_loss,
            "sigreg_loss": sigreg_val,
        }
