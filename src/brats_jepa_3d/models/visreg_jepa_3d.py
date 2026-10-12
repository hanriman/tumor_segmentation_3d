from .single_encoder_jepa_3d import SingleEncoderJEPA3D


class VisRegJEPA3D(SingleEncoderJEPA3D):
    r"""
    3D VisReg JEPA: Single-Encoder Predictive Architecture with Decoupled Sliced-Wasserstein Regularization.

    Mathematical Rationale & Defense Context (Wu, Balestriero, & Levine, 2026):
    ---------------------------------------------------------------------------
    Decouples representation learning into JEPA predictive task while preventing collapse
    via three statistical regularizers: Center (zero mean), Scale (two-sided unit variance),
    and Shape (1D Sliced-Wasserstein distance to Gaussian quantiles).
    """
