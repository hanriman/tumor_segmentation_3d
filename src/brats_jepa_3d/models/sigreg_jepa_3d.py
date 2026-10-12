from .single_encoder_jepa_3d import SingleEncoderJEPA3D


class SigRegJEPA3D(SingleEncoderJEPA3D):
    r"""
    3D SigReg JEPA (LeJEPA): Single-Encoder Predictive Architecture with Sketched Gaussianity.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Elimination of EMA Momentum Teacher:
       SigReg eliminates the duplicated EMA teacher network entirely. The single encoder evaluates
       both context (with gradients) and target features (with stop-gradient). Collapse is prevented
       by mathematically enforcing maximal differential entropy via the Epps-Pulley test statistic.
    2. Projector MLP as an Information Decoupler:
       Directly regularizing encoder features D=384 to standard normal N(0, I) would erase non-Gaussian
       multi-modal tissue clusters. A 2-layer Projector MLP (384 -> 1024 -> 128) maps tokens to an
       auxiliary space where Gaussianity is enforced, leaving the backbone free to organize rich
       pathological features (Tishby et al., 2000; Chen et al., 2020; Balestriero & LeCun, 2025).
    """
