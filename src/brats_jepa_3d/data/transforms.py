import torch
from torch import nn


class RandomModalityDropout3D(nn.Module):
    r"""
    Random 3D Multi-Modal Channel Dropout with Guaranteed Active Channel Fallback.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Clinical Relevance (Havaei et al., 2017; Dorent et al., 2019):
       Multi-modal brain MRI acquisitions frequently suffer from missing or corrupted sequences
       in emergency hospital triage (e.g., absent T1c due to severe renal contraindications to
       gadolinium contrast; motion-corrupted T2-FLAIR in agitated patients).
    2. Modality Independence Regularization:
       Stochastically dropping channels during pre-training and downstream fine-tuning forces the
       encoder to decouple cross-modal co-dependencies and build joint multi-spectral representations
       rather than relying exclusively on high-contrast sequences (such as T1c for enhancing rims).
    3. Guaranteed Non-Empty Fallback:
       If all 4 channels are sampled for dropout (\prod_c (1 - m_c) = 1), one channel is randomly
       forced active, preventing degenerate all-zero inputs from generating zero-gradient steps.
    4. Inverted-Dropout Magnitude Preservation:
       Kept channels are rescaled by `C / keep_count` so that the expected channel
       sum matches evaluation (no-dropout) magnitude — i.e. `E[output|train] ≈ input`.
       Without rescaling, training sees ~25% lower expected activation magnitude
       than testing (train/test shift). The binary `mask` semantics (0/1 kept/dropped)
       are preserved; only the kept-channel gain changes.
    5. Strict PyTorch PRNG Determinism:
       Uses `torch.bernoulli` and `torch.randint` on `image.device` rather than Python's non-seeded
       `random` module, guaranteeing exact bitwise reproducibility under `torch.manual_seed` across
       DataLoader multiprocessing workers and distributed ranks.
    """

    def __init__(self, p_drop: float = 0.25):
        super().__init__()
        self.p_drop = p_drop

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: [C=4, D, H, W] or [B, C=4, D, H, W]
        """
        if not self.training or self.p_drop <= 0.0:
            return image

        is_batched = image.dim() == 5
        if not is_batched:
            image = image.unsqueeze(0)

        B, C = image.shape[0], image.shape[1]
        out = image.clone()

        for b in range(B):
            # Sample Bernoulli mask for channels
            mask = torch.bernoulli(torch.full((C,), 1.0 - self.p_drop, device=image.device))
            if mask.sum() == 0:
                # Guaranteed active channel fallback using PyTorch PRNG for seed consistency
                active_idx = torch.randint(0, C, (1,), device=image.device).item()
                mask[active_idx] = 1.0

            # Inverted-dropout rescaling: preserve expected activation magnitude
            # (fallback single channel scales by C to match full-input energy).
            keep = float(mask.sum().item())
            scale = (C / keep) if keep > 0 else 1.0
            out[b] = out[b] * mask.view(C, 1, 1, 1) * scale

        if not is_batched:
            out = out.squeeze(0)
        return out


class VolumetricAugmentations3D:
    r"""
    3D Spatial & Intensity Augmentation Pipeline for Multi-Modal MRI Volumes.

    RNG contract: all stochastic decisions use torch PRNG (never Python `random`),
    so `torch.manual_seed` reproduces bitwise-identical outputs across DataLoader
    workers. An optional `torch.Generator` may be passed per call; when None, draws
    derive from the global torch RNG state.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        noise_prob: float = 0.3,
        noise_std: float = 0.05,
        modality_dropout_prob: float = 0.25,
        is_training: bool = True,
    ):
        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.is_training = is_training
        self.modality_dropout = RandomModalityDropout3D(p_drop=modality_dropout_prob)

    @staticmethod
    def _coin_flip(prob: float, generator: torch.Generator | None) -> bool:
        if prob <= 0.0:
            return False
        if prob >= 1.0:
            return True
        return bool((torch.rand((), generator=generator).item() < prob))

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
        brain_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ):
        """
        image: [4, D, H, W]
        mask: [1, D, H, W] or None
        brain_mask: [1, D, H, W] or None
        generator: optional torch.Generator for fully explicit determinism.
        """
        if not self.is_training:
            return (image, mask, brain_mask) if brain_mask is not None else (image, mask)

        # 1. 3D Random Axis Flips (Left-Right, Anterior-Posterior, Superior-Inferior)
        for axis in (-3, -2, -1):  # Spatial dimensions (D, H, W) regardless of leading batch dimension
            if self._coin_flip(self.flip_prob, generator):
                image = torch.flip(image, dims=[axis])
                if mask is not None:
                    mask = torch.flip(mask, dims=[axis])
                if brain_mask is not None:
                    brain_mask = torch.flip(brain_mask, dims=[axis])

        # 2. Additive Gaussian Electronics Noise (Parenchyma Only)
        if self._coin_flip(self.noise_prob, generator):
            parenchyma_mask = (brain_mask > 0) if brain_mask is not None else (image != 0)
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise * parenchyma_mask.float()

        # 3. Random Modality Dropout
        image = self.modality_dropout(image)

        if brain_mask is not None:
            return image, mask, brain_mask
        return image, mask


def apply_rician_noise_3d(
    image: torch.Tensor, sigma: float = 0.10, brain_mask: torch.Tensor | None = None
) -> torch.Tensor:
    r"""
    Simulates 3D MRI quadrature Rician noise on volumetric MRI scans.

    Mathematical Formulation (Gudbjartsson & Patz, 1995):
        M = \sqrt{(X + \eta_1)^2 + \eta_2^2}, \quad \eta_1, \eta_2 \sim \mathcal{N}(0, \sigma^2)

    Design decisions (documented approximations):
    - Magnitude is applied to **non-negative voxels** with true Rician statistics
      ``M = sqrt((X+ + eta1)^2 + eta2^2)`` (``X+ = max(X, 0)``); voxels that are
      already negative after Z-score normalization receive additive Gaussian
      noise ``X + eta1`` (the high-SNR limit where Rician -> Gaussian). This
      preserves sub-mean tissue contrast instead of rectifying it to zero.
    - Background air follows Rayleigh statistics (`X = 0` limit) instead of being
      forced to zero — zero air is unphysical for magnitude MRI.
    - `sigma <= 0` and empty inputs are identity (no-op) paths.
    - Re-baseline notes: 2026-09-21 (zero air, `min_val` shift) and this fix
      (negative-preserving tissue noise) both change OOD numbers — re-run
      `evaluate_ood_3d.py` before citing.
    """
    if sigma <= 0:
        return image
    if brain_mask is None:
        bm_bool = image != 0
    else:
        bm_bool = brain_mask > 0
        if bm_bool.dim() < image.dim():
            bm_bool = bm_bool.unsqueeze(0)
        if bm_bool.shape != image.shape:
            bm_bool = bm_bool.expand_as(image)

    if not bm_bool.any():
        return image

    eta1 = torch.randn_like(image) * sigma
    eta2 = torch.randn_like(image) * sigma
    nonneg = image >= 0
    rician = torch.sqrt((torch.clamp(image, min=0.0) + eta1) ** 2 + eta2**2)
    gaussian = image + eta1
    tissue_noisy = torch.where(nonneg, rician, gaussian)
    air = torch.sqrt(eta1**2 + eta2**2)
    return torch.where(bm_bool, tissue_noisy, air)


def apply_b1_bias_field_3d(
    image: torch.Tensor,
    strength: float = 0.3,
    brain_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""
    Applies smooth multiplicative 3D B1 radiofrequency transmit/receive bias field.

    Mathematical Formulation (Sled et al., 1998; Lebrun et al., 2021):
        X_{corrupt} = X \cdot (1 + \sum_{i+j+k \le 2} c_{ijk} z^i y^j x^k)

    Design decisions:
    - The 10 polynomial coefficients `c_ijk` are sampled `N(0, 1)` **per sample
      in the batch** (shape `[B, 10]`), so every volume sees a different smooth
      field instead of one fixed field for the whole cohort. Pass `generator`
      for fully explicit determinism (same generator state → identical fields).
    - The field is clamped to `[-0.5, 0.5]` before the `strength` scaling, so
      the multiplicative gain stays in `[1 - 0.5*strength, 1 + 0.5*strength]`
      (e.g. `[0.85, 1.15]` at the default `strength=0.3`; no contrast sign
      flips).
    - The gain multiplies the original (possibly Z-scored, possibly negative)
      voxel value, preserving sub-mean contrast; background stays zero (bias is
      a gain on acquired signal, and air carries none). This intentionally
      differs from `apply_rician_noise_3d`, which synthesizes Rayleigh air:
      Rician models the magnitude noise floor (air has noise), B1 models a
      multiplicative receive/transmit gain (air has no signal to scale).
    - Re-baseline note (2026-09-21 remediation, extended per-sample): the
      pre-fix field was a single fixed coefficient set for the whole cohort,
      and the interim fix used one random field broadcast across the batch, so
      previously reported B1 numbers are NOT comparable to post-fix
      (per-sample random field) numbers.
    """
    if brain_mask is None:
        bm_bool = image != 0
    else:
        bm_bool = brain_mask > 0
        if bm_bool.dim() < image.dim():
            bm_bool = bm_bool.unsqueeze(0)
        if bm_bool.shape != image.shape:
            bm_bool = bm_bool.expand_as(image)

    if not bm_bool.any():
        return image

    D, H, W = image.shape[-3], image.shape[-2], image.shape[-1]
    dev, dt = image.device, image.dtype
    z = torch.linspace(-1, 1, D, device=dev, dtype=torch.float32)
    y = torch.linspace(-1, 1, H, device=dev, dtype=torch.float32)
    x = torch.linspace(-1, 1, W, device=dev, dtype=torch.float32)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")

    # 10 monomials with i+j+k <= 2: 1, z, y, x, z^2, y^2, x^2, zy, zx, yx.
    monos = torch.stack([
        torch.ones_like(grid_z),
        grid_z, grid_y, grid_x,
        grid_z**2, grid_y**2, grid_x**2,
        grid_z * grid_y, grid_z * grid_x, grid_y * grid_x,
    ])  # [10, D, H, W]
    # Per-sample coefficients: sample on CPU (generator-compatible across
    # devices) then move to the image device. Sequential draws keep seeded and
    # explicit-generator determinism: same state -> identical fields.
    if image.dim() == 5:
        n_batch = image.shape[0]
        coeffs = torch.randn(n_batch, 10, generator=generator, dtype=torch.float32).to(dev)
        field = torch.clamp(torch.einsum("bi,idhw->bdhw", coeffs, monos), -0.5, 0.5)
        bias = (1.0 + strength * field).to(dt).unsqueeze(1)  # [B, 1, D, H, W]
    else:
        coeffs = torch.randn(10, generator=generator, dtype=torch.float32).to(dev)
        field = torch.clamp((coeffs[:, None, None, None] * monos).sum(dim=0), -0.5, 0.5)
        bias = (1.0 + strength * field).to(dt).unsqueeze(0)  # [1, D, H, W]

    return torch.where(bm_bool, image * bias, torch.zeros_like(image))

