import random

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
    4. Strict PyTorch PRNG Determinism:
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

            mask = mask.view(C, 1, 1, 1)
            out[b] = out[b] * mask

        if not is_batched:
            out = out.squeeze(0)
        return out


class VolumetricAugmentations3D:
    r"""
    3D Spatial & Intensity Augmentation Pipeline for Multi-Modal MRI Volumes.
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

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
        brain_mask: torch.Tensor | None = None,
    ):
        """
        image: [4, D, H, W]
        mask: [1, D, H, W] or None
        brain_mask: [1, D, H, W] or None
        """
        if not self.is_training:
            return (image, mask, brain_mask) if brain_mask is not None else (image, mask)

        # 1. 3D Random Axis Flips (Left-Right, Anterior-Posterior, Superior-Inferior)
        for axis in (1, 2, 3):  # D=1, H=2, W=3
            if random.random() < self.flip_prob:
                image = torch.flip(image, dims=[axis])
                if mask is not None:
                    mask = torch.flip(mask, dims=[axis])
                if brain_mask is not None:
                    brain_mask = torch.flip(brain_mask, dims=[axis])

        # 2. Additive Gaussian Electronics Noise (Parenchyma Only)
        if random.random() < self.noise_prob:
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
    Simulates 3D MRI quadrature Rician noise while preserving tissue contrast
    on Z-score normalized volumetric MRI scans.

    Mathematical Formulation (Gudbjartsson & Patz, 1995):
        M = \sqrt{(X + \eta_1)^2 + \eta_2^2}, \quad \eta_1, \eta_2 \sim \mathcal{N}(0, \sigma^2)
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

    min_val = image[bm_bool].min()
    # Shift non-zero brain parenchyma so min intensity is non-negative
    shifted = image - min_val if min_val < 0 else image
    eta1 = torch.randn_like(image) * sigma
    eta2 = torch.randn_like(image) * sigma
    noisy_shifted = torch.sqrt((shifted + eta1) ** 2 + eta2**2)
    noisy = noisy_shifted + min_val if min_val < 0 else noisy_shifted
    return torch.where(bm_bool, noisy, torch.zeros_like(image))


def apply_b1_bias_field_3d(
    image: torch.Tensor, strength: float = 0.3, brain_mask: torch.Tensor | None = None
) -> torch.Tensor:
    r"""
    Applies smooth multiplicative 3D B1 radiofrequency transmit/receive bias field.

    Mathematical Formulation (Sled et al., 1998; Lebrun et al., 2021):
        X_{corrupt} = X \cdot (1 + \sum_{i+j+k \le 2} c_{ijk} z^i y^j x^k)

    Parenchyma intensities are shifted to non-negative baseline prior to multiplicative
    scaling to prevent artificial contrast inversion on Z-score normalized data.
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
    z = torch.linspace(-1, 1, D, device=image.device, dtype=image.dtype)
    y = torch.linspace(-1, 1, H, device=image.device, dtype=image.dtype)
    x = torch.linspace(-1, 1, W, device=image.device, dtype=image.dtype)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")

    # Smooth 2nd-order polynomial field
    bias = 1.0 + strength * (
        0.5 * grid_z + 0.3 * grid_y - 0.4 * grid_x + 0.2 * (grid_z**2 + grid_y**2 + grid_x**2)
    )
    bias = bias.unsqueeze(0).unsqueeze(0) if image.dim() == 5 else bias.unsqueeze(0)

    min_val = image[bm_bool].min()
    shifted = image - min_val if min_val < 0 else image
    biased_shifted = shifted * bias
    biased = biased_shifted + min_val if min_val < 0 else biased_shifted
    return torch.where(bm_bool, biased, torch.zeros_like(image))

