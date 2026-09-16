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

        B, C, D, H, W = image.shape
        out = image.clone()

        for b in range(B):
            # Sample Bernoulli mask for channels
            mask = torch.bernoulli(torch.full((C,), 1.0 - self.p_drop, device=image.device))
            if mask.sum() == 0:
                # Guaranteed active channel fallback
                active_idx = random.randint(0, C - 1)
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        image: [4, D, H, W]
        mask: [1, D, H, W] or None
        """
        if not self.is_training:
            return image, mask

        # 1. 3D Random Axis Flips (Left-Right, Anterior-Posterior, Superior-Inferior)
        for axis in (1, 2, 3):  # D=1, H=2, W=3
            if random.random() < self.flip_prob:
                image = torch.flip(image, dims=[axis])
                if mask is not None:
                    mask = torch.flip(mask, dims=[axis])

        # 2. Additive Gaussian Electronics Noise (Parenchyma Only)
        if random.random() < self.noise_prob:
            parenchyma_mask = image != 0
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise * parenchyma_mask.float()

        # 3. Random Modality Dropout
        image = self.modality_dropout(image)

        return image, mask
