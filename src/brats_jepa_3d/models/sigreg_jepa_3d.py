from typing import Any

import torch
from torch import nn

from .predictor_3d import JEPAPredictor3D
from .vision_transformer_3d import VisionTransformerEncoder3D


class SigRegJEPA3D(nn.Module):
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

    def __init__(
        self,
        img_size: tuple[int, int, int] = (128, 128, 128),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 4,
        embed_dim: int = 384,
        proj_dim: int = 128,
        encoder_depth: int = 8,
        predictor_depth: int = 4,
        predictor_embed_dim: int = 192,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        sigreg_weight: float = 1.0,
    ):
        super().__init__()
        self.sigreg_weight = sigreg_weight

        # Single online encoder
        self.context_encoder = VisionTransformerEncoder3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # Projector MLP: 384 -> 1024 -> 128
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, proj_dim),
        )

        # 3D Predictor
        self.predictor = JEPAPredictor3D(
            embed_dim=embed_dim,
            pred_embed_dim=predictor_embed_dim,
            num_patches=self.context_encoder.num_patches,
            grid_size=self.context_encoder.grid_size,
            depth=predictor_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

    def update_target_encoder(self, momentum: float | None = None):
        """No-op: SigReg JEPA is heuristic-free and does not use an EMA teacher network."""

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: list[torch.Tensor],
    ) -> dict[str, Any]:
        # 1. Forward encoder on full image without gradients for target representations
        was_training = self.context_encoder.training
        self.context_encoder.eval()
        with torch.no_grad():
            target_full_tokens = self.context_encoder(images)  # [B, 512, embed_dim]
        if was_training:
            self.context_encoder.train()

        # 2. Forward encoder on ONLY visible context patches WITH gradients
        context_tokens = self.context_encoder(
            images, patch_indices=context_indices
        )  # [B, N_ctx, embed_dim]

        # 3. Project context tokens through MLP for Epps-Pulley Gaussianity regularization
        projected_tokens = self.projector(context_tokens)  # [B, N_ctx, proj_dim]

        predictions = []
        targets = []

        B, _, D = target_full_tokens.shape
        for target_indices in target_indices_list:
            N_tgt = target_indices.shape[1]
            gather_idx = target_indices.unsqueeze(-1).expand(B, N_tgt, D)
            target_repr = torch.gather(target_full_tokens, dim=1, index=gather_idx).detach()

            pred_repr = self.predictor(context_tokens, context_indices, target_indices)
            predictions.append(pred_repr)
            targets.append(target_repr)

        return {
            "predictions": predictions,
            "targets": targets,
            "context_tokens": context_tokens,
            "projected_tokens": projected_tokens,
            "target_tokens": target_full_tokens,
            "sigreg_weight": self.sigreg_weight,
        }
