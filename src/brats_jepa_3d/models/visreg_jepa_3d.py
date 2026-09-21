from typing import Any

import torch
from torch import nn

from .predictor_3d import JEPAPredictor3D
from .vision_transformer_3d import VisionTransformerEncoder3D, dropout_disabled
from ._tissue_filter import filter_tissue_tokens


class VisRegJEPA3D(nn.Module):
    r"""
    3D VisReg JEPA: Single-Encoder Predictive Architecture with Decoupled Sliced-Wasserstein Regularization.

    Mathematical Rationale & Defense Context (Wu, Balestriero, & Levine, 2026):
    ---------------------------------------------------------------------------
    Decouples representation learning into JEPA predictive task while preventing collapse
    via three statistical regularizers: Center (zero mean), Scale (two-sided unit variance),
    and Shape (1D Sliced-Wasserstein distance to Gaussian quantiles).
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
    ):
        super().__init__()
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
        """No-op: VisReg JEPA is heuristic-free and does not use an EMA teacher network."""

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: list[torch.Tensor],
        context_tissue_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        # 1. Forward encoder on full image without gradients for target representations.
        # Deterministic targets via dropout-disabled pass: never mutates training
        # mode (no .eval()/.train() flip), so forward() has no caller-visible
        # side effect and is safe under DDP / threads. Encoder input includes
        # air voxels by design; only the projector regularizer is tissue-filtered.
        with torch.no_grad(), dropout_disabled(self.context_encoder):
            target_full_tokens = self.context_encoder(images)  # [B, 512, embed_dim]

        # 2. Forward encoder on ONLY visible context patches WITH gradients
        context_tokens = self.context_encoder(
            images, patch_indices=context_indices
        )  # [B, N_ctx, embed_dim]

        # 3. Project context tokens through MLP for VisReg regularization.
        # Tissue-only filtering (shared helper): the Gaussian match fits the
        # tissue manifold, not the air-padding spike; per-sample fallback.
        full_projected = self.projector(context_tokens)  # [B, N_ctx, proj_dim]
        projected_tokens = filter_tissue_tokens(full_projected, context_tissue_mask)

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
        }
