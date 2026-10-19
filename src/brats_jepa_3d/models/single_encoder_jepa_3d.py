from typing import Any

import torch
from torch import nn

from .predictor_3d import JEPAPredictor3D
from .vision_transformer_3d import VisionTransformerEncoder3D, dropout_disabled
from ._tissue_filter import filter_tissue_tokens


class SingleEncoderJEPA3D(nn.Module):
    r"""
    3D Single-Encoder Joint-Embedding Predictive Architecture.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Single-Encoder Predictive Topology (Balestriero & LeCun, 2025; Wu et al., 2026):
       Eliminates dual-network heuristics (EMA teacher networks, stop-gradient momentum drift).
       A single online encoder evaluates visible context patches with gradients, while
       evaluating the target representations on the full volume with stopped gradients and
       temporarily disabled dropout (`dropout_disabled`).
    2. Direct Encoder-Space Statistical Regularization (Balestriero & LeCun, 2025, Eq. 9; Wu et al., 2026, Alg. 1):
       Anti-collapse statistical regularization (VISReg or SigReg) is applied directly to the
       tissue-filtered context encoder representations in R^{embed_dim} without an intermediate
       nonlinear projector MLP, preventing shortcut solutions where the projector satisfies
       Gaussianity while the encoder backbone collapses onto a low-rank subspace.
    3. Multi-Target Optimized Predictor:
       Context projection and positional gather are executed once per forward pass via
       `predictor.forward_multi_targets`, avoiding redundant matrix multiplications across target cuboids.
    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (128, 128, 128),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 4,
        embed_dim: int = 384,
        encoder_depth: int = 8,
        predictor_depth: int = 4,
        predictor_embed_dim: int = 192,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        proj_dim: int = 128,
        projector_hidden_dim: int = 1024,
        use_projector: bool = True,
        **kwargs,
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

        self.use_projector = use_projector
        if use_projector:
            self.projector = nn.Sequential(
                nn.Linear(embed_dim, projector_hidden_dim),
                nn.LayerNorm(projector_hidden_dim),
                nn.GELU(),
                nn.Linear(projector_hidden_dim, proj_dim),
            )
        else:
            self.projector = None

        self.predictor = JEPAPredictor3D(
            embed_dim=embed_dim,
            pred_embed_dim=predictor_embed_dim,
            num_patches=self.context_encoder.num_patches,
            grid_size=self.context_encoder.grid_size,
            depth=predictor_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: list[torch.Tensor],
        context_tissue_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        # 1. Target representations: full image pass without gradients and dropout disabled
        with torch.no_grad(), dropout_disabled(self.context_encoder):
            target_full_tokens = self.context_encoder(images)  # [B, N, embed_dim]

        # 2. Context representations: visible context patches with gradients
        context_tokens = self.context_encoder(
            images, patch_indices=context_indices
        )  # [B, N_ctx, embed_dim]

        # 3. Filter context tokens or project before filtering
        if self.projector is not None:
            full_projected = self.projector(context_tokens)  # [B, N_ctx, proj_dim]
            projected_tokens = filter_tissue_tokens(full_projected, context_tissue_mask)
            reg_tokens = projected_tokens
        else:
            projected_tokens = None
            reg_tokens = filter_tissue_tokens(context_tokens, context_tissue_mask)

        # 4. Predict target representations efficiently
        predictions = self.predictor.forward_multi_targets(
            context_tokens, context_indices, target_indices_list
        )

        # 5. Extract ground truth target representations from target_full_tokens
        targets = []
        B, _, D = target_full_tokens.shape
        for target_indices in target_indices_list:
            N_tgt = target_indices.shape[1]
            gather_idx = target_indices.unsqueeze(-1).expand(B, N_tgt, D)
            target_repr = torch.gather(target_full_tokens, dim=1, index=gather_idx).detach()
            targets.append(target_repr)

        return {
            "predictions": predictions,
            "targets": targets,
            "context_tokens": context_tokens,
            "tissue_tokens": reg_tokens,
            "projected_tokens": projected_tokens if projected_tokens is not None else reg_tokens,
            "target_tokens": target_full_tokens,
        }


VisRegJEPA3D = SingleEncoderJEPA3D
SigRegJEPA3D = SingleEncoderJEPA3D
