import copy
from typing import Any

import torch
from torch import nn

from .predictor_3d import JEPAPredictor3D
from .vision_transformer_3d import VisionTransformerEncoder3D


class IJEPA3D(nn.Module):
    r"""
    3D Volumetric Multi-Modal I-JEPA (Assran et al., CVPR 2023).

    Architecture:
    - Context Encoder E_theta: processes visible context patches [B, 192, 384]
    - Target Encoder E_theta_bar: evaluates full volume [B, 512, 384] without gradients (updated via EMA)
    - 3D Predictor P_phi: predicts target representations from context + target position queries
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
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.ema_momentum = ema_momentum

        # 1. Online Context Encoder
        self.context_encoder = VisionTransformerEncoder3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # 2. Target Encoder (EMA updated, gradients stopped, evaluated in eval mode)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

        # 3. 3D Predictor
        self.predictor = JEPAPredictor3D(
            embed_dim=embed_dim,
            pred_embed_dim=predictor_embed_dim,
            num_patches=self.context_encoder.num_patches,
            grid_size=self.context_encoder.grid_size,
            depth=predictor_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

    def train(self, mode: bool = True):
        r"""Override train to keep target_encoder strictly in eval mode (Assran et al., 2023)."""
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None):
        """EMA momentum update of target encoder weights."""
        m = momentum if momentum is not None else self.ema_momentum
        for param_q, param_k in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: list[torch.Tensor],
    ) -> dict[str, Any]:
        """
        images: [B, 4, 128, 128, 128]
        context_indices: [B, N_ctx]
        target_indices_list: list of [B, N_tgt]
        """
        # 1. Target representations from target encoder on full volume (no gradients)
        with torch.no_grad():
            target_full_tokens = self.target_encoder(images)  # [B, 512, embed_dim]

        # 2. Context representations from context encoder (evaluated strictly on visible context)
        context_tokens = self.context_encoder(
            images, patch_indices=context_indices
        )  # [B, N_ctx, embed_dim]

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
            "target_tokens": target_full_tokens,
        }
