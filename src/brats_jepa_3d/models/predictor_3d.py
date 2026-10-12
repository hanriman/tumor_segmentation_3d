import torch
from torch import nn

from .vision_transformer_3d import TransformerBlock, build_3d_sinusoidal_pos_embedding


class JEPAPredictor3D(nn.Module):
    r"""
    3D Latent Space Predictor for Joint-Embedding Predictive Architectures.

    Mathematical Rationale & Defense Context (Assran et al., CVPR 2023):
    --------------------------------------------------------------------
    1. Narrow Bottleneck Capacity:
       The predictor is intentionally lightweight (4 layers, pred_embed_dim = 192).
       If the predictor were overly expressive, it could absorb complex spatial mapping
       heuristics internally, disincentivizing the context encoder from organizing rich
       semantic representations. A compact predictor forces representation learning into the encoder.
    2. Positional Query Conditioning:
       The predictor receives context representations and target spatial position queries,
       predicting target representations \hat{y}_{\text{tgt}} entirely within latent embedding space
       without reconstructing corrupted voxel intensities.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        pred_embed_dim: int = 192,
        num_patches: int = 512,
        grid_size: tuple[int, int, int] = (8, 8, 8),
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pred_embed_dim = pred_embed_dim
        self.num_patches = num_patches

        # Linear projection to narrower predictor dimension
        self.context_proj = nn.Linear(embed_dim, pred_embed_dim)

        # 3D sinusoidal coordinate positional embeddings for predictor queries.
        # Frozen like the encoder topology (see VisionTransformerEncoder3D).
        pe = build_3d_sinusoidal_pos_embedding(grid_size, pred_embed_dim)
        self.pos_embed = nn.Parameter(pe, requires_grad=False)

        # Learnable mask query token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=pred_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(pred_embed_dim)

        # Projection back to encoder representation dimension
        self.out_proj = nn.Linear(pred_embed_dim, embed_dim)

    def predict_target_block(
        self,
        ctx: torch.Tensor,
        N_ctx: int,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluates a single target query given pre-projected and position-conditioned context."""
        B, N_tgt = target_indices.shape
        D_pred = self.pred_embed_dim

        # Target positional queries
        tgt_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1),
            dim=1,
            index=target_indices.unsqueeze(-1).expand(B, N_tgt, D_pred),
        )
        # Initialize target queries with mask token + target positional embeddings
        tgt = self.mask_token.expand(B, N_tgt, -1) + tgt_pos

        # Concatenate [Context, Target Queries]
        combined = torch.cat([ctx, tgt], dim=1)  # [B, N_ctx + N_tgt, D_pred]

        for block in self.blocks:
            combined = block(combined)

        combined = self.norm(combined)

        # Extract only the target token positions
        pred_tgt = combined[:, N_ctx:, :]

        # Project back to encoder embedding space
        return self.out_proj(pred_tgt)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        context_tokens: [B, N_ctx, embed_dim]
        context_indices: [B, N_ctx]
        target_indices: [B, N_tgt]
        Returns: predicted target representations [B, N_tgt, embed_dim]
        """
        B, N_ctx, _ = context_tokens.shape
        D_pred = self.pred_embed_dim

        # Project context tokens to predictor dimension and add sinusoidal coordinates
        ctx = self.context_proj(context_tokens)
        ctx_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1),
            dim=1,
            index=context_indices.unsqueeze(-1).expand(B, N_ctx, D_pred),
        )
        ctx = ctx + ctx_pos

        return self.predict_target_block(ctx, N_ctx, target_indices)

    def forward_multi_targets(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Efficient multi-target evaluation: computes context linear projection and positional
        conditioning once per forward pass across all target cuboid queries.
        """
        B, N_ctx, _ = context_tokens.shape
        D_pred = self.pred_embed_dim

        ctx = self.context_proj(context_tokens)
        ctx_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1),
            dim=1,
            index=context_indices.unsqueeze(-1).expand(B, N_ctx, D_pred),
        )
        ctx = ctx + ctx_pos

        return [
            self.predict_target_block(ctx, N_ctx, tgt_idx)
            for tgt_idx in target_indices_list
        ]

