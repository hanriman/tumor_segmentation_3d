import math
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class PatchEmbed3D(nn.Module):
    r"""
    3D Volumetric Patch Embedding via 3D Convolution.

    Mathematical Rationale:
    -----------------------
    Transforms a continuous volumetric tensor \mathbf{X} \in \mathbb{R}^{B \times C \times D \times H \times W}
    into a discrete sequence of patch tokens. For canonical shape 128x128x128 and patch size 16x16x16:
        Grid size: G = (128/16, 128/16, 128/16) = (8, 8, 8)
        Total tokens: N = 8 \times 8 \times 8 = 512 patches.
    Using Conv3d with kernel_size=stride=(16, 16, 16) performs non-overlapping linear projection
    of each 16^3 voxel neighborhood into the latent embedding dimension D=384.
    """
    def __init__(
        self,
        img_size: tuple[int, int, int] = (128, 128, 128),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 4,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W] -> proj: [B, embed_dim, G_d, G_h, G_w] -> [B, N, embed_dim]
        B, C, D, H, W = x.shape
        feat = self.proj(x)
        tokens = feat.flatten(2).transpose(1, 2)
        return tokens


def build_3d_sinusoidal_pos_embedding(
    grid_size: tuple[int, int, int] = (8, 8, 8),
    embed_dim: int = 384,
    temperature: float = 10000.0,
) -> torch.Tensor:
    r"""
    Generates separable 3D sinusoidal coordinate positional embeddings.

    Theoretical Justification (Feichtenhofer et al., NeurIPS 2022; Vaswani et al., 2017):
    ------------------------------------------------------------------------------------
    Allocates \frac{D}{3} feature channels each to z, y, and x spatial coordinate axes.
    Provides immediate Euclidean distance metric topology from step 0, accelerating
    spatial attention convergence compared to randomly initialized flat embeddings.
    """
    gz, gy, gx = grid_size
    dim_per_axis = embed_dim // 3
    # Extra channels assigned to x if not evenly divisible by 3
    dim_z = dim_per_axis
    dim_y = dim_per_axis
    dim_x = embed_dim - (dim_z + dim_y)

    def get_1d_sincos(length: int, dim: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32)
        half_dim = dim // 2
        omega = torch.arange(half_dim, dtype=torch.float32) / half_dim
        omega = 1.0 / (temperature ** omega)
        out = torch.einsum("m,d->md", pos, omega)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(length, 1, dtype=torch.float32)], dim=1)
        return emb

    pe_z = get_1d_sincos(gz, dim_z)  # [gz, dim_z]
    pe_y = get_1d_sincos(gy, dim_y)  # [gy, dim_y]
    pe_x = get_1d_sincos(gx, dim_x)  # [gx, dim_x]

    # Combine into 3D grid: [gz, gy, gx, embed_dim]
    pe = torch.zeros(gz, gy, gx, embed_dim, dtype=torch.float32)
    pe[..., :dim_z] = pe_z[:, None, None, :]
    pe[..., dim_z : dim_z + dim_y] = pe_y[None, :, None, :]
    pe[..., dim_z + dim_y :] = pe_x[None, None, :, :]

    # Flatten spatial grid to sequence: [1, 512, embed_dim]
    pe = pe.view(1, gz * gy * gx, embed_dim)
    return pe


class TransformerBlock(nn.Module):
    r"""
    Pre-LayerNorm Transformer Block (Xiong et al., ICML 2020).
    """
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm self-attention with residual skip
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out

        # Pre-LayerNorm MLP with residual skip
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerEncoder3D(nn.Module):
    r"""
    3D Vision Transformer Encoder with Zero Attention Leakage Token Selection.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Zero-Leakage Selective Evaluation (Assran et al., CVPR 2023):
       When patch_indices [B, N_ctx] are provided, only visible context tokens are gathered
       before feeding into the transformer blocks. Unlike BERT/standard MAE which pass masked
       tokens through self-attention, keys and queries never interact with masked target voxels.
       Attention complexity drops from O(512^2) to O(192^2), yielding an 85.9% reduction in attention FLOPs.
    2. Hierarchical Intermediate Feature Extraction:
       Supports `return_intermediate=True` to extract intermediate representations across depths
       (e.g., L_2, L_4, L_6, L_8) for downstream multi-scale FPN decoding.
    """
    def __init__(
        self,
        img_size: tuple[int, int, int] = (128, 128, 128),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 4,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size
        self.embed_dim = embed_dim

        # Initialize 3D sinusoidal coordinate positional embeddings (learnable)
        pe = build_3d_sinusoidal_pos_embedding(self.grid_size, embed_dim)
        self.pos_embed = nn.Parameter(pe)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        patch_indices: torch.Tensor | None = None,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        x: [B, C, D, H, W]
        patch_indices: [B, N_ctx] or None (full grid 512 tokens)
        """
        tokens = self.patch_embed(x) + self.pos_embed

        if patch_indices is not None:
            # Zero-leakage gather: select only visible context tokens
            B, N_ctx = patch_indices.shape
            D = tokens.shape[-1]
            gather_indices = patch_indices.unsqueeze(-1).expand(B, N_ctx, D)
            tokens = torch.gather(tokens, dim=1, index=gather_indices)

        intermediates = []
        for block in self.blocks:
            tokens = block(tokens)
            if return_intermediate:
                intermediates.append(tokens)

        out = self.norm(tokens)

        if return_intermediate:
            return out, intermediates
        return out
