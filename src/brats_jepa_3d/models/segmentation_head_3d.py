import contextlib

import torch
from torch import nn

from .vision_transformer_3d import VisionTransformerEncoder3D


class ViTSegmentationDecoder3D(nn.Module):
    r"""
    Progressive 4-Stage Transpose-Convolutional Volumetric Upsampling Decoder.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Spatial Token Reshaping & Progressive 3D Upsampling:
       The Vision Transformer produces 1D patch tokens [B, N=512, D=384]. This decoder reshapes
       the sequence into a 3D spatial feature map of shape [B, D, 8, 8, 8] and progressively
       doubles resolution across 4 strided transpose convolutions:
           8 \times 8 \times 8 \xrightarrow{2\times} 16 \times 16 \times 16 \xrightarrow{2\times} 32 \times 32 \times 32
           \xrightarrow{2\times} 64 \times 64 \times 64 \xrightarrow{2\times} 128 \times 128 \times 128
       reconstructing full voxel-level resolution matching the original canonical MRI volume.

    2. Group Normalization (Wu & He, ECCV 2018):
       Batch Normalization in small downstream fine-tuning batches (B <= 4) suffers from noisy
       mean and variance estimates, which destabilizes fine-tuning. GroupNorm divides channels
       into independent groups (e.g. 16, 8, 4), computing statistics along spatial and sub-channel
       dimensions per-sample, guaranteeing robust convergence across variable batch sizes.
    """

    def __init__(
        self, in_dim: int = 384, out_channels: int = 1, grid_size: tuple[int, int, int] = (8, 8, 8)
    ):
        super().__init__()
        self.in_dim = in_dim
        self.grid_size = grid_size

        self.decoder = nn.Sequential(
            # Stage 1: 8^3 -> 16^3
            nn.ConvTranspose3d(in_dim, 192, kernel_size=2, stride=2),
            nn.GroupNorm(16, 192),
            nn.GELU(),
            # Stage 2: 16^3 -> 32^3
            nn.ConvTranspose3d(192, 96, kernel_size=2, stride=2),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            # Stage 3: 32^3 -> 64^3
            nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2),
            nn.GroupNorm(4, 48),
            nn.GELU(),
            # Stage 4: 64^3 -> 128^3
            nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            # Projection head to logits
            nn.Conv3d(24, out_channels, kernel_size=1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [B, N=512, D=384] -> [B, D, 8, 8, 8]
        B, _N, D = patch_tokens.shape
        gz, gy, gx = self.grid_size
        x = patch_tokens.permute(0, 2, 1).reshape(B, D, gz, gy, gx)
        return self.decoder(x)


class MultiScaleViTSegmentationDecoder3D(nn.Module):
    r"""
    Multi-Scale Feature Pyramid Decoder for 3D Vision Transformer Representations.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Restoration of Boundary Contours:
       Bottleneck decoding abstracts away fine localized voxel geometry in favor of global invariant
       semantics, leading to blurred tumor margins and elevated HD95 distance errors.
    2. Hierarchical Multi-Scale Fusion (Lin et al., CVPR 2017; Hatamizadeh et al., WACV 2022):
       Intermediate transformer layers preserve distinct geometric scales:
       - Shallow layers (L_2, L_4): fine localized 3D edge gradients, tissue boundaries, and microvascular rims.
       - Deep layers (L_6, L_8): high-level anatomical context, ventricular topology, and tumor semantics.
       Lateral skip connections progressively project and concatenate intermediate features into
       corresponding decoder upsampling stages, restoring fine-grained 3D boundary delineation.
    """

    def __init__(
        self,
        in_dim: int = 384,
        out_channels: int = 1,
        grid_size: tuple[int, int, int] = (8, 8, 8),
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.grid_size = grid_size
        self.deep_supervision = deep_supervision

        # Stage 1: Bottleneck 8^3 -> 16^3
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(in_dim, 192, kernel_size=2, stride=2),
            nn.GroupNorm(16, 192),
            nn.GELU(),
        )
        # Skip connection from L6 (8^3 -> 16^3)
        self.skip_l6 = nn.Sequential(
            nn.ConvTranspose3d(in_dim, 192, kernel_size=2, stride=2),
            nn.GroupNorm(16, 192),
            nn.GELU(),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv3d(192 + 192, 192, kernel_size=3, padding=1),
            nn.GroupNorm(16, 192),
            nn.GELU(),
        )

        # Stage 2: 16^3 -> 32^3
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(192, 96, kernel_size=2, stride=2),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        # Skip connection from L4 (8^3 -> 32^3 via 4x transpose conv)
        self.skip_l4 = nn.Sequential(
            nn.ConvTranspose3d(in_dim, 96, kernel_size=4, stride=4),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv3d(96 + 96, 96, kernel_size=3, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )

        # Stage 3: 32^3 -> 64^3
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2),
            nn.GroupNorm(4, 48),
            nn.GELU(),
        )
        # Skip connection from L2 (8^3 -> 32^3 -> 64^3 via progressive transpose convs)
        self.skip_l2 = nn.Sequential(
            nn.ConvTranspose3d(in_dim, 96, kernel_size=4, stride=4),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2),
            nn.GroupNorm(4, 48),
            nn.GELU(),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv3d(48 + 48, 48, kernel_size=3, padding=1),
            nn.GroupNorm(4, 48),
            nn.GELU(),
        )

        # Stage 4: 64^3 -> 128^3
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2),
            nn.GroupNorm(4, 24),
            nn.GELU(),
        )

        # Final projection to output logits
        self.head = nn.Conv3d(24, out_channels, kernel_size=1)

        if deep_supervision:
            self.ds3 = nn.Conv3d(48, out_channels, kernel_size=1)
            self.ds2 = nn.Conv3d(96, out_channels, kernel_size=1)
            self.ds1 = nn.Conv3d(192, out_channels, kernel_size=1)

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """Converts [B, N=512, D] patch tokens to [B, D, 8, 8, 8] spatial feature map."""
        B, _N, D = tokens.shape
        gz, gy, gx = self.grid_size
        return tokens.permute(0, 2, 1).reshape(B, D, gz, gy, gx)

    def forward(
        self,
        intermediate_tokens: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(intermediate_tokens, (list, tuple)):
            n_layers = len(intermediate_tokens)
            idx2 = max(0, n_layers // 4 - 1)
            idx4 = max(0, n_layers // 2 - 1)
            idx6 = max(0, 3 * n_layers // 4 - 1)
            idx8 = n_layers - 1
            z2 = self._tokens_to_spatial(intermediate_tokens[idx2])
            z4 = self._tokens_to_spatial(intermediate_tokens[idx4])
            z6 = self._tokens_to_spatial(intermediate_tokens[idx6])
            z8 = self._tokens_to_spatial(intermediate_tokens[idx8])
        else:
            z = self._tokens_to_spatial(intermediate_tokens)
            z2 = z4 = z6 = z8 = z

        # Stage 1: 8^3 -> 16^3
        x1 = self.up1(z8)
        s1 = self.skip_l6(z6)
        x1 = self.fuse1(torch.cat([x1, s1], dim=1))

        # Stage 2: 16^3 -> 32^3
        x2 = self.up2(x1)
        s2 = self.skip_l4(z4)
        x2 = self.fuse2(torch.cat([x2, s2], dim=1))

        # Stage 3: 32^3 -> 64^3
        x3 = self.up3(x2)
        s3 = self.skip_l2(z2)
        x3 = self.fuse3(torch.cat([x3, s3], dim=1))

        # Stage 4: 64^3 -> 128^3
        x4 = self.up4(x3)
        out = self.head(x4)

        if self.deep_supervision and self.training:
            return [out, self.ds3(x3), self.ds2(x2), self.ds1(x1)]
        return out


class HybridUNETRDecoder3D(nn.Module):
    r"""
    UNETR-Style Hybrid Decoder: pre-trained ViT-FPN stream + native convolutional stem.

    Mathematical Rationale & Defense Context (Hatamizadeh et al., WACV 2022):
    -------------------------------------------------------------------------
    Pure latent upsampling from the 8^3 token grid forces the final 64^3 -> 128^3
    stage to hallucinate single-voxel boundaries by 16x interpolation from tokens
    that each summarize 16^3 = 4,096 voxels. A lightweight convolutional stem
    processes the raw 4-channel volume at native 128^3 (16 ch) and 64^3 (32 ch)
    and fuses these high-frequency edge maps into decoder stages 3 and 4:

        Input (4 x 128^3) -> stem_128 (16ch @128^3) - - - -> fuse4 (24+16 -> 24)
                                -> stem_64 (32ch @64^3) - -> fuse3 (48+48+32 -> 48)

    Stages 1-2, all ViT lateral skips, DS heads, and train/eval conventions are
    identical to MultiScaleViTSegmentationDecoder3D, so pre-trained FPN weights
    transfer with strict=False (only fuse3/fuse4 differ in shape plus the fresh
    stem; up4/head are shape-identical to the multiscale decoder and transfer).
    """

    def __init__(
        self,
        in_dim: int = 384,
        out_channels: int = 1,
        in_channels: int = 4,
        grid_size: tuple[int, int, int] = (8, 8, 8),
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.grid_size = grid_size
        self.deep_supervision = deep_supervision

        # Native-resolution convolutional stem (trained fresh in finetuning).
        self.stem_128 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.stem_64 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )

        # Stage 1: Bottleneck 8^3 -> 16^3 (identical to multiscale FPN).
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(in_dim, 192, kernel_size=2, stride=2),
            nn.GroupNorm(16, 192),
            nn.GELU(),
        )
        self.skip_l6 = nn.Sequential(
            nn.ConvTranspose3d(in_dim, 192, kernel_size=2, stride=2),
            nn.GroupNorm(16, 192),
            nn.GELU(),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv3d(192 + 192, 192, kernel_size=3, padding=1),
            nn.GroupNorm(16, 192),
            nn.GELU(),
        )

        # Stage 2: 16^3 -> 32^3 (identical to multiscale FPN).
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(192, 96, kernel_size=2, stride=2),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.skip_l4 = nn.Sequential(
            nn.ConvTranspose3d(in_dim, 96, kernel_size=4, stride=4),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv3d(96 + 96, 96, kernel_size=3, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )

        # Stage 3: 32^3 -> 64^3 with stem_64 fusion (48 + 48 + 32 -> 48).
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2),
            nn.GroupNorm(4, 48),
            nn.GELU(),
        )
        self.skip_l2 = nn.Sequential(
            nn.ConvTranspose3d(in_dim, 96, kernel_size=4, stride=4),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2),
            nn.GroupNorm(4, 48),
            nn.GELU(),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv3d(48 + 48 + 32, 48, kernel_size=3, padding=1),
            nn.GroupNorm(4, 48),
            nn.GELU(),
        )

        # Stage 4: 64^3 -> 128^3 with stem_128 fusion (24 + 16 -> 24).
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2),
            nn.GroupNorm(4, 24),
            nn.GELU(),
        )
        self.fuse4 = nn.Sequential(
            nn.Conv3d(24 + 16, 24, kernel_size=3, padding=1),
            nn.GroupNorm(4, 24),
            nn.GELU(),
        )

        # Final projection to output logits
        self.head = nn.Conv3d(24, out_channels, kernel_size=1)

        if deep_supervision:
            self.ds3 = nn.Conv3d(48, out_channels, kernel_size=1)
            self.ds2 = nn.Conv3d(96, out_channels, kernel_size=1)
            self.ds1 = nn.Conv3d(192, out_channels, kernel_size=1)

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """Converts [B, N=512, D] patch tokens to [B, D, 8, 8, 8] spatial feature map."""
        B, _N, D = tokens.shape
        gz, gy, gx = self.grid_size
        return tokens.permute(0, 2, 1).reshape(B, D, gz, gy, gx)

    def forward(
        self,
        intermediate_tokens: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
        raw_volume: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(intermediate_tokens, (list, tuple)):
            n_layers = len(intermediate_tokens)
            idx2 = max(0, n_layers // 4 - 1)
            idx4 = max(0, n_layers // 2 - 1)
            idx6 = max(0, 3 * n_layers // 4 - 1)
            idx8 = n_layers - 1
            z2 = self._tokens_to_spatial(intermediate_tokens[idx2])
            z4 = self._tokens_to_spatial(intermediate_tokens[idx4])
            z6 = self._tokens_to_spatial(intermediate_tokens[idx6])
            z8 = self._tokens_to_spatial(intermediate_tokens[idx8])
        else:
            z = self._tokens_to_spatial(intermediate_tokens)
            z2 = z4 = z6 = z8 = z

        # Stage 1: 8^3 -> 16^3
        x1 = self.up1(z8)
        s1 = self.skip_l6(z6)
        x1 = self.fuse1(torch.cat([x1, s1], dim=1))

        # Stage 2: 16^3 -> 32^3
        x2 = self.up2(x1)
        s2 = self.skip_l4(z4)
        x2 = self.fuse2(torch.cat([x2, s2], dim=1))

        # Native stem features (zeros when raw volume is unavailable, e.g. probing).
        if raw_volume is not None:
            f128 = self.stem_128(raw_volume)
            f64 = self.stem_64(f128)
        else:
            B = x2.shape[0]
            dev, dt = x2.device, x2.dtype
            s64 = x2.shape[-3:]
            s128 = (s64[0] * 2, s64[1] * 2, s64[2] * 2)
            f64 = torch.zeros(B, 32, *s64, device=dev, dtype=dt)
            f128 = torch.zeros(B, 16, *s128, device=dev, dtype=dt)

        # Stage 3: 32^3 -> 64^3 with stem fusion
        x3 = self.up3(x2)
        s3 = self.skip_l2(z2)
        x3 = self.fuse3(torch.cat([x3, s3, f64], dim=1))

        # Stage 4: 64^3 -> 128^3 with stem fusion
        x4 = self.up4(x3)
        x4 = self.fuse4(torch.cat([x4, f128], dim=1))
        out = self.head(x4)

        if self.deep_supervision and self.training:
            return [out, self.ds3(x3), self.ds2(x2), self.ds1(x1)]
        return out


class JEPASegmentationModel3D(nn.Module):
    r"""
    Unified Downstream 3D Volumetric Segmentation Architecture.
    Couples pre-trained 3D ViT Encoder with either Bottleneck or Hierarchical FPN Decoder.
    Supports full fine-tuning and frozen encoder linear/decoder probing.
    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (128, 128, 128),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 4,
        embed_dim: int = 384,
        encoder_depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        out_channels: int = 1,
        freeze_encoder: bool = False,
        decoder_type: str = "multiscale",
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.encoder = VisionTransformerEncoder3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.decoder_type = decoder_type
        self.deep_supervision = deep_supervision

        if decoder_type == "multiscale":
            self.decoder = MultiScaleViTSegmentationDecoder3D(
                in_dim=embed_dim,
                out_channels=out_channels,
                grid_size=self.encoder.grid_size,
                deep_supervision=deep_supervision,
            )
        elif decoder_type == "unetr_hybrid":
            self.decoder = HybridUNETRDecoder3D(
                in_dim=embed_dim,
                out_channels=out_channels,
                in_channels=in_channels,
                grid_size=self.encoder.grid_size,
                deep_supervision=deep_supervision,
            )
        else:
            self.decoder = ViTSegmentationDecoder3D(
                in_dim=embed_dim,
                out_channels=out_channels,
                grid_size=self.encoder.grid_size,
            )

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if decoder_type == "bottleneck" and deep_supervision:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "bottleneck decoder has no DS heads — training without deep supervision."
            )

    def load_pretrained_encoder(self, state_dict: dict):
        r"""
        Loads pre-trained SSL JEPA encoder weights with robust prefix stripping and validation.
        Accepts full checkpoint dicts (with 'encoder_state_dict' or 'model_state_dict') or state dicts.
        Strips common prefixes ('context_encoder.', 'encoder.', 'backbone.', 'module.').
        Raises RuntimeError if no matching keys are found.
        """
        if "encoder_state_dict" in state_dict:
            raw_dict = state_dict["encoder_state_dict"]
        elif "model_state_dict" in state_dict:
            raw_dict = state_dict["model_state_dict"]
        else:
            raw_dict = state_dict

        target_keys = set(self.encoder.state_dict().keys())
        clean_dict = {}
        prefixes = ("context_encoder.", "encoder.", "backbone.", "module.", "_orig_mod.")
        for k, v in raw_dict.items():
            new_k = k
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]
                        changed = True
            if new_k in target_keys:
                clean_dict[new_k] = v

        if len(clean_dict) == 0:
            raise RuntimeError(
                f"No matching encoder weights found in provided state dict. "
                f"Available keys sample: {list(raw_dict.keys())[:5]}, "
                f"expected encoder keys sample: {list(target_keys)[:5]}"
            )

        missing, unexpected = self.encoder.load_state_dict(clean_dict, strict=False)
        return {"loaded_keys": len(clean_dict), "missing_keys": missing, "unexpected_keys": unexpected}

    def train(self, mode: bool = True):
        """Override to keep frozen encoder in eval mode (freezing LayerNorm stats and dropout)."""
        super().train(mode)
        if self.freeze_encoder and mode:
            self.encoder.eval()
        return self

    def _encode(self, x: torch.Tensor, return_intermediate: bool = True):
        """Encoder pass honoring freeze probing (no-grad) without branch duplication."""
        ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with ctx:
            return self.encoder(x, return_intermediate=return_intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        if self.decoder_type == "multiscale":
            _, intermediates = self._encode(x, return_intermediate=True)
            return self.decoder(intermediates)
        elif self.decoder_type == "unetr_hybrid":
            _, intermediates = self._encode(x, return_intermediate=True)
            return self.decoder(intermediates, raw_volume=x)
        else:
            tokens = self._encode(x, return_intermediate=False)
            return self.decoder(tokens)
