import contextlib

import torch
from torch import nn

from .vision_transformer_3d import VisionTransformerEncoder3D


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
        out_channels: int = 5,
        grid_size: tuple[int, int, int] = (8, 8, 8),
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.grid_size = grid_size
        self.deep_supervision = deep_supervision
        gz = grid_size[0]

        if gz == 16:
            # High-resolution 16³ grid (patch_size 8³ on 128³ volume):
            # Latent tokens are already at 16³ resolution.
            # Stage 1: 16³ -> 16³ (stride 1 conv)
            self.up1 = nn.Sequential(
                nn.Conv3d(in_dim, 192, kernel_size=3, padding=1),
                nn.GroupNorm(16, 192),
                nn.GELU(),
            )
            # Skip connection from L6 (16³ -> 16³)
            self.skip_l6 = nn.Sequential(
                nn.Conv3d(in_dim, 192, kernel_size=3, padding=1),
                nn.GroupNorm(16, 192),
                nn.GELU(),
            )
            self.fuse1 = nn.Sequential(
                nn.Conv3d(192 + 192, 192, kernel_size=3, padding=1),
                nn.GroupNorm(16, 192),
                nn.GELU(),
            )

            # Stage 2: 16³ -> 32³ (stride 2)
            self.up2 = nn.Sequential(
                nn.ConvTranspose3d(192, 96, kernel_size=2, stride=2),
                nn.GroupNorm(8, 96),
                nn.GELU(),
            )
            # Skip connection from L4 (16³ -> 32³ via 2x transpose conv)
            self.skip_l4 = nn.Sequential(
                nn.ConvTranspose3d(in_dim, 96, kernel_size=2, stride=2),
                nn.GroupNorm(8, 96),
                nn.GELU(),
            )
            self.fuse2 = nn.Sequential(
                nn.Conv3d(96 + 96, 96, kernel_size=3, padding=1),
                nn.GroupNorm(8, 96),
                nn.GELU(),
            )

            # Stage 3: 32³ -> 64³ (stride 2)
            self.up3 = nn.Sequential(
                nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2),
                nn.GroupNorm(4, 48),
                nn.GELU(),
            )
            # Skip connection from L2 (16³ -> 64³ via 4x transpose conv)
            self.skip_l2 = nn.Sequential(
                nn.ConvTranspose3d(in_dim, 48, kernel_size=4, stride=4),
                nn.GroupNorm(4, 48),
                nn.GELU(),
            )
            self.fuse3 = nn.Sequential(
                nn.Conv3d(48 + 48, 48, kernel_size=3, padding=1),
                nn.GroupNorm(4, 48),
                nn.GELU(),
            )
        else:
            # Canonical 8³ grid (patch_size 16³ on 128³ volume):
            # Stage 1: Bottleneck 8³ -> 16³
            self.up1 = nn.Sequential(
                nn.ConvTranspose3d(in_dim, 192, kernel_size=2, stride=2),
                nn.GroupNorm(16, 192),
                nn.GELU(),
            )
            # Skip connection from L6 (8³ -> 16³)
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

            # Stage 2: 16³ -> 32³
            self.up2 = nn.Sequential(
                nn.ConvTranspose3d(192, 96, kernel_size=2, stride=2),
                nn.GroupNorm(8, 96),
                nn.GELU(),
            )
            # Skip connection from L4 (8³ -> 32³ via 4x transpose conv)
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

            # Stage 3: 32³ -> 64³
            self.up3 = nn.Sequential(
                nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2),
                nn.GroupNorm(4, 48),
                nn.GELU(),
            )
            # Skip connection from L2 (8³ -> 32³ -> 64³ via progressive transpose convs)
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

        # Stage 4: 64³ -> 128³
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
        """Converts [B, N, D] patch tokens to [B, D, gz, gy, gx] spatial feature map."""
        B, _N, D = tokens.shape
        gz, gy, gx = self.grid_size
        return tokens.permute(0, 2, 1).reshape(B, D, gz, gy, gx)

    def _extract_spatial_stages(
        self,
        intermediate_tokens: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return z2, z4, z6, z8

    def forward(
        self,
        intermediate_tokens: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        z2, z4, z6, z8 = self._extract_spatial_stages(intermediate_tokens)

        # Stage 1: bottleneck -> 16^3
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


class HybridUNETRDecoder3D(MultiScaleViTSegmentationDecoder3D):
    r"""
    UNETR-Style Hybrid Decoder: pre-trained ViT-FPN stream + native convolutional stem.

    Mathematical Rationale & Defense Context (Hatamizadeh et al., WACV 2022):
    -------------------------------------------------------------------------
    Subclasses `MultiScaleViTSegmentationDecoder3D` and adds a lightweight native-resolution
    convolutional stem (`stem_128` at 128^3 with 16ch, `stem_64` at 64^3 with 32ch) that fuses
    into stages 3 (`fuse3`: 48+48+32 -> 48) and 4 (`fuse4`: 24+16 -> 24):

        Input (4 x 128^3) -> stem_128 (16ch @128^3) - - - -> fuse4 (24+16 -> 24)
                                -> stem_64 (32ch @64^3) - -> fuse3 (48+48+32 -> 48)

    Stages 1-2, all ViT lateral skips, DS heads, and train/eval conventions are
    inherited directly from `MultiScaleViTSegmentationDecoder3D`.
    """

    def __init__(
        self,
        in_dim: int = 384,
        out_channels: int = 5,
        in_channels: int = 4,
        grid_size: tuple[int, int, int] = (8, 8, 8),
        deep_supervision: bool = False,
    ):
        super().__init__(
            in_dim=in_dim,
            out_channels=out_channels,
            grid_size=grid_size,
            deep_supervision=deep_supervision,
        )

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

        # Override stage-3 fusion to include stem_64 (48 + 48 + 32 -> 48) and add stage-4 stem_128 fusion.
        self.fuse3 = nn.Sequential(
            nn.Conv3d(48 + 48 + 32, 48, kernel_size=3, padding=1),
            nn.GroupNorm(4, 48),
            nn.GELU(),
        )
        self.fuse4 = nn.Sequential(
            nn.Conv3d(24 + 16, 24, kernel_size=3, padding=1),
            nn.GroupNorm(4, 24),
            nn.GELU(),
        )

    def forward(
        self,
        intermediate_tokens: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
        raw_volume: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        z2, z4, z6, z8 = self._extract_spatial_stages(intermediate_tokens)

        # Stage 1: bottleneck -> 16^3
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
            s64 = (x2.shape[-3] * 2, x2.shape[-2] * 2, x2.shape[-1] * 2)
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


def load_downstream_state_dict(
    model: torch.nn.Module, sd: dict, label: str = "model"
) -> tuple[list, list]:
    """strict=False load with a loud channel-mismatch guard.

    C=1 checkpoints cannot load into C=5 heads (size mismatch still raises
    under strict=False); surface that as an actionable error naming the
    likely cause instead of a raw shape traceback.
    """
    try:
        return model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        raise RuntimeError(
            f"{label}: checkpoint incompatible with model architecture — most "
            f"often an out_channels mismatch (e.g. C=1 weights into a C=5 "
            f"model): retrain with matching config or point at the matching "
            f"checkpoint. Underlying error: {e}"
        ) from e


class JEPASegmentationModel3D(nn.Module):
    r"""
    Unified Downstream 3D Volumetric Segmentation Architecture.
    Couples a 3D ViT Encoder with either the Multi-Scale FPN Decoder (`multiscale`)
    or the Hybrid UNETR Decoder (`unetr_hybrid`).
    Supports full fine-tuning and frozen encoder decoder probing.
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
        out_channels: int = 5,
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
            raise ValueError(
                f"Unknown decoder_type '{decoder_type}'. Expected 'multiscale' or 'unetr_hybrid'."
            )

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

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
        _, intermediates = self._encode(x, return_intermediate=True)
        if self.decoder_type == "unetr_hybrid":
            return self.decoder(intermediates, raw_volume=x)
        return self.decoder(intermediates)
