import torch
from monai.networks.nets import UNet
from torch import nn


class BraTS3DUNet(nn.Module):
    r"""
    3D Residual UNet Baseline for Multi-Modal Brain Glioma Segmentation.

    Mathematical Rationale & Architecture (Milletari et al., 3DV 2016; Ronneberger et al., 2015):
    ----------------------------------------------------------------------------------------------
    Uses MONAI UNet with spatial_dims=3, in_channels=4, out_channels=1.
    Encoder channels: (32, 64, 128, 256, 512) with 2 residual convolutional units per stage
    and Instance Normalization (Ulyanov et al., 2016).

    Multi-Scale Deep Supervision (Isensee et al., Nature Methods 2021):
    -------------------------------------------------------------------
    MONAI's UNet is a recursive module tree that does not expose intermediate decoder
    maps, so auxiliary taps are captured with shape-keyed forward hooks: every 5D
    submodule output is recorded and, per target resolution (16^3, 32^3, 64^3), the
    LAST recorded tensor (closest to the network output) is taken as the stage output.
    With the standard 5-level channel pyramid (32, 64, 128, 256, 512), decoder widths
    at (16^3, 32^3, 64^3) are (128, 64, 32) = channels[-3], channels[-4], channels[-5],
    each projected by a 1x1x1 head. In training mode with deep supervision enabled the
    forward returns [out_128, ds3_64, ds2_32, ds1_16] for DeepSupervisionLoss3D; in
    evaluation mode (or when disabled) it returns the single full-resolution tensor,
    preserving the exact pre-existing interface and checkpoint compatibility
    (new ds* keys simply append; old checkpoints load with strict=False).

    Version-sensitivity contract (verified against installed MONAI; see tests):
    the taps assume (a) decoder stage outputs keep encoder widths per resolution,
    (b) skip-merge modules subclass-name-match "Skip" (their concat outputs are
    excluded), and (c) the up-path block fires immediately before its skip parent.
    Any structural drift raises RuntimeError naming the captured shapes instead of
    silently miswiring heads.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
        dropout: float = 0.1,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm="instance",
            dropout=dropout,
        )
        self.out_channels = out_channels
        self.deep_supervision = bool(deep_supervision) and len(channels) >= 5

        self.ds1: nn.Conv3d | None = None  # 16^3 head
        self.ds2: nn.Conv3d | None = None  # 32^3 head
        self.ds3: nn.Conv3d | None = None  # 64^3 head
        if self.deep_supervision:
            in16, in32, in64 = channels[-3], channels[-4], channels[-5]
            self.ds1 = nn.Conv3d(in16, out_channels, kernel_size=1)
            self.ds2 = nn.Conv3d(in32, out_channels, kernel_size=1)
            self.ds3 = nn.Conv3d(in64, out_channels, kernel_size=1)

        # Shape-keyed capture buffer (populated by hooks during forward).
        self._ds_captured: list[torch.Tensor] = []

    def _make_hook(self):
        buf = self._ds_captured

        def _hook(mod: nn.Module, _inp: object, out: object) -> None:
            # Skip cat-style modules (e.g. MONAI SkipConnection concatenates the
            # skip input with the sub-block output, doubling channels); the true
            # stage output is the up-path Sequential firing immediately before.
            if isinstance(out, torch.Tensor) and out.dim() == 5 and "Skip" not in type(mod).__name__:
                buf.append((type(mod).__name__, out))

        return _hook

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        # (Re)bind capture hooks to the current buffer (hooks hold no stale state).
        if not hasattr(self, "_hooks_bound"):
            for mod in self.unet.modules():
                mod.register_forward_hook(self._make_hook())
            self._hooks_bound = True
        self._ds_captured.clear()

        out = self.unet(x)

        # Last recorded non-skip tensor per spatial resolution = stage output.
        # Drained in place (never rebound: hook closures hold this list object).
        # Draining also frees eval-mode intermediates instead of pinning them.
        captured = list(self._ds_captured)
        del self._ds_captured[:]
        last_by_shape: dict[tuple[int, int, int], torch.Tensor] = {}
        for _modname, t in captured:
            last_by_shape[tuple(t.shape[2:])] = t

        if not (self.deep_supervision and self.training):
            return out

        D = x.shape[2]
        heads: list[torch.Tensor] = [out]
        for edge, head in ((D // 2, self.ds3), (D // 4, self.ds2), (D // 8, self.ds1)):
            feat = last_by_shape.get((edge, edge, edge))
            if feat is None or head is None:
                raise RuntimeError(
                    f"Deep-supervision tap at {edge}^3 not found "
                    f"(captured shapes: {sorted(last_by_shape)})."
                )
            if feat.shape[1] != head.in_channels:
                raise RuntimeError(
                    f"DS head channel mismatch at {edge}^3: "
                    f"captured {feat.shape[1]} vs head {head.in_channels}."
                )
            heads.append(head(feat))
        return heads
