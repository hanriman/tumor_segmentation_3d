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
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
        dropout: float = 0.1,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)
