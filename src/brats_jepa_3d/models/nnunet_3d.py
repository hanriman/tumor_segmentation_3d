import torch
from monai.networks.nets import DynUNet
from torch import nn


class BraTS3DnnUNet(nn.Module):
    r"""
    3D nnU-Net Baseline Architecture (Isensee et al., Nature Methods 2021).

    Mathematical Rationale & Architecture:
    --------------------------------------
    Uses MONAI DynUNet with deep supervision, leaky ReLU activations, instance normalization,
    and residual blocks. Multi-scale heads output predictions at 128^3, 64^3, 32^3, 16^3
    to provide gradient highways that prevent vanishing gradients during volumetric training.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        kernel_size: list[list[int]] | None = None,
        strides: list[list[int]] | None = None,
        upsample_kernel_size: list[list[int]] | None = None,
        filters: list[int] | None = None,
        deep_supervision: bool = True,
        deep_supr_num: int = 3,
        res_block: bool = True,
    ):
        super().__init__()
        if kernel_size is None:
            kernel_size = [[3, 3, 3]] * 5
        if strides is None:
            strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        if upsample_kernel_size is None:
            upsample_kernel_size = [[2, 2, 2]] * 4
        if filters is None:
            filters = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision
        self.dynunet = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=filters,
            norm_name="instance",
            act_name="leakyrelu",
            deep_supervision=deep_supervision,
            deep_supr_num=deep_supr_num,
            res_block=res_block,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        out = self.dynunet(x)
        if isinstance(out, torch.Tensor) and out.dim() == 6:
            if self.training and self.deep_supervision:
                return [out[:, i] for i in range(out.shape[1])]
            return out[:, 0]
        return out
