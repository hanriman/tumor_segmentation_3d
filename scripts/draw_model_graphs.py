#!/usr/bin/env python3
"""Render torchview architecture graphs (SVG, vector) for every thesis_3d model.

Models (all eval mode, CPU, real input [1,4,128,128,128]):
  1. visreg_pretrain : VisRegJEPA3D (encoder + predictor + projector)
  2. vit_fpn         : JEPASegmentationModel3D, multiscale decoder, 5ch
  3. vit_unetr       : JEPASegmentationModel3D, unetr_hybrid decoder, 5ch
  4. resunet         : BraTS3DUNet (MONAI UNet), 5ch
  5. dynunet         : BraTS3DnnUNet (MONAI DynUNet), 5ch

Usage:
    uv run python scripts/draw_model_graphs.py [--depth 3] [--outdir outputs/graphs]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from torchview import draw_graph  # noqa: E402

from brats_jepa_3d.models.nnunet_3d import BraTS3DnnUNet  # noqa: E402
from brats_jepa_3d.models.segmentation_head_3d import (  # noqa: E402
    JEPASegmentationModel3D,
)
from brats_jepa_3d.models.unet_3d import BraTS3DUNet  # noqa: E402
from brats_jepa_3d.models.visreg_jepa_3d import VisRegJEPA3D  # noqa: E402


class PretrainWrapper(nn.Module):
    """Single-tensor entry for VisRegJEPA3D (fixed mask; shapes unaffected)."""

    def __init__(self, model: VisRegJEPA3D):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        ctx = torch.arange(192).unsqueeze(0).expand(B, -1)
        tgts = [
            torch.arange(i * 27, (i + 1) * 27).unsqueeze(0).expand(B, -1)
            for i in range(4)
        ]
        out = self.model(images, ctx, tgts)
        return torch.stack(out["predictions"], dim=0)  # [4,B,27,384]


def build_models() -> dict[str, nn.Module]:
    vit_common = dict(
        img_size=(128, 128, 128),
        patch_size=(8, 8, 8),
        in_channels=4,
        embed_dim=384,
        encoder_depth=8,
        num_heads=6,
        out_channels=5,
        deep_supervision=True,
    )
    return {
        "visreg_pretrain": PretrainWrapper(VisRegJEPA3D()),
        "vit_fpn": JEPASegmentationModel3D(
            **vit_common, decoder_type="multiscale"
        ),
        "vit_unetr": JEPASegmentationModel3D(
            **vit_common, decoder_type="unetr_hybrid"
        ),
        "resunet": BraTS3DUNet(
            in_channels=4,
            out_channels=5,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.1,
            deep_supervision=True,
        ),
        "dynunet": BraTS3DnnUNet(
            in_channels=4,
            out_channels=5,
            deep_supervision=True,
            deep_supr_num=3,
            res_block=True,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--outdir", type=str, default="outputs/graphs")
    args = ap.parse_args()

    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 4, 128, 128, 128)

    for name, model in build_models().items():
        model.eval()
        t0 = time.time()
        print(f"[{name}] tracing...", flush=True)
        try:
            g = draw_graph(
                model,
                input_data=x,
                graph_name=name,
                depth=args.depth,
                save_graph=False,
            )
            g.visual_graph.render(
                str(outdir / name), format="svg", cleanup=True
            )
            print(f"[{name}] ok in {time.time()-t0:.0f}s -> {outdir / (name + '.svg')}")
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
