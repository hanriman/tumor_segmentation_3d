#!/usr/bin/env python3
"""
scripts/figures/plot_downstream_fpn_3d.py
-----------------------------------------
Generates 3D architecture diagram for ViT + Multi-Scale FPN Decoder (24.82M params).
Features:
- Pure Latent Probe: evaluates self-supervised ViT representations with zero raw image skip connections
- Intermediate lateral taps from ViT layers L2, L4, L6, L8
- Flush progressive 3D transposed convolution upsampling stages
- Publication-grade PlotNeuralNet aesthetics with classic cohesive palette
"""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    to_custom_head,
    to_custom_cor,
    to_custom_begin,
    to_custom_Conv,
    to_custom_ConvConvRelu,
    to_custom_UnPool,
    to_custom_Softmax,
    compile_tex_to_pdf_and_png,
    export_architecture_figure,
    to_connection,
    to_skip,
    to_end,
    to_generate,
    to_ViT_block,
)


def generate_fpn_3d_arch():
    arch = [
        to_custom_head(),
        to_custom_cor(),
        to_custom_begin(),

        # Input Volume: 4 MRI channels @ 128^3
        to_custom_Conv(
            name="input_vol",
            s_filer="128",
            n_filer="4",
            offset="(0,0,0)",
            to="(0,0,0)",
            width=2.0,
            height=36,
            depth=36,
            fill=r"\InputColor",
            caption="Input MRI (4x128)",
        ),

        # 3D Patch Embedding: Conv3d (kernel 8^3, stride 8^3) -> 4096 tokens, D=384
        to_custom_Conv(
            name="patch_emb",
            s_filer="16",
            n_filer="384",
            offset="(2.0, 0, 0)",
            to="(input_vol-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="PatchEmbed (8x8x8)",
        ),
        to_connection("input_vol", "patch_emb"),

        # ViT Transformer Backbone: 8 Layers (D=384, 6 heads)
        to_ViT_block(
            name="vit_l1",
            s_filer="16",
            n_filer="384",
            offset="(1.5, 0, 0)",
            to="(patch_emb-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="ViT L1",
        ),
        to_connection("patch_emb", "vit_l1"),

        to_ViT_block(
            name="vit_l2",
            s_filer="16",
            n_filer="384",
            offset="(1.0, 0, 0)",
            to="(vit_l1-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="ViT L2 (Tap)",
        ),
        to_connection("vit_l1", "vit_l2"),

        to_ViT_block(
            name="vit_l3",
            s_filer="16",
            n_filer="384",
            offset="(1.0, 0, 0)",
            to="(vit_l2-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="ViT L3",
        ),
        to_connection("vit_l2", "vit_l3"),

        to_ViT_block(
            name="vit_l4",
            s_filer="16",
            n_filer="384",
            offset="(1.0, 0, 0)",
            to="(vit_l3-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="ViT L4 (Tap)",
        ),
        to_connection("vit_l3", "vit_l4"),

        to_ViT_block(
            name="vit_l5",
            s_filer="16",
            n_filer="384",
            offset="(1.0, 0, 0)",
            to="(vit_l4-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="ViT L5",
        ),
        to_connection("vit_l4", "vit_l5"),

        to_ViT_block(
            name="vit_l6",
            s_filer="16",
            n_filer="384",
            offset="(1.0, 0, 0)",
            to="(vit_l5-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="ViT L6 (Tap)",
        ),
        to_connection("vit_l5", "vit_l6"),

        to_ViT_block(
            name="vit_l7",
            s_filer="16",
            n_filer="384",
            offset="(1.0, 0, 0)",
            to="(vit_l6-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="ViT L7",
        ),
        to_connection("vit_l6", "vit_l7"),

        to_ViT_block(
            name="vit_l8",
            s_filer="16",
            n_filer="384",
            offset="(1.0, 0, 0)",
            to="(vit_l7-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="ViT L8 (Bottleneck)",
        ),
        to_connection("vit_l7", "vit_l8"),

        # --- MULTI-SCALE FPN DECODER (Flush Transposed Convs + Lateral Fusion) ---
        # Stage 1: Bottleneck 16^3 + Skip L6 (384 -> 192)
        to_custom_Conv(
            name="fpn_stage1",
            s_filer="16",
            n_filer="192",
            offset="(2.0, 0, 0)",
            to="(vit_l8-east)",
            width=6.0,
            height=14,
            depth=14,
            caption="Stage 1 Fuse (192ch)",
        ),
        to_connection("vit_l8", "fpn_stage1"),
        to_skip(of="vit_l6", to="fpn_stage1", pos=1.25),

        # Stage 2: 16^3 -> 32^3 (TransConv 192 -> 96) + Skip L4 (384 -> 96)
        to_custom_UnPool(
            name="up2",
            offset="(1.8, 0, 0)",
            to="(fpn_stage1-east)",
            width=1.2,
            height=20,
            depth=20,
        ),
        to_connection("fpn_stage1", "up2"),
        to_custom_ConvConvRelu(
            name="fpn_stage2",
            s_filer="32",
            n_filer=(96, 96),
            offset="(0, 0, 0)",
            to="(up2-east)",
            width=(2.5, 2.5),
            height=20,
            depth=20,
            caption="Stage 2 Fuse (96ch)",
        ),
        to_skip(of="vit_l4", to="fpn_stage2", pos=1.35),

        # Stage 3: 32^3 -> 64^3 (TransConv 96 -> 48) + Skip L2 (384 -> 48)
        to_custom_UnPool(
            name="up3",
            offset="(1.8, 0, 0)",
            to="(fpn_stage2-east)",
            width=1.2,
            height=28,
            depth=28,
        ),
        to_connection("fpn_stage2", "up3"),
        to_custom_ConvConvRelu(
            name="fpn_stage3",
            s_filer="64",
            n_filer=(48, 48),
            offset="(0, 0, 0)",
            to="(up3-east)",
            width=(2.0, 2.0),
            height=28,
            depth=28,
            caption="Stage 3 Fuse (48ch)",
        ),
        to_skip(of="vit_l2", to="fpn_stage3", pos=1.45),

        # Stage 4: 64^3 -> 128^3 (TransConv 48 -> 24)
        to_custom_UnPool(
            name="up4",
            offset="(1.8, 0, 0)",
            to="(fpn_stage3-east)",
            width=1.2,
            height=36,
            depth=36,
        ),
        to_connection("fpn_stage3", "up4"),
        to_custom_Conv(
            name="fpn_stage4",
            s_filer="128",
            n_filer="24",
            offset="(0, 0, 0)",
            to="(up4-east)",
            width=4.0,
            height=36,
            depth=36,
            caption="Stage 4 (24ch)",
        ),

        # Final Output Head: 5 channels @ 128^3
        to_custom_Softmax(
            name="final_head",
            s_filer="128",
            n_filer="5",
            offset="(1.5, 0, 0)",
            to="(fpn_stage4-east)",
            width=2.0,
            height=36,
            depth=36,
            caption="Output Head (5ch)",
        ),
        to_connection("fpn_stage4", "final_head"),

        to_end(),
    ]
    return arch


def main():
    arch = generate_fpn_3d_arch()
    export_architecture_figure(arch, "arch_fpn_decoder_3d", width_px=3000)


if __name__ == "__main__":
    main()
