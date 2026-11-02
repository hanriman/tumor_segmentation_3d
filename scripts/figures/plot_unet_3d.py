#!/usr/bin/env python3
"""
scripts/figures/plot_unet_3d.py
-------------------------------
Generates 3D isometric architecture diagram for 3D Residual UNet (BraTS3DUNet, 19.23M params).
Features:
- Dual-banded 3D residual convolution stages with proportional width (channel capacity) scaling
- Direct strided transitions without detached floating stride slices
- Flush transposed convolution blocks on decoder stages
- Multi-scale deep supervision heads (ds1, ds2, ds3) and final 5-channel output head
- Publication-grade PlotNeuralNet aesthetics
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
    to_ds_head,
    compile_tex_to_pdf_and_png,
    export_architecture_figure,
    to_connection,
    to_skip,
    to_end,
    to_generate,
)


def generate_unet_3d_arch():
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

        # --- ENCODER (Proportional length and height/depth scaling) ---
        # Stage 1: 32ch @ 128^3
        to_custom_ConvConvRelu(
            name="enc1",
            s_filer="128",
            n_filer=(32, 32),
            offset="(2.0, 0, 0)",
            to="(input_vol-east)",
            width=(2.0, 2.0),
            height=36,
            depth=36,
            caption="Enc 1 (32ch)",
        ),
        to_connection("input_vol", "enc1"),

        # Stage 2: 64ch @ 64^3 (Direct strided transition)
        to_custom_ConvConvRelu(
            name="enc2",
            s_filer="64",
            n_filer=(64, 64),
            offset="(1.8, 0, 0)",
            to="(enc1-east)",
            width=(2.5, 2.5),
            height=28,
            depth=28,
            caption="Enc 2 (64ch)",
        ),
        to_connection("enc1", "enc2"),

        # Stage 3: 128ch @ 32^3
        to_custom_ConvConvRelu(
            name="enc3",
            s_filer="32",
            n_filer=(128, 128),
            offset="(1.8, 0, 0)",
            to="(enc2-east)",
            width=(3.0, 3.0),
            height=20,
            depth=20,
            caption="Enc 3 (128ch)",
        ),
        to_connection("enc2", "enc3"),

        # Stage 4: 256ch @ 16^3
        to_custom_ConvConvRelu(
            name="enc4",
            s_filer="16",
            n_filer=(256, 256),
            offset="(1.8, 0, 0)",
            to="(enc3-east)",
            width=(3.5, 3.5),
            height=14,
            depth=14,
            caption="Enc 4 (256ch)",
        ),
        to_connection("enc3", "enc4"),

        # Bottleneck Stage 5: 512ch @ 8^3
        to_custom_ConvConvRelu(
            name="bottleneck",
            s_filer="8",
            n_filer=(512, 512),
            offset="(1.8, 0, 0)",
            to="(enc4-east)",
            width=(4.5, 4.5),
            height=8,
            depth=8,
            caption="Bottleneck (512ch)",
        ),
        to_connection("enc4", "bottleneck"),

        # --- DECODER (Flush Transposed Conv + Arched Skips + Deep Supervision) ---
        # Stage 4: 256ch @ 16^3
        to_custom_UnPool(
            name="up4",
            offset="(1.8, 0, 0)",
            to="(bottleneck-east)",
            width=1.2,
            height=14,
            depth=14,
        ),
        to_connection("bottleneck", "up4"),
        to_custom_ConvConvRelu(
            name="dec4",
            s_filer="16",
            n_filer=(256, 256),
            offset="(0, 0, 0)",
            to="(up4-east)",
            width=(3.5, 3.5),
            height=14,
            depth=14,
            caption="Dec 4 (256ch)",
        ),
        to_skip(of="enc4", to="dec4", pos=1.25),
        to_ds_head(
            name="ds1_head",
            of="dec4",
            offset="(0, -2.5, 0)",
            s_filer=16,
            n_filer=5,
            width=2.0,
            height=9,
            depth=9,
            caption="DS1 (16)",
        ),

        # Stage 3: 128ch @ 32^3
        to_custom_UnPool(
            name="up3",
            offset="(1.8, 0, 0)",
            to="(dec4-east)",
            width=1.2,
            height=20,
            depth=20,
        ),
        to_connection("dec4", "up3"),
        to_custom_ConvConvRelu(
            name="dec3",
            s_filer="32",
            n_filer=(128, 128),
            offset="(0, 0, 0)",
            to="(up3-east)",
            width=(3.0, 3.0),
            height=20,
            depth=20,
            caption="Dec 3 (128ch)",
        ),
        to_skip(of="enc3", to="dec3", pos=1.25),
        to_ds_head(
            name="ds2_head",
            of="dec3",
            offset="(0, -3.2, 0)",
            s_filer=32,
            n_filer=5,
            width=2.0,
            height=12,
            depth=12,
            caption="DS2 (32)",
        ),

        # Stage 2: 64ch @ 64^3
        to_custom_UnPool(
            name="up2",
            offset="(1.8, 0, 0)",
            to="(dec3-east)",
            width=1.2,
            height=28,
            depth=28,
        ),
        to_connection("dec3", "up2"),
        to_custom_ConvConvRelu(
            name="dec2",
            s_filer="64",
            n_filer=(64, 64),
            offset="(0, 0, 0)",
            to="(up2-east)",
            width=(2.5, 2.5),
            height=28,
            depth=28,
            caption="Dec 2 (64ch)",
        ),
        to_skip(of="enc2", to="dec2", pos=1.25),
        to_ds_head(
            name="ds3_head",
            of="dec2",
            offset="(0, -3.8, 0)",
            s_filer=64,
            n_filer=5,
            width=2.0,
            height=15,
            depth=15,
            caption="DS3 (64)",
        ),

        # Stage 1: 32ch @ 128^3
        to_custom_UnPool(
            name="up1",
            offset="(1.8, 0, 0)",
            to="(dec2-east)",
            width=1.2,
            height=36,
            depth=36,
        ),
        to_connection("dec2", "up1"),
        to_custom_ConvConvRelu(
            name="dec1",
            s_filer="128",
            n_filer=(32, 32),
            offset="(0, 0, 0)",
            to="(up1-east)",
            width=(2.0, 2.0),
            height=36,
            depth=36,
            caption="Dec 1 (32ch)",
        ),
        to_skip(of="enc1", to="dec1", pos=1.25),

        # Final Output Head: 5 channels @ 128^3
        to_custom_Softmax(
            name="output_head",
            s_filer="128",
            n_filer="5",
            offset="(1.5, 0, 0)",
            to="(dec1-east)",
            width=2.0,
            height=36,
            depth=36,
            caption="Output Head (5ch)",
        ),
        to_connection("dec1", "output_head"),

        to_end(),
    ]
    return arch


def main():
    arch = generate_unet_3d_arch()
    export_architecture_figure(arch, "arch_unet_3d", width_px=3000)


if __name__ == "__main__":
    main()
