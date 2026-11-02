#!/usr/bin/env python3
"""
scripts/figures/plot_jepa_pretrain_3d.py
----------------------------------------
Generates 3D architecture diagram for Single-Encoder 3D VISReg JEPA Pre-Training (19.80M params).
Depicts:
- 3D Patch Embedding (8^3) + 3D Positional Embeddings
- Context (1,536 tokens, 37.5 pct) vs Target (6 anisotropic cuboids, 776 tokens)
- ViT Context Encoder (8 layers, D=384)
- Auxiliary Projector (384 -> 1024 -> 128) with VISReg Sliced-Wasserstein Loss
- Narrow Predictor (4 layers, D=192) with Target Position Queries -> JEPA Loss
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
    compile_tex_to_pdf_and_png,
    export_architecture_figure,
    to_connection,
    to_end,
    to_generate,
    to_ViT_block,
    to_Predictor_block,
    to_Projector_block,
)


def generate_jepa_pretrain_3d_arch():
    arch = [
        to_custom_head(),
        to_custom_cor(),
        to_custom_begin(),

        # 1. Input Volume: 4 MRI channels @ 128^3
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

        # 2. 3D Patch Embedding + 3D Positional Embeddings: 4,096 tokens, D=384
        to_custom_Conv(
            name="patch_emb",
            s_filer="16",
            n_filer="384",
            offset="(2.2, 0, 0)",
            to="(input_vol-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="PatchEmbed + 3D PE",
        ),
        to_connection("input_vol", "patch_emb"),

        # 3. Visible Context Gathering: 1,536 tokens (37.5 pct volume via 3D BFS)
        to_custom_Conv(
            name="ctx_tokens",
            s_filer="1536",
            n_filer="384",
            offset="(2.2, 2.5, 0)",
            to="(patch_emb-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="Context Gather (37.5 pct)",
        ),
        to_connection("patch_emb", "ctx_tokens"),

        # 4. ViT Context Encoder: 8 Transformer Blocks (D=384, 6 heads)
        to_ViT_block(
            name="vit_l1",
            s_filer="1536",
            n_filer="384",
            offset="(1.5, 0, 0)",
            to="(ctx_tokens-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="ViT L1",
        ),
        to_connection("ctx_tokens", "vit_l1"),

        to_ViT_block(
            name="vit_l2",
            s_filer="1536",
            n_filer="384",
            offset="(1.0, 0, 0)",
            to="(vit_l1-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="ViT L2",
        ),
        to_connection("vit_l1", "vit_l2"),

        to_ViT_block(
            name="vit_l3",
            s_filer="1536",
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
            s_filer="1536",
            n_filer="384",
            offset="(1.0, 0, 0)",
            to="(vit_l3-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="ViT L4",
        ),
        to_connection("vit_l3", "vit_l4"),

        to_ViT_block(
            name="vit_l5",
            s_filer="1536",
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
            s_filer="1536",
            n_filer="384",
            offset="(1.0, 0, 0)",
            to="(vit_l5-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="ViT L6",
        ),
        to_connection("vit_l5", "vit_l6"),

        to_ViT_block(
            name="vit_l7",
            s_filer="1536",
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
            s_filer="1536",
            n_filer="384",
            offset="(1.0, 0, 0)",
            to="(vit_l7-east)",
            width=8.0,
            height=14,
            depth=14,
            caption="ViT L8",
        ),
        to_connection("vit_l7", "vit_l8"),

        # 5. Auxiliary Projector MLP: 384 -> 1024 -> 128 (branching above ViT Encoder L8)
        to_Projector_block(
            name="projector",
            s_filer="1536",
            n_filer="128",
            offset="(0, 3.5, 0)",
            to="(vit_l8-north)",
            width=6.0,
            height=14,
            depth=14,
            caption="Projector MLP (128d)",
        ),
        r"\draw [connection] (vit_l8-north) -- node {\midarrow} (projector-south);",

        # VISReg Loss Node
        to_custom_Conv(
            name="visreg_loss",
            s_filer="1024",
            n_filer="128",
            offset="(2.0, 0, 0)",
            to="(projector-east)",
            width=2.0,
            height=8,
            depth=8,
            fill=r"\LossColor",
            caption="L VISReg (SWD)",
        ),
        to_connection("projector", "visreg_loss"),

        # 6. Target Position Queries + Context Proj into Narrow Predictor
        to_Predictor_block(
            name="predictor",
            s_filer="776",
            n_filer="192",
            offset="(2.2, 0, 0)",
            to="(vit_l8-east)",
            width=6.0,
            height=10,
            depth=10,
            caption="Predictor (4 Blocks)",
        ),
        to_connection("vit_l8", "predictor"),

        # 7. Predicted Target Representations (Y_hat)
        to_custom_Conv(
            name="pred_targets",
            s_filer="776",
            n_filer="384",
            offset="(2.0, 0, 0)",
            to="(predictor-east)",
            width=8.0,
            height=10,
            depth=10,
            caption="Predictions (Y hat)",
        ),
        to_connection("predictor", "pred_targets"),

        # 8. Target Extraction Path (branching from patch_emb, positioned under predictor/targets)
        # target_gather: 6 Anisotropic Cuboids (776 tokens)
        to_custom_Conv(
            name="target_gather",
            s_filer="776",
            n_filer="384",
            offset="(-3.6, -4.5, 0)",
            to="(pred_targets-west)",
            width=8.0,
            height=10,
            depth=10,
            caption="Target Gather (6 Cuboids)",
        ),
        r"\draw [connection] (patch_emb-south) |- (target_gather-west);",

        # Target Encoder (no_grad pass / shared weights) - directly aligned under pred_targets
        to_ViT_block(
            name="target_enc",
            s_filer="776",
            n_filer="384",
            offset="(2.0, 0, 0)",
            to="(target_gather-east)",
            width=8.0,
            height=10,
            depth=10,
            caption="Target Pass (no grad)",
        ),
        to_connection("target_gather", "target_enc"),

        # 9. JEPA Prediction Loss Node (Smooth-L1)
        to_custom_Conv(
            name="jepa_loss",
            s_filer="384",
            n_filer="1",
            offset="(2.5, -2.25, 0)",
            to="(pred_targets-east)",
            width=2.0,
            height=8,
            depth=8,
            fill=r"\LossColor",
            caption="L JEPA (Smooth L1)",
        ),
        to_connection("pred_targets", "jepa_loss"),
        r"\draw [connection] (target_enc-east) -- node {\midarrow} (jepa_loss-southwest);",

        to_end(),
    ]
    return arch


def main():
    arch = generate_jepa_pretrain_3d_arch()
    export_architecture_figure(arch, "arch_jepa_pretrain_3d", width_px=3000)


if __name__ == "__main__":
    main()
