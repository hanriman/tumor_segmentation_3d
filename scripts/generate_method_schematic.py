#!/usr/bin/env python3
"""
scripts/generate_method_schematic.py
------------------------------------
Generates the publication-quality architectural schematic for Figure 1 in the thesis paper:
"System Architecture Overview: Single-Encoder 3D VisReg JEPA and Volumetric Segmentation Decoders."

Renders directly to paper/latex/figures/method_architecture_schematic.png at 300 DPI.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PNG = PROJECT_ROOT / "paper" / "latex" / "figures" / "method_architecture_schematic.png"


def create_schematic():
    # High-resolution figure: 18.5 x 10.5 inches at 300 DPI -> ~5550 x 3150 px
    fig, ax = plt.subplots(figsize=(18.5, 10.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Typography & Colors
    font_family = "sans-serif"
    title_fs = 11.5
    header_fs = 10.0
    body_fs = 8.5
    detail_fs = 7.5

    # Palette
    c_blue_bg = "#EBF3FB"
    c_blue_border = "#2A6496"
    c_blue_title = "#1A446C"

    c_cyan_bg = "#E0F2F7"
    c_cyan_border = "#00838F"
    c_cyan_title = "#004D40"

    c_teal_bg = "#E0F7FA"
    c_teal_border = "#00ACC1"
    c_teal_title = "#006064"

    c_amber_bg = "#FFF8E1"
    c_amber_border = "#FFA000"
    c_amber_title = "#E65100"

    c_rose_bg = "#FCE4EC"
    c_rose_border = "#D81B60"
    c_rose_title = "#880E4F"

    c_purple_bg = "#EDE7F6"
    c_purple_border = "#5E35B1"
    c_purple_title = "#311B92"

    c_gray_bg = "#ECEFF1"
    c_gray_border = "#546E7A"
    c_gray_title = "#263238"

    def draw_box(x, y, w, h, bg, border, title, lines, title_color="black"):
        # Draw fancy rounded rectangle
        box = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            facecolor=bg,
            edgecolor=border,
            linewidth=1.8,
            zorder=2,
        )
        ax.add_patch(box)

        # Title
        ax.text(
            x + w / 2,
            y + h - 1.8,
            title,
            ha="center",
            va="center",
            fontsize=header_fs,
            fontweight="bold",
            color=title_color,
            family=font_family,
            zorder=3,
        )

        # Content lines
        curr_y = y + h - 4.0
        line_spacing = (h - 5.0) / max(len(lines), 1)
        for line, is_bold, sz, col in lines:
            fw = "bold" if is_bold else "normal"
            ax.text(
                x + w / 2,
                curr_y,
                line,
                ha="center",
                va="center",
                fontsize=sz,
                fontweight=fw,
                color=col,
                family=font_family,
                zorder=3,
            )
            curr_y -= line_spacing

    def draw_arrow(x1, y1, x2, y2, color="#455A64", lw=2.0, label=None, label_pos=(0.5, 0.5), label_col="#37474F"):
        arrow = patches.FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            color=color,
            linewidth=lw,
            zorder=4,
        )
        ax.add_patch(arrow)
        if label:
            lx = x1 + (x2 - x1) * label_pos[0]
            ly = y1 + (y2 - y1) * label_pos[1]
            ax.text(
                lx,
                ly,
                label,
                ha="center",
                va="center",
                fontsize=detail_fs,
                fontweight="bold",
                color=label_col,
                family=font_family,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.9, lw=0.8),
                zorder=5,
            )

    # -------------------------------------------------------------
    # SECTION TITLES (Headers along top and middle)
    # -------------------------------------------------------------
    ax.text(1.5, 96.5, "(A) 3D Multi-Modal Volumetric Input", fontsize=title_fs, fontweight="bold", color="#1A237E")
    ax.text(23.5, 96.5, "(B) 3D Tokenization & Anisotropic Jitter Masking", fontsize=title_fs, fontweight="bold", color="#004D40")
    ax.text(49.0, 96.5, "(C) ViT Context & Predictor", fontsize=title_fs, fontweight="bold", color="#E65100")
    ax.text(73.5, 96.5, "(D) Auxiliary Projector & VisReg Regularization", fontsize=title_fs, fontweight="bold", color="#880E4F")

    ax.text(1.5, 39.0, "(E) Downstream Volumetric Segmentation Decoders (UNETR vs. FPN)", fontsize=title_fs, fontweight="bold", color="#311B92")
    ax.text(62.0, 39.0, "(F) Supervised Volumetric Baselines", fontsize=title_fs, fontweight="bold", color="#263238")

    # -------------------------------------------------------------
    # PANEL A: Input & Preprocessing
    # -------------------------------------------------------------
    # A1: Raw MRI
    draw_box(
        x=2.0, y=77.0, w=18.5, h=17.5,
        bg=c_blue_bg, border=c_blue_border,
        title="Raw BraTS 3D MRI",
        lines=[
            ("4 Co-Registered Channels:", True, body_fs, "#1A237E"),
            ("T1, T1c (contrast), T2, FLAIR", False, body_fs, "#37474F"),
            ("Native Voxel Resolution:", True, detail_fs, "#455A64"),
            ("1.0 mm³ Isotropic (182×218×182)", False, detail_fs, "#546E7A"),
            ("Intracranial Tissue Mask (Ω_fg)", False, detail_fs, "#546E7A"),
        ],
        title_color=c_blue_title,
    )

    # A2: Canonical Preprocessing
    draw_box(
        x=2.0, y=57.0, w=18.5, h=16.0,
        bg=c_blue_bg, border=c_blue_border,
        title="Canonical Preprocessing",
        lines=[
            ("Intracranial Bounding Box Crop", True, body_fs, "#1A237E"),
            ("Discards zero background air", False, detail_fs, "#455A64"),
            ("Trilinear Resampling to Grid:", True, body_fs, "#1A237E"),
            ("X ∈ ℝ^{4 × 128 × 128 × 128}", True, body_fs, "#0D47A1"),
            ("Fixed isotropic aspect ratio", False, detail_fs, "#546E7A"),
        ],
        title_color=c_blue_title,
    )

    # A3: Normalization & RMD
    draw_box(
        x=2.0, y=41.5, w=18.5, h=12.5,
        bg=c_blue_bg, border=c_blue_border,
        title="Parenchyma Norm & RMD",
        lines=[
            ("Z-Score on Non-Zero Brain Voxels", False, detail_fs, "#37474F"),
            ("Random Modality Dropout (p=0.25)", True, detail_fs, "#C2185B"),
            ("Zero-imputed fallback; 1ch guaranteed", False, detail_fs, "#546E7A"),
        ],
        title_color=c_blue_title,
    )

    draw_arrow(11.25, 77.0, 11.25, 73.0, color="#1E88E5", lw=2.2)
    draw_arrow(11.25, 57.0, 11.25, 54.0, color="#1E88E5", lw=2.2)

    # -------------------------------------------------------------
    # PANEL B: Tokenization & Masking
    # -------------------------------------------------------------
    # B1: PatchEmbed3D
    draw_box(
        x=23.5, y=77.0, w=22.5, h=17.5,
        bg=c_cyan_bg, border=c_cyan_border,
        title="3D Patch Embedding",
        lines=[
            ("Conv3D (k=8, s=8, C=384)", True, body_fs, "#004D40"),
            ("4,096 Total Patches (16×16×16 Grid)", True, body_fs, "#00695C"),
            ("Embedding Dimension D = 384", False, detail_fs, "#37474F"),
            ("+ Separable 3D Sinusoidal Pos Embed", True, detail_fs, "#00796B"),
            ("[PE(z) || PE(y) || PE(x)] (Frozen)", False, detail_fs, "#546E7A"),
        ],
        title_color=c_cyan_title,
    )

    # B2: Jitter Target Masking & 3D BFS Context
    draw_box(
        x=23.5, y=44.0, w=22.5, h=29.0,
        bg=c_cyan_bg, border=c_cyan_border,
        title="Dual 3D Jitter Masking",
        lines=[
            ("6 Anisotropic Target Cuboids (S):", True, body_fs, "#004D40"),
            ("[(6,6,6), (4,6,6), (6,4,6),", False, detail_fs, "#37474F"),
            ("(6,6,4), (4,4,4), (4,4,4)]", False, detail_fs, "#37474F"),
            ("N_tgt = 776 tokens (18.9% volume)", True, body_fs, "#00796B"),
            ("Brain-Aware Tissue Acceptance:", True, detail_fs, "#D81B60"),
            ("tissue_count ≥ max(2, ⌊0.25 × vol⌋)", False, detail_fs, "#C2185B"),
            ("-----------------------------------", False, detail_fs, "#90A4AE"),
            ("3D 26-Connected BFS Context:", True, body_fs, "#004D40"),
            ("N_ctx = 1,536 patches (37.5% volume)", True, body_fs, "#00796B"),
            ("Contiguous tissue-anchored cluster", False, detail_fs, "#546E7A"),
            ("85.9% Attention FLOP Reduction", True, detail_fs, "#E65100"),
        ],
        title_color=c_cyan_title,
    )

    # Arrow from A to B
    draw_arrow(20.5, 63.5, 23.5, 63.5, color="#00897B", lw=2.2, label="Volume X", label_pos=(0.4, 0.5))
    draw_arrow(34.75, 77.0, 34.75, 73.0, color="#00897B", lw=2.2)

    # -------------------------------------------------------------
    # PANEL C: Context Encoder & Predictor
    # -------------------------------------------------------------
    # C1: Online Context Encoder
    draw_box(
        x=49.0, y=72.0, w=21.5, h=22.5,
        bg=c_teal_bg, border=c_teal_border,
        title="3D ViT Context Encoder",
        lines=[
            ("8 Pre-LN Transformer Blocks", True, body_fs, "#006064"),
            ("D = 384, heads = 6, MLP = 4×", False, detail_fs, "#37474F"),
            ("Gathers ONLY Visible Context Tokens:", True, body_fs, "#00838F"),
            ("Z_ctx = Encoder(gather(Z_0, idx_ctx))", True, detail_fs, "#004D40"),
            ("Shape: ℝ^{B × 1536 × 384}", True, body_fs, "#00695C"),
            ("Zero Mask Leakage in Attention", False, detail_fs, "#546E7A"),
            ("Eliminates masked token computation", False, detail_fs, "#546E7A"),
        ],
        title_color=c_teal_title,
    )

    # C2: Predictor
    draw_box(
        x=49.0, y=44.0, w=21.5, h=24.0,
        bg=c_amber_bg, border=c_amber_border,
        title="3D JEPA Predictor",
        lines=[
            ("Narrow 4-Block Transformer", True, body_fs, "#E65100"),
            ("D_pred = 192, heads = 6", False, detail_fs, "#37474F"),
            ("Inputs: Projected Context + Mask Queries", True, detail_fs, "#EF6C00"),
            ("[Linear(Z_ctx) + P_ctx ; e_mask + P_tgt]", False, detail_fs, "#37474F"),
            ("Predicts Target Representations:", True, body_fs, "#E65100"),
            ("Ŷ_tgt = Linear_{192→384}(P_φ(...))", True, detail_fs, "#BF360C"),
            ("Huber Smooth-L1 Prediction Loss:", True, detail_fs, "#D84315"),
            ("L_pred(θ, φ) vs LayerNorm(Y_tgt)", True, detail_fs, "#C2185B"),
        ],
        title_color=c_amber_title,
    )

    # Arrows between B, C1, C2
    draw_arrow(46.0, 83.25, 49.0, 83.25, color="#0097A7", lw=2.2, label="Visible Tokens", label_pos=(0.45, 0.5))
    draw_arrow(46.0, 56.0, 49.0, 56.0, color="#FB8C00", lw=2.0, label="Target Queries", label_pos=(0.45, 0.5))
    draw_arrow(59.75, 72.0, 59.75, 68.0, color="#FB8C00", lw=2.2, label="Z_ctx", label_pos=(0.5, 0.5))

    # -------------------------------------------------------------
    # PANEL D: Auxiliary Projector & VisReg Regularization
    # -------------------------------------------------------------
    # D1: Auxiliary Projector MLP
    draw_box(
        x=73.5, y=77.0, w=24.5, h=17.5,
        bg=c_rose_bg, border=c_rose_border,
        title="Auxiliary Projector MLP",
        lines=[
            ("2-Layer Projection Head (g_ψ):", True, body_fs, "#880E4F"),
            ("Linear(384→1024) → GELU → Linear(1024→128)", True, detail_fs, "#AD1457"),
            ("Evaluates Tissue-Filtered Context:", True, detail_fs, "#37474F"),
            ("Z_tissue = Z_ctx[tissue_mask] ∈ ℝ^{N_t × 128}", False, detail_fs, "#546E7A"),
            ("Decouples representation from downstream", False, detail_fs, "#880E4F"),
            ("Projector is discarded during fine-tuning", True, detail_fs, "#C2185B"),
        ],
        title_color=c_rose_title,
    )

    # D2: VisReg Decoupled Loss
    draw_box(
        x=73.5, y=44.0, w=24.5, h=29.0,
        bg=c_rose_bg, border=c_rose_border,
        title="Decoupled VisReg Loss (L_VisReg)",
        lines=[
            ("1. Center Variance Anchoring:", True, body_fs, "#880E4F"),
            ("L_c = (1/D) ||μ||₂²  (Zero-mean centering)", False, detail_fs, "#37474F"),
            ("2. Coordinate Scale Regularization:", True, body_fs, "#880E4F"),
            ("L_sc = (1/D) Σ (1 - σ_d)²  (Unit variance)", False, detail_fs, "#37474F"),
            ("3. Sliced-Wasserstein Shape Matching:", True, body_fs, "#880E4F"),
            ("L_sh = W₂² over M=1024 1D random rays", True, detail_fs, "#AD1457"),
            ("Normalized with detached μ, σ (stop-grad)", False, detail_fs, "#37474F"),
            ("-----------------------------------", False, detail_fs, "#B0BEC5"),
            ("Teacher-Free Single-Encoder Formulation", True, body_fs, "#2E7D32"),
            ("Zero EMA momentum memory overhead", False, detail_fs, "#388E3C"),
            ("Mathematically prevents representation collapse", True, detail_fs, "#1B5E20"),
        ],
        title_color=c_rose_title,
    )

    draw_arrow(70.5, 85.75, 73.5, 85.75, color="#D81B60", lw=2.2, label="Z_tissue", label_pos=(0.45, 0.5))
    draw_arrow(85.75, 77.0, 85.75, 73.0, color="#D81B60", lw=2.2, label="Latents (128D)", label_pos=(0.5, 0.5))

    # -------------------------------------------------------------
    # PANEL E: Downstream Decoders (UNETR vs. FPN)
    # -------------------------------------------------------------
    # E1: UNETR Hybrid Decoder
    draw_box(
        x=2.0, y=7.0, w=27.5, h=30.0,
        bg=c_purple_bg, border=c_purple_border,
        title="1. 3D UNETR Decoder (Hybrid)",
        lines=[
            ("High-Resolution Convolutional Stem on Raw MRI:", True, body_fs, "#311B92"),
            ("stem_128 (128³, 16ch) & stem_64 (64³, 32ch)", True, detail_fs, "#4527A0"),
            ("Direct raw-image gradient bypass", False, detail_fs, "#546E7A"),
            ("-----------------------------------", False, detail_fs, "#B0BEC5"),
            ("Lateral ViT Feature Ingestion:", True, body_fs, "#311B92"),
            ("Intermediate ViT tokens from blocks L₂, L₄, L₆, L₈", False, detail_fs, "#37474F"),
            ("Reshaped to volumetric 3D feature grids", False, detail_fs, "#546E7A"),
            ("Fused via 3D Deconv + Residual Conv blocks", True, detail_fs, "#5E35B1"),
            ("Combines global ViT context with local CNN edge priors", False, detail_fs, "#4527A0"),
            ("Params: 29.52M (Encoder: 16.4M + Decoder: 13.1M)", False, detail_fs, "#7B1FA2"),
        ],
        title_color=c_purple_title,
    )

    # E2: FPN Pure Latent Decoder
    draw_box(
        x=32.0, y=7.0, w=27.5, h=30.0,
        bg=c_purple_bg, border=c_purple_border,
        title="2. 3D FPN Decoder (Pure Latent)",
        lines=[
            ("Pure Latent Multi-Scale Feature Pyramid:", True, body_fs, "#311B92"),
            ("Decodes STRICTLY from intermediate ViT tokens", True, body_fs, "#4527A0"),
            ("ViT blocks L₂ (16³), L₄ (16³), L₆ (16³), L₈ (16³)", False, detail_fs, "#37474F"),
            ("Zero Raw MRI Skip Connections (No image bypass)", True, detail_fs, "#C2185B"),
            ("-----------------------------------", False, detail_fs, "#B0BEC5"),
            ("Top-Down Lateral Connections & Upsampling:", True, body_fs, "#311B92"),
            ("Lateral 1×1×1 Convs + 3D Transposed Convolutions", False, detail_fs, "#37474F"),
            ("Progressive GroupNorm + GELU up to 128³", False, detail_fs, "#5E35B1"),
            ("Direct test of latent space spatial decodability", True, detail_fs, "#2E7D32"),
            ("Params: 24.8M (Encoder: 16.4M + Decoder: 8.4M)", False, detail_fs, "#7B1FA2"),
        ],
        title_color=c_purple_title,
    )

    # Output mapping box connecting E1 & E2
    draw_box(
        x=12.0, y=0.5, w=37.0, h=5.5,
        bg="#F3E5F5", border="#7B1FA2",
        title="Dense 5-Class Output → Clinical BraTS Multi-Region Unions",
        lines=[
            ("Softmax Logits [5, 128³] → Whole Tumor (WT: {1,2,3}), Tumor Core (TC: {1,3}), Enhancing Tumor (ET: {3})", True, detail_fs, "#4A148C"),
        ],
        title_color="#4A148C",
    )

    draw_arrow(15.75, 7.0, 20.0, 6.0, color="#7B1FA2", lw=2.0)
    draw_arrow(45.75, 7.0, 41.5, 6.0, color="#7B1FA2", lw=2.0)

    # Arrow from pre-trained encoder to downstream decoders
    draw_arrow(59.75, 44.0, 31.0, 37.0, color="#5E35B1", lw=2.2, label="Pre-trained ViT Weights", label_pos=(0.4, 0.5))

    # -------------------------------------------------------------
    # PANEL F: Supervised Baselines
    # -------------------------------------------------------------
    draw_box(
        x=62.0, y=7.0, w=36.0, h=30.0,
        bg=c_gray_bg, border=c_gray_border,
        title="Supervised 3D Convolutional Baselines",
        lines=[
            ("1. 3D nnU-Net (DynUNet with Deep Supervision):", True, body_fs, "#263238"),
            ("Self-configuring dynamic U-Net (22.75M parameters)", False, detail_fs, "#37474F"),
            ("3-stage multi-scale deep supervision loss:", True, detail_fs, "#00695C"),
            ("L_total = Σ (2^{-s} / Σ 2^{-j}) L_stage(s)", False, detail_fs, "#004D40"),
            ("Establishes the supervised full-data ceiling (70.86% Dice)", True, detail_fs, "#2E7D32"),
            ("--------------------------------------------------", False, detail_fs, "#B0BEC5"),
            ("2. 3D Residual UNet (Standard ResUNet):", True, body_fs, "#263238"),
            ("19.23M parameters with residual blocks & InstanceNorm", False, detail_fs, "#37474F"),
            ("Identical 60-epoch supervised schedule & augmentation", False, detail_fs, "#546E7A"),
            ("Fastest inference throughput (488.82 ms per 128³ volume)", True, detail_fs, "#E65100"),
        ],
        title_color=c_gray_title,
    )

    plt.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"✓ Saved updated architectural schematic to: {OUTPUT_PNG}")


if __name__ == "__main__":
    create_schematic()
