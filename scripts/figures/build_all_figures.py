#!/usr/bin/env python3
"""
scripts/figures/build_all_figures.py
------------------------------------
Master build script that generates, compiles, and verifies publication-quality
3D architecture figures for all models in the repository:
1. Single-Encoder 3D VISReg JEPA (Pre-Training)
2. 3D Residual UNet Baseline (BraTS3DUNet)
3. 3D nnU-Net Baseline (BraTS3DnnUNet / DynUNet)
4. ViT + Multi-Scale FPN Decoder (Pure Latent Probe)
5. ViT + Hybrid UNETR Decoder (Dual-Stream ViT + Conv Stem)

Outputs vector PDFs and 300-DPI PNGs to both:
- paper/latex/figures/
- docs/figures/
"""

import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

FIGURE_SCRIPTS = [
    "plot_jepa_pretrain_3d.py",
    "plot_unet_3d.py",
    "plot_nnunet_3d.py",
    "plot_downstream_fpn_3d.py",
    "plot_downstream_unetr_3d.py",
]

OUTPUT_DIRS = [
    PROJECT_ROOT / "paper" / "latex" / "figures",
    PROJECT_ROOT / "docs" / "figures",
]

TARGET_FIGURES = [
    "arch_jepa_pretrain_3d",
    "arch_unet_3d",
    "arch_nnunet_3d",
    "arch_fpn_decoder_3d",
    "arch_unetr_decoder_3d",
]


def main():
    print("=" * 80)
    print("BUILDING ALL 3D MODEL ARCHITECTURE FIGURES (PlotNeuralNet / TikZ)")
    print("=" * 80)

    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for script_name in FIGURE_SCRIPTS:
        script_path = SCRIPT_DIR / script_name
        print(f"\n--- Running {script_name} ---")
        res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"❌ Failed to run {script_name}:")
            print(res.stderr)
            print(res.stdout[-1500:])
            failed.append(script_name)
        else:
            print(f"✓ Successfully executed {script_name}")

    if failed:
        print("\n❌ Build finished with errors on:", failed)
        sys.exit(1)

    print("\n" + "=" * 80)
    print("VERIFYING GENERATED ASSETS")
    print("=" * 80)
    all_present = True
    for fig_base in TARGET_FIGURES:
        for out_dir in OUTPUT_DIRS:
            pdf_path = out_dir / f"{fig_base}.pdf"
            png_path = out_dir / f"{fig_base}.png"
            if pdf_path.exists() and png_path.exists():
                size_png = png_path.stat().st_size / 1024
                print(f"✓ {out_dir.relative_to(PROJECT_ROOT)}/{fig_base} (.pdf + .png, {size_png:.1f} KB)")
            else:
                print(f"❌ Missing in {out_dir.relative_to(PROJECT_ROOT)}: {fig_base}")
                all_present = False

    if not all_present:
        print("\n❌ Some target figure files were missing.")
        sys.exit(1)

    print("\n✓ ALL 5 MODEL ARCHITECTURE FIGURES SUCCESSFULLY BUILT AND VERIFIED!")


if __name__ == "__main__":
    main()
