#!/usr/bin/env python3
"""
End-to-End Pipeline Smoke Test Runner.
--------------------------------------
Executes a rapid smoke-test pass across the complete 3D JEPA segmentation pipeline:
1. Data preparation (subset sample limit)
2. Self-supervised pretraining (VisReg / SigReg JEPA)
3. Downstream fine-tuning across all 3 key architectures:
   - VisReg JEPA (Multi-Scale FPN)
   - VisReg JEPA (UNETR Hybrid)
   - ViT-UNETR Hybrid (From Scratch control)
4. Supervised CNN baselines (3D Residual UNet and 3D nnU-Net DynUNet)
5. Multi-region master evaluation battery
6. Artifact & figure consolidation

By default, all smoke test outputs (checkpoints, logs, metrics) are isolated
in a temporary directory and automatically deleted upon completion to ensure
zero pollution of production results. Pass --keep_artifacts to preserve them.

Usage:
    python tests/smoke_test_pipeline.py
    python tests/smoke_test_pipeline.py --keep_artifacts
    python tests/smoke_test_pipeline.py --model_type all
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from brats_jepa_3d.config import PROJECT_ROOT
from brats_jepa_3d.utils import sort_checkpoints_by_epoch


def run_cmd(cmd_list: list[str]) -> None:
    print(f"\n[SMOKE RUNNER]: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT), check=False)
    if res.returncode != 0:
        print(f"[ERROR]: Step failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End 3D Pipeline Smoke Test")
    parser.add_argument(
        "--model_type",
        type=str,
        default="visreg_jepa",
        choices=["visreg_jepa", "sigreg_jepa", "all"],
        help="SSL JEPA model architecture to smoke test (default: visreg_jepa)",
    )
    parser.add_argument(
        "--skip_data_prep",
        action="store_true",
        default=True,
        help="Skip data preparation step if dataset is already prepared (default: True)",
    )
    parser.add_argument(
        "--run_data_prep",
        action="store_false",
        dest="skip_data_prep",
        help="Force running data preparation on 10 volumes",
    )
    parser.add_argument(
        "--keep_artifacts",
        action="store_true",
        default=False,
        help="Preserve generated smoke test checkpoints and metrics instead of auto-cleaning (default: False)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory for smoke test runs (default: isolated temporary directory)",
    )
    parser.add_argument(
        "--full_run",
        action="store_true",
        default=False,
        help="Run full training schedule instead of smoke test (Warning: very slow)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    py = sys.executable
    smoke_flags = [] if args.full_run else ["--smoke_test"]

    # Isolate all smoke test outputs into a temporary directory
    smoke_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else PROJECT_ROOT / "outputs" / ".smoke_tmp"
    )
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_ckpts = smoke_dir / "checkpoints"
    smoke_ckpts.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("🚀 Launching 3D Pipeline Smoke Test Suite")
    print(f"Mode: {'FULL RUN' if args.full_run else 'SMOKE TEST (Fast Verification)'}")
    print(f"Architecture Arm: {args.model_type}")
    print(f"Output Staging Dir: {smoke_dir}")
    print(f"Auto-Clean Artifacts: {'DISABLED (--keep_artifacts)' if args.keep_artifacts else 'ENABLED (Zero Leftover Pollutants)'}")
    print("=================================================================\n")

    try:
        # 1. Data Preparation (if requested)
        if not args.skip_data_prep:
            limit_args = ["--limit", "10"] if not args.full_run else []
            prep_dir = smoke_dir / "data_prep_smoke"
            prep_dir.mkdir(parents=True, exist_ok=True)
            print("--- [Stage 1/6] Data Preparation ---")
            run_cmd([py, "scripts/prepare_data_3d.py", "--output_dir", str(prep_dir)] + limit_args)

        models_to_run = (
            ["visreg_jepa", "sigreg_jepa"]
            if args.model_type == "all"
            else [args.model_type]
        )

        # 2. SSL Pre-training & Downstream Fine-Tuning
        for m_type in models_to_run:
            print(f"\n--- [Stage 2/6] SSL Pre-Training ({m_type}) ---")
            run_cmd(
                [
                    py,
                    "scripts/train_jepa_3d.py",
                    "--model_type",
                    m_type,
                    "--output_dir",
                    str(smoke_dir),
                ]
                + smoke_flags
            )

            best_ckpts = sorted(smoke_ckpts.glob(f"{m_type}*_3d_best.pt"))
            epoch_ckpts = sort_checkpoints_by_epoch(
                list(smoke_ckpts.glob(f"{m_type}*epoch*.pt"))
            )
            selected_ckpt = best_ckpts[-1] if best_ckpts else (epoch_ckpts[-1] if epoch_ckpts else None)
            pretrained_arg = ["--pretrained_checkpoint", str(selected_ckpt)] if selected_ckpt else []

            print(f"\n--- [Stage 3/6] Downstream Fine-Tuning: {m_type} (Multiscale FPN) ---")
            run_cmd(
                [
                    py,
                    "scripts/train_downstream_3d.py",
                    "--model_type",
                    m_type,
                    "--decoder_type",
                    "multiscale",
                    "--output_dir",
                    str(smoke_dir),
                ]
                + pretrained_arg
                + smoke_flags
            )

            print(f"\n--- [Stage 3/6] Downstream Fine-Tuning: {m_type} (UNETR Hybrid) ---")
            run_cmd(
                [
                    py,
                    "scripts/train_downstream_3d.py",
                    "--model_type",
                    m_type,
                    "--decoder_type",
                    "unetr_hybrid",
                    "--output_dir",
                    str(smoke_dir),
                ]
                + pretrained_arg
                + smoke_flags
            )

        # 3. From-Scratch ViT Control
        print("\n--- [Stage 3/6] Downstream Fine-Tuning: ViT-UNETR Hybrid (From Scratch Control) ---")
        run_cmd(
            [
                py,
                "scripts/train_downstream_3d.py",
                "--model_type",
                "visreg_jepa",
                "--decoder_type",
                "unetr_hybrid",
                "--from_scratch",
                "--output_dir",
                str(smoke_dir),
            ]
            + smoke_flags
        )

        # 4. Supervised CNN Baselines
        print("\n--- [Stage 4/6] Supervised CNN Baselines ---")
        run_cmd([py, "scripts/train_unet_3d.py", "--output_dir", str(smoke_dir)] + smoke_flags)
        run_cmd([py, "scripts/train_nnunet_3d.py", "--output_dir", str(smoke_dir)] + smoke_flags)

        # 5. Master Benchmark Evaluation
        print("\n--- [Stage 5/6] Master Benchmark Evaluation ---")
        eval_model_arg = ["--all_models"] if args.model_type == "all" else ["--model_type", args.model_type]
        run_cmd(
            [
                py,
                "scripts/evaluate_3d.py",
                "--output_dir",
                str(smoke_dir),
            ]
            + eval_model_arg
            + smoke_flags
        )

        # 6. Artifact & Publication Figure Generation (on full runs)
        if args.full_run:
            print("\n--- [Stage 6/6] Consolidating Artifacts & Figures ---")
            run_cmd(
                [
                    py,
                    "scripts/combine_and_generate_paper_artifacts.py",
                    "--outputs_dir",
                    str(smoke_dir),
                    "--no_zip",
                ]
            )

        print("\n=================================================================")
        print("✅ All 3D Pipeline Smoke Test Stages Completed Successfully!")
        print("=================================================================\n")

    finally:
        if not args.keep_artifacts:
            print(f"🧹 Auto-cleaning temporary smoke test artifacts from: {smoke_dir} ...")
            shutil.rmtree(smoke_dir, ignore_errors=True)
            print("✓ Cleanup complete. Zero leftover artifacts on disk.\n")
        else:
            print(f"📁 Smoke test artifacts preserved at: {smoke_dir}\n")


if __name__ == "__main__":
    main()
