#!/usr/bin/env python3
"""
3D Master Automation Pipeline Orchestrator.
Sequentially runs data prep, SSL pretraining, downstream fine-tuning, baselines, and evaluation.
"""

import argparse
import subprocess
import sys

from brats_jepa_3d.config import PROJECT_ROOT


def run_cmd(cmd_list: list[str]):
    print(f"\n[RUNNING]: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        print(f"[ERROR]: Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run Full 3D JEPA Pipeline")
    parser.add_argument(
        "--smoke_test", action="store_true", help="Run 1-epoch smoke test on all stages"
    )
    parser.add_argument("--skip_data_prep", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    # 1. Data Preparation
    if not args.skip_data_prep:
        limit = ["--limit", "20"] if args.smoke_test else []
        run_cmd([py, "scripts/prepare_data_3d.py"] + limit)

    smoke_flag = ["--smoke_test"] if args.smoke_test else []

    # 2. SSL Pre-training (SigReg JEPA)
    run_cmd([py, "scripts/train_jepa_3d.py", "--model_type", "sigreg_jepa"] + smoke_flag)

    # 3. Downstream Fine-tuning
    sigreg_ckpts = sorted((PROJECT_ROOT / "outputs/checkpoints").glob("sigreg_jepa_3d_epoch_*.pt"))
    pretrained_arg = ["--pretrained_checkpoint", str(sigreg_ckpts[-1])] if sigreg_ckpts else []
    run_cmd(
        [
            py,
            "scripts/train_downstream_3d.py",
            "--model_type",
            "sigreg_jepa",
            "--decoder_type",
            "multiscale",
        ]
        + pretrained_arg
        + smoke_flag
    )

    # 4. Supervised Baselines
    run_cmd([py, "scripts/train_unet_3d.py"] + smoke_flag)
    run_cmd([py, "scripts/train_nnunet_3d.py"] + smoke_flag)

    # 5. Master Benchmark Evaluation
    run_cmd([py, "scripts/evaluate_3d.py"] + smoke_flag)

    # 6. Figures
    run_cmd([py, "scripts/generate_figures_3d.py"])

    print("\n=======================================================")
    print("3D Multimodal JEPA Research Pipeline Completed Successfully!")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
