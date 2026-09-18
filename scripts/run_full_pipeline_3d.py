#!/usr/bin/env python3
"""
3D Master Automation Pipeline Orchestrator.
Sequentially runs data prep, SSL pretraining, downstream fine-tuning, baselines, and evaluation.
"""

import argparse
import subprocess
import sys

from brats_jepa_3d.config import CHECKPOINTS_DIR, PROJECT_ROOT
from brats_jepa_3d.utils import sort_checkpoints_by_epoch


def run_cmd(cmd_list: list[str]):
    print(f"\n[RUNNING]: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT), check=False)
    if res.returncode != 0:
        print(f"[ERROR]: Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run Full 3D JEPA Pipeline")
    parser.add_argument(
        "--model_type",
        type=str,
        default="visreg_jepa",
        choices=["visreg_jepa", "sigreg_jepa", "ijepa", "all"],
        help="SSL JEPA model architecture to train and evaluate (default: visreg_jepa)",
    )
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

    # Models to train and fine-tune
    models_to_run = (
        ["visreg_jepa", "sigreg_jepa", "ijepa"]
        if args.model_type == "all"
        else [args.model_type]
    )

    # 2. SSL Pre-training & 3. Downstream Fine-tuning
    for m_type in models_to_run:
        run_cmd([py, "scripts/train_jepa_3d.py", "--model_type", m_type] + smoke_flag)

        best_ckpts = sorted(CHECKPOINTS_DIR.glob(f"{m_type}*_3d_best.pt"))
        epoch_ckpts = sort_checkpoints_by_epoch(
            list(CHECKPOINTS_DIR.glob(f"{m_type}*epoch*.pt"))
        )
        selected_ckpt = best_ckpts[-1] if best_ckpts else (epoch_ckpts[-1] if epoch_ckpts else None)
        pretrained_arg = ["--pretrained_checkpoint", str(selected_ckpt)] if selected_ckpt else []
        run_cmd(
            [
                py,
                "scripts/train_downstream_3d.py",
                "--model_type",
                m_type,
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
    eval_model_arg = ["--all_models"] if args.model_type == "all" else ["--model_type", args.model_type]
    run_cmd([py, "scripts/evaluate_3d.py"] + eval_model_arg + smoke_flag)

    # 6. Figures
    run_cmd([py, "scripts/generate_figures_3d.py"])

    print("\n=======================================================")
    print("3D Multimodal JEPA Research Pipeline Completed Successfully!")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
