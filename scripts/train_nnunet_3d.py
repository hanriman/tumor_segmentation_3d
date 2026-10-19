#!/usr/bin/env python3
"""
3D Supervised nnU-Net Baseline Runner (DynUNet with Deep Supervision).
"""

import argparse
from pathlib import Path

from brats_jepa_3d.config import (
    OUTPUTS_DIR,
    ensure_directories,
    resolve_default_output_dir,
)
from brats_jepa_3d.models import build_segmentation_model_for_training
from brats_jepa_3d.training import (
    add_common_supervised_args,
    build_train_val_loaders,
    evaluate_validation_split,
    load_supervised_configs,
    run_supervised_training,
)
from brats_jepa_3d.utils import get_device, set_seed, setup_logger

evaluate = evaluate_validation_split


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train 3D nnU-Net Baseline")
    add_common_supervised_args(parser)
    return parser.parse_args(args)


def main():
    args = parse_args()
    args = load_supervised_configs(args, default_model_yaml="nnunet_3d.yaml")

    out_dir = resolve_default_output_dir(
        args.output_dir,
        experiment_name="nnunet_outputs",
        is_smoke_test=getattr(args, "smoke_test", False),
    )
    checkpoints_dir = out_dir / "checkpoints"
    logs_dir = out_dir / "logs"
    ensure_directories(out_dir)
    set_seed(args.seed)
    device = get_device()
    logger = setup_logger("train_nnunet_3d", logs_dir / "nnunet_3d_train.log")
    logger.info(
        f"Starting 3D nnU-Net Training with Deep Supervision: Device={device}, AMP={args.amp}"
    )

    train_loader, val_loader, _ = build_train_val_loaders(args, device, logger)

    built = build_segmentation_model_for_training(
        model_type="nnunet_3d",
        device=device,
        deep_supervision=getattr(args, "deep_supervision", True),
        out_channels=getattr(args, "out_channels", 5),
        model_overrides={
            "in_channels": getattr(args, "in_channels", None),
            "deep_supr_num": getattr(args, "deep_supr_num", None),
            "res_block": getattr(args, "res_block", None),
        },
    )
    assert built is not None
    model, _ = built

    run_supervised_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        device=device,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        ckpt_prefix="nnunet_3d",
        metrics_prefix="nnunet_3d",
        logger=logger,
        require_resume_exists=True,
    )
    logger.info("3D nnU-Net training completed.")


if __name__ == "__main__":
    main()
