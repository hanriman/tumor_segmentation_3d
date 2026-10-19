#!/usr/bin/env python3
"""
3D Supervised Residual UNet Baseline Runner.
Trains MONAI 3D Residual UNet on multi-modal BraTS 2024 GLI volumes.
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
    parser = argparse.ArgumentParser(description="Train 3D Residual UNet Baseline")
    add_common_supervised_args(parser)
    return parser.parse_args(args)


def main():
    args = parse_args()
    args = load_supervised_configs(args, default_model_yaml="unet_3d.yaml")

    out_dir = resolve_default_output_dir(
        args.output_dir,
        experiment_name="unet_outputs",
        is_smoke_test=getattr(args, "smoke_test", False),
    )
    checkpoints_dir = out_dir / "checkpoints"
    logs_dir = out_dir / "logs"
    ensure_directories(out_dir)
    set_seed(args.seed)
    device = get_device()
    logger = setup_logger("train_unet_3d", logs_dir / "unet_3d_train.log")
    logger.info(f"Starting 3D Residual UNet Training: Device={device}, AMP={args.amp}")
    logger.info(f"Deep supervision: {getattr(args, 'deep_supervision', True)}")

    train_loader, val_loader, _ = build_train_val_loaders(args, device, logger)

    built = build_segmentation_model_for_training(
        model_type="unet_3d",
        device=device,
        deep_supervision=getattr(args, "deep_supervision", True),
        out_channels=getattr(args, "out_channels", 5),
        model_overrides={
            "in_channels": getattr(args, "in_channels", None),
            "channels": getattr(args, "channels", None),
            "strides": getattr(args, "strides", None),
            "num_res_units": getattr(args, "num_res_units", None),
            "dropout": getattr(args, "dropout", None),
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
        ckpt_prefix="unet_3d",
        metrics_prefix="unet_3d",
        logger=logger,
        require_resume_exists=True,
    )
    logger.info("3D Residual UNet training completed.")


if __name__ == "__main__":
    main()
