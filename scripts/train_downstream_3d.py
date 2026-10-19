#!/usr/bin/env python3
"""
3D Downstream Volumetric Fine-Tuning & Decoder Probing Runner.
Couples pre-trained 3D JEPA encoders with Multi-Scale FPN or UNETR Hybrid Decoders.
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
    parser = argparse.ArgumentParser(description="Downstream 3D Volumetric Fine-Tuning")
    parser.add_argument("--model_type", type=str, default="sigreg_jepa")
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="multiscale",
        choices=["multiscale", "unetr_hybrid"],
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained SSL encoder checkpoint",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Decoder probing: freeze encoder weights",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        default=False,
        help="Train 3D ViT downstream model from random initialization (no pre-trained checkpoint)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs="+",
        default=None,
        help="Voxel patch size (e.g. 8 8 8 for high-res A100 or 16 16 16 for standard, default: from config)",
    )
    add_common_supervised_args(parser)
    return parser.parse_args(args)


def main():
    args = parse_args()
    args = load_supervised_configs(args, default_model_yaml=f"{args.model_type}_3d.yaml")

    if args.from_scratch:
        exp_name = f"scratch_{args.decoder_type}_outputs"
    elif args.freeze_encoder:
        exp_name = f"frozen_{args.model_type}_{args.decoder_type}_outputs"
    else:
        exp_name = f"{args.model_type}_{args.decoder_type}_outputs"

    out_dir = resolve_default_output_dir(
        args.output_dir,
        experiment_name=exp_name,
        is_smoke_test=getattr(args, "smoke_test", False),
    )
    checkpoints_dir = out_dir / "checkpoints"
    logs_dir = out_dir / "logs"
    ensure_directories(out_dir)
    set_seed(args.seed)
    device = get_device()

    log_suffix = "_scratch" if args.from_scratch else ""
    logger = setup_logger(
        "downstream_3d", logs_dir / f"{args.model_type}_{args.decoder_type}{log_suffix}_downstream.log"
    )
    logger.info(
        f"Starting 3D Downstream Fine-Tuning: Model={args.model_type}, Decoder={args.decoder_type}, "
        f"FromScratch={args.from_scratch}, FreezeEncoder={args.freeze_encoder}"
    )

    train_loader, val_loader, resolved_data_dir = build_train_val_loaders(args, device, logger)

    built = build_segmentation_model_for_training(
        model_type=args.model_type,
        device=device,
        decoder_type=args.decoder_type,
        from_scratch=args.from_scratch,
        freeze_encoder=args.freeze_encoder,
        deep_supervision=args.deep_supervision,
        patch_size=getattr(args, "patch_size", None),
        out_channels=getattr(args, "out_channels", 5),
        checkpoints_dir=checkpoints_dir,
        pretrained_checkpoint=getattr(args, "pretrained_checkpoint", None),
        resolved_data_dir=resolved_data_dir,
        logger=logger,
        model_overrides={
            "spatial_shape": getattr(args, "spatial_shape", None),
            "in_channels": getattr(args, "in_channels", None),
            "embed_dim": getattr(args, "embed_dim", None),
            "encoder_depth": getattr(args, "encoder_depth", None),
            "num_heads": getattr(args, "num_heads", None),
            "mlp_ratio": getattr(args, "mlp_ratio", None),
        },
        label="3D ViT-FPN",
    )
    assert built is not None
    model, _ = built

    ckpt_suffix = "_scratch" if args.from_scratch else ("_frozen" if args.freeze_encoder else "")
    ckpt_prefix = f"{args.model_type}_{args.decoder_type}{ckpt_suffix}"
    metrics_prefix = f"{args.model_type}_{args.decoder_type}{log_suffix}_downstream"

    run_supervised_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        device=device,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        ckpt_prefix=ckpt_prefix,
        metrics_prefix=metrics_prefix,
        logger=logger,
    )
    logger.info(
        f"3D Downstream Fine-Tuning finished ({'from scratch' if args.from_scratch else 'pre-trained'})."
    )


if __name__ == "__main__":
    main()
