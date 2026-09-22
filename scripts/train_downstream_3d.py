#!/usr/bin/env python3
"""
3D Downstream Volumetric Fine-Tuning & Linear Probing Runner.
Couples pre-trained 3D JEPA encoders with Bottleneck or Multi-Scale FPN Decoders.
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from brats_jepa_3d.config import (
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    LOGS_DIR,
    ensure_directories,
    get_dataset_dir,
    load_yaml_config,
    merge_config_with_args,
)
from brats_jepa_3d.data import BraTS3DDataset, VolumetricAugmentations3D
from brats_jepa_3d.losses import build_segmentation_criterion, resolve_seg_loss_type
from brats_jepa_3d.metrics import compute_volumetric_metrics_3d, validation_dice_iou
from brats_jepa_3d.models import JEPASegmentationModel3D
from brats_jepa_3d.utils import (
    MetricTracker,
    check_pool_match,
    dataset_fingerprint,
    get_autocast_context,
    get_device,
    set_seed,
    setup_logger,
    sort_checkpoints_by_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Downstream 3D Volumetric Fine-Tuning")
    parser.add_argument("--model_type", type=str, default="sigreg_jepa")
    parser.add_argument(
        "--decoder_type", type=str, default="multiscale", choices=["multiscale", "bottleneck", "unetr_hybrid"]
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
        help="Linear/decoder probing: freeze encoder weights",
    )
    parser.add_argument("--deep_supervision", action="store_true", default=True)
    parser.add_argument("--no_deep_supervision", "--no-deep-supervision", action="store_false", dest="deep_supervision")
    parser.add_argument(
        "--loss_type", type=str, default="dice_bce", choices=["dice_bce", "tversky"],
        help="Overlap loss: symmetric Dice+BCE (default) or asymmetric Tversky(beta=0.7)+BCE",
    )
    parser.add_argument("--tversky_alpha", type=float, default=0.3)
    parser.add_argument("--tversky_beta", type=float, default=0.7)
    parser.add_argument("--config", type=str, default=None, help="Path to base YAML config")
    parser.add_argument("--include_background", action="store_true", default=None,
        help="Include background in overlap loss (default: auto → excluded for C>1, included for C=1)")
    parser.add_argument("--no_include_background", action="store_false", dest="include_background",
        help="Exclude background from overlap loss")
    parser.add_argument("--model_config", type=str, default=None, help="Path to model YAML config")
    parser.add_argument(
        "--dataset_config", type=str, default=None, help="Path to dataset YAML config"
    )
    parser.add_argument(
        "--exp_config", type=str, default=None, help="Path to experiment YAML config"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", "--no-amp", action="store_false", dest="amp")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of parallel DataLoader worker processes (default: 2)",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Enable pinned memory for faster host-to-device transfers (default: True)",
    )
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        default=False,
        help="Train 3D ViT-FPN downstream model from random initialization (no pre-trained checkpoint)",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=0,
        help="Frequency (in epochs) to save periodic model checkpoints (default: 0, disabled)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument("--smoke_test", action="store_true", help="Run fast verification")
    return parser.parse_args()


def evaluate(model, loader, device, amp: bool = True, smoke_test: bool = False) -> dict[str, float]:
    model.eval()
    all_dices, all_ious = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validating", leave=False)):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with get_autocast_context(device, enabled=amp):
                logits = model(images)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]

            if logits.shape[1] == 1:
                metrics = compute_volumetric_metrics_3d(logits, masks, compute_hd95=False)
                all_dices.extend(metrics["dice_per_sample"])
                all_ious.extend(metrics["iou_per_sample"])
            else:
                # Multi-class protocol: mean over WT/TC/ET keeps model selection
                # honest (WT-only is blind to ET collapse).
                d, i = validation_dice_iou(logits, masks)
                all_dices.append(d)
                all_ious.append(i)

            if smoke_test and batch_idx >= 1:
                break

    return {
        "val_dice": float(np.mean(all_dices)) if all_dices else 0.0,
        "val_iou": float(np.mean(all_ious)) if all_ious else 0.0,
    }


def main():
    args = parse_args()

    # Load configs if available
    base_cfg_path = args.config or (CONFIGS_DIR / "base.yaml")
    if Path(base_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(base_cfg_path), args)

    model_cfg_path = args.model_config or (CONFIGS_DIR / "model" / f"{args.model_type}_3d.yaml")
    if Path(model_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(model_cfg_path), args)

    dataset_cfg_path = args.dataset_config or (CONFIGS_DIR / "dataset" / "brats3d.yaml")
    if Path(dataset_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(dataset_cfg_path), args)

    exp_cfg_path = args.exp_config or (CONFIGS_DIR / "experiment" / "finetune_30ep.yaml")
    if Path(exp_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(exp_cfg_path), args)

    ensure_directories()
    set_seed(args.seed)
    device = get_device()
    suffix = "_scratch" if args.from_scratch else ""
    logger = setup_logger(
        "downstream_3d", LOGS_DIR / f"{args.model_type}_{args.decoder_type}{suffix}_downstream.log"
    )
    logger.info(
        f"Starting 3D Downstream Fine-Tuning: Model={args.model_type}, Decoder={args.decoder_type}, "
        f"FromScratch={args.from_scratch}, FreezeEncoder={args.freeze_encoder}"
    )

    aug_tf = VolumetricAugmentations3D(
        flip_prob=getattr(args, "rand_flip_prob", getattr(args, "flip_prob", 0.5)),
        noise_prob=getattr(args, "rand_noise_prob", getattr(args, "noise_prob", 0.3)),
        noise_std=getattr(args, "noise_std", 0.05),
        modality_dropout_prob=getattr(args, "modality_dropout_prob", 0.25),
        is_training=True,
    )

    try:
        train_dataset = BraTS3DDataset(split="train", augmentations=aug_tf)
        val_dataset = BraTS3DDataset(split="val", augmentations=None)
    except FileNotFoundError:
        if not args.smoke_test:
            raise
        logger.warning(
            "Processed dataset not found. Generating synthetic volume dataset for smoke test verification."
        )
        train_dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": (torch.rand(1, 128, 128, 128) > 0.95).float(),
                "patient_id": f"train_{i}",
            }
            for i in range(4)
        ]
        val_dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": (torch.rand(1, 128, 128, 128) > 0.95).float(),
                "patient_id": f"val_{i}",
            }
            for i in range(2)
        ]

    num_workers = getattr(args, "num_workers", 2)
    use_pin_memory = getattr(args, "pin_memory", True) and (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size if not args.smoke_test else 2,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    spatial_shape = tuple(getattr(args, "spatial_shape", (128, 128, 128)))
    patch_size = tuple(getattr(args, "patch_size", (16, 16, 16)))
    in_channels = getattr(args, "in_channels", 4)
    embed_dim = getattr(args, "embed_dim", 384)
    encoder_depth = getattr(args, "encoder_depth", 8)
    num_heads = getattr(args, "num_heads", 6)
    mlp_ratio = getattr(args, "mlp_ratio", 4.0)

    # Initialize Downstream Model
    model = JEPASegmentationModel3D(
        img_size=spatial_shape,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        out_channels=getattr(args, "out_channels", 1),
        freeze_encoder=args.freeze_encoder,
        decoder_type=args.decoder_type,
        deep_supervision=args.deep_supervision,
    ).to(device)

    # Load pre-trained encoder checkpoint if specified or auto-discover
    ckpt_path = None
    if args.from_scratch:
        logger.info(
            "Initializing 3D ViT-FPN from SCRATCH (random initialization, no pre-trained checkpoint) [--from_scratch active]"
        )
    elif args.pretrained_checkpoint:
        ckpt_path = Path(args.pretrained_checkpoint)
    else:
        ssl_best = sorted(CHECKPOINTS_DIR.glob(f"{args.model_type}*_3d_best.pt"))
        ssl_epochs = sort_checkpoints_by_epoch(list(CHECKPOINTS_DIR.glob(f"{args.model_type}*_epoch_*.pt")))
        if ssl_best:
            ckpt_path = ssl_best[-1]
        elif ssl_epochs:
            ckpt_path = ssl_epochs[-1]
        else:
            candidates = [
                p
                for p in sorted(CHECKPOINTS_DIR.glob(f"{args.model_type}*.pt"))
                if not any(dec in p.name for dec in ("multiscale", "bottleneck", "unetr_hybrid", "hybrid", "downstream"))
            ]
            if candidates:
                ckpt_path = candidates[-1]

    if not args.from_scratch and ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        check_pool_match(
            ckpt.get("pool_fingerprint"),
            dataset_fingerprint(get_dataset_dir()),
            "downstream",
        )
        res = model.load_pretrained_encoder(ckpt)
        logger.info(
            f"Loaded pre-trained encoder weights from {ckpt_path} ({res['loaded_keys']} keys matched)"
        )
    elif not args.from_scratch:
        if args.freeze_encoder:
            raise FileNotFoundError(
                f"Cannot freeze encoder without a pre-trained checkpoint! "
                f"No pre-trained checkpoint found for '{args.model_type}' (specified: '{args.pretrained_checkpoint}')."
            )
        logger.warning(
            "No pre-trained checkpoint specified or found. Training encoder from random initialization."
        )

    criterion = build_segmentation_criterion(
        resolve_seg_loss_type(args),
        deep_supervision=args.deep_supervision,
        tversky_alpha=getattr(args, "tversky_alpha", 0.3),
        tversky_beta=getattr(args, "tversky_beta", 0.7),
        num_classes=getattr(args, "out_channels", 1),
        include_background=getattr(args, "include_background", None),
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    epochs = 1 if args.smoke_test else args.epochs

    warmup_epochs = max(1, min(5, epochs // 5)) if epochs > 1 else 0
    if epochs > 1 and warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler(
        device="cuda" if device.type == "cuda" else "cpu",
        enabled=args.amp and device.type == "cuda",
    )

    best_val_dice = -1.0
    start_epoch = 1
    tracker = MetricTracker()

    if args.resume and Path(args.resume).exists():
        resume_path = Path(args.resume)
        logger.info(f"Resuming training state from checkpoint: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device)
        if "model_state_dict" in resume_ckpt:
            model.load_state_dict(resume_ckpt["model_state_dict"])
        if "optimizer_state_dict" in resume_ckpt and resume_ckpt["optimizer_state_dict"]:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt and resume_ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in resume_ckpt and resume_ckpt["scaler_state_dict"] and scaler.is_enabled():
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        if "epoch" in resume_ckpt:
            start_epoch = resume_ckpt["epoch"] + 1
        if "best_val_dice" in resume_ckpt:
            best_val_dice = float(resume_ckpt["best_val_dice"])
        logger.info(f"Resumed successfully at epoch {start_epoch} with best_val_dice: {best_val_dice:.4f}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            with get_autocast_context(device, enabled=args.amp):
                logits = model(images)
                loss_dict = criterion(logits, masks)
                loss = loss_dict["loss"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if args.smoke_test and batch_idx >= 1:
                break

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, amp=args.amp, smoke_test=args.smoke_test)
        avg_train_loss = train_loss / max(1, num_batches)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch}/{epochs} - LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Dice: {val_metrics['val_dice']:.4f} | Val IoU: {val_metrics['val_iou']:.4f}"
        )

        tracker.update(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": avg_train_loss,
                **val_metrics,
            }
        )

        suffix = "_scratch" if args.from_scratch else ""
        state_dict_to_save = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "val_dice": val_metrics["val_dice"],
            "best_val_dice": best_val_dice,
        }

        # 1. Best model checkpoint
        if val_metrics["val_dice"] > best_val_dice or args.smoke_test:
            best_val_dice = val_metrics["val_dice"]
            state_dict_to_save["best_val_dice"] = best_val_dice
            best_ckpt_path = CHECKPOINTS_DIR / f"{args.model_type}_{args.decoder_type}{suffix}_best.pt"
            torch.save(state_dict_to_save, best_ckpt_path)
            logger.info(f"New best model saved: {best_ckpt_path} (Val Dice: {best_val_dice:.4f})")

        # 2. Latest checkpoint saved every epoch for safe resumption
        latest_ckpt_path = CHECKPOINTS_DIR / f"{args.model_type}_{args.decoder_type}{suffix}_latest.pt"
        torch.save(state_dict_to_save, latest_ckpt_path)

        # 3. Optional periodic checkpoint if save_freq > 0
        save_freq = getattr(args, "save_freq", 0)
        if save_freq > 0 and epoch % save_freq == 0 and not args.smoke_test:
            periodic_ckpt_path = CHECKPOINTS_DIR / f"{args.model_type}_{args.decoder_type}{suffix}_epoch_{epoch}.pt"
            torch.save(state_dict_to_save, periodic_ckpt_path)
            logger.info(f"Periodic checkpoint saved: {periodic_ckpt_path}")

        # Explicit CPU RAM and CUDA memory cleanup between epochs
        if "images" in locals():
            del images
        if "masks" in locals():
            del masks
        if "batch" in locals():
            del batch
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    suffix = "_scratch" if args.from_scratch else ""
    tracker.save_json(LOGS_DIR / f"{args.model_type}_{args.decoder_type}{suffix}_downstream_metrics.json")
    tracker.save_csv(LOGS_DIR / f"{args.model_type}_{args.decoder_type}{suffix}_downstream_metrics.csv")
    logger.info(
        f"3D Downstream Fine-Tuning finished ({'from scratch' if args.from_scratch else 'pre-trained'})."
    )


if __name__ == "__main__":
    main()
