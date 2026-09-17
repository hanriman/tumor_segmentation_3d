#!/usr/bin/env python3
"""
3D Supervised Residual UNet Baseline Runner.
Trains MONAI 3D Residual UNet on multi-modal BraTS 2024 GLI volumes.
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
    load_yaml_config,
    merge_config_with_args,
)
from brats_jepa_3d.data import BraTS3DDataset, VolumetricAugmentations3D
from brats_jepa_3d.losses import CombinedDiceBCELoss3D
from brats_jepa_3d.metrics import compute_volumetric_metrics_3d
from brats_jepa_3d.models import BraTS3DUNet
from brats_jepa_3d.utils import (
    MetricTracker,
    get_autocast_context,
    get_device,
    set_seed,
    setup_logger,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D Residual UNet Baseline")
    parser.add_argument("--config", type=str, default=None, help="Path to base YAML config")
    parser.add_argument("--model_config", type=str, default=None, help="Path to model YAML config")
    parser.add_argument(
        "--dataset_config", type=str, default=None, help="Path to dataset YAML config"
    )
    parser.add_argument(
        "--exp_config", type=str, default=None, help="Path to experiment YAML config"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
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
    parser.add_argument("--smoke_test", action="store_true", help="Run fast verification")
    return parser.parse_args()


def evaluate(model, loader, device, amp: bool = True, smoke_test: bool = False) -> dict[str, float]:
    model.eval()
    all_dices, all_ious, all_hd95s = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with get_autocast_context(device, enabled=amp):
                logits = model(images)

            metrics = compute_volumetric_metrics_3d(logits, masks)
            all_dices.extend(metrics["dice_per_sample"])
            all_ious.extend(metrics["iou_per_sample"])
            all_hd95s.extend(metrics["hd95_per_sample"])

            if smoke_test and batch_idx >= 1:
                break

    return {
        "val_dice": float(np.mean(all_dices)) if all_dices else 0.0,
        "val_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "val_hd95": float(np.mean(all_hd95s)) if all_hd95s else 0.0,
    }


def main():
    args = parse_args()

    # Load configs if available
    base_cfg_path = args.config or (CONFIGS_DIR / "base.yaml")
    if Path(base_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(base_cfg_path), args)

    model_cfg_path = args.model_config or (CONFIGS_DIR / "model" / "unet_3d.yaml")
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
    logger = setup_logger("train_unet_3d", LOGS_DIR / "unet_3d_train.log")
    logger.info(f"Starting 3D Residual UNet Training: Device={device}, AMP={args.amp}")

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
        logger.warning(
            "Processed dataset not found. Generating synthetic volume dataset for verification."
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
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model = BraTS3DUNet(
        in_channels=getattr(args, "in_channels", 4),
        out_channels=getattr(args, "out_channels", 1),
        channels=tuple(getattr(args, "channels", (32, 64, 128, 256, 512))),
        strides=tuple(getattr(args, "strides", (2, 2, 2, 2))),
        num_res_units=getattr(args, "num_res_units", 2),
        dropout=getattr(args, "dropout", 0.1),
    ).to(device)

    criterion = CombinedDiceBCELoss3D()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
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
    tracker = MetricTracker()

    for epoch in range(1, epochs + 1):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            f"Val Dice: {val_metrics['val_dice']:.4f} | Val HD95: {val_metrics['val_hd95']:.2f} mm"
        )

        tracker.update(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": avg_train_loss,
                **val_metrics,
            }
        )

        if val_metrics["val_dice"] > best_val_dice or args.smoke_test:
            best_val_dice = val_metrics["val_dice"]
            best_ckpt_path = CHECKPOINTS_DIR / "unet_3d_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": best_val_dice,
                },
                best_ckpt_path,
            )
            logger.info(f"New best model saved: {best_ckpt_path} (Val Dice: {best_val_dice:.4f})")

        # Explicit CPU RAM and CUDA memory cleanup between epochs
        del images, masks, batch
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tracker.save_json(LOGS_DIR / "unet_3d_metrics.json")
    tracker.save_csv(LOGS_DIR / "unet_3d_metrics.csv")
    logger.info("3D Residual UNet training completed.")


if __name__ == "__main__":
    main()
