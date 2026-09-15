#!/usr/bin/env python3
"""
3D Supervised Residual UNet Baseline Runner.
Trains MONAI 3D Residual UNet on multi-modal BraTS 2024 GLI volumes.
"""

import argparse
import sys
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
from brats_jepa_3d.utils import MetricTracker, get_autocast_context, get_device, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D Residual UNet Baseline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
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
    ensure_directories()
    set_seed(args.seed)
    device = get_device()
    logger = setup_logger("train_unet_3d", LOGS_DIR / "unet_3d_train.log")
    logger.info(f"Starting 3D Residual UNet Training: Device={device}, AMP={args.amp}")

    aug_tf = VolumetricAugmentations3D(
        flip_prob=0.5,
        noise_prob=0.3,
        modality_dropout_prob=0.25,
        is_training=True,
    )

    try:
        train_dataset = BraTS3DDataset(split="train", augmentations=aug_tf)
        val_dataset = BraTS3DDataset(split="val", augmentations=None)
    except FileNotFoundError:
        logger.warning("Processed dataset not found. Generating synthetic volume dataset for verification.")
        train_dataset = [
            {"image": torch.randn(4, 128, 128, 128), "mask": (torch.rand(1, 128, 128, 128) > 0.95).float(), "patient_id": f"train_{i}"}
            for i in range(4)
        ]
        val_dataset = [
            {"image": torch.randn(4, 128, 128, 128), "mask": (torch.rand(1, 128, 128, 128) > 0.95).float(), "patient_id": f"val_{i}"}
            for i in range(2)
        ]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size if not args.smoke_test else 2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = BraTS3DUNet(
        in_channels=4,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    criterion = CombinedDiceBCELoss3D()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    epochs = 1 if args.smoke_test else args.epochs
    scaler = torch.amp.GradScaler(device="cuda" if device.type == "cuda" else "cpu", enabled=args.amp and device.type == "cuda")

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
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if args.smoke_test and batch_idx >= 1:
                break

        val_metrics = evaluate(model, val_loader, device, amp=args.amp, smoke_test=args.smoke_test)
        avg_train_loss = train_loss / max(1, num_batches)
        logger.info(
            f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} | "
            f"Val Dice: {val_metrics['val_dice']:.4f} | Val HD95: {val_metrics['val_hd95']:.2f} mm"
        )

        tracker.update({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **val_metrics,
        })

        if val_metrics["val_dice"] > best_val_dice or args.smoke_test:
            best_val_dice = val_metrics["val_dice"]
            best_ckpt_path = CHECKPOINTS_DIR / "unet_3d_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": best_val_dice,
            }, best_ckpt_path)
            logger.info(f"New best model saved: {best_ckpt_path} (Val Dice: {best_val_dice:.4f})")

    tracker.save_json(LOGS_DIR / "unet_3d_metrics.json")
    tracker.save_csv(LOGS_DIR / "unet_3d_metrics.csv")
    logger.info("3D Residual UNet training completed.")


if __name__ == "__main__":
    main()
