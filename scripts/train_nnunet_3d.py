#!/usr/bin/env python3
"""
3D Supervised nnU-Net Baseline Runner (DynUNet with Deep Supervision).
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
from brats_jepa_3d.losses import DeepSupervisionLoss3D
from brats_jepa_3d.metrics import compute_volumetric_metrics_3d
from brats_jepa_3d.models import BraTS3DnnUNet
from brats_jepa_3d.utils import (
    MetricTracker,
    get_autocast_context,
    get_device,
    set_seed,
    setup_logger,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D nnU-Net Baseline")
    parser.add_argument("--config", type=str, default=None, help="Path to base YAML config")
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
                out = model(images)
                logits = out[0] if isinstance(out, (list, tuple)) else out

            metrics = compute_volumetric_metrics_3d(logits, masks, compute_hd95=False)
            all_dices.extend(metrics["dice_per_sample"])
            all_ious.extend(metrics["iou_per_sample"])

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

    model_cfg_path = args.model_config or (CONFIGS_DIR / "model" / "nnunet_3d.yaml")
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
    logger = setup_logger("train_nnunet_3d", LOGS_DIR / "nnunet_3d_train.log")
    logger.info(
        f"Starting 3D nnU-Net Training with Deep Supervision: Device={device}, AMP={args.amp}"
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

    model = BraTS3DnnUNet(
        in_channels=getattr(args, "in_channels", 4),
        out_channels=getattr(args, "out_channels", 1),
        deep_supervision=getattr(args, "deep_supervision", True),
        deep_supr_num=getattr(args, "deep_supr_num", 3),
        res_block=getattr(args, "res_block", True),
    ).to(device)

    criterion = DeepSupervisionLoss3D()
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

    start_epoch = 1
    best_val_dice = -1.0
    tracker = MetricTracker()

    # Resume training if checkpoint specified
    if getattr(args, "resume", None):
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found for --resume: {resume_path}")
        logger.info(f"Resuming training from checkpoint: {resume_path}")
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
                out = model(images)
                loss_dict = criterion(out, masks)
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
            best_ckpt_path = CHECKPOINTS_DIR / "nnunet_3d_best.pt"
            torch.save(state_dict_to_save, best_ckpt_path)
            logger.info(f"New best model saved: {best_ckpt_path} (Val Dice: {best_val_dice:.4f})")

        # 2. Latest checkpoint saved every epoch for safe resumption (overwritten in place)
        latest_ckpt_path = CHECKPOINTS_DIR / "nnunet_3d_latest.pt"
        torch.save(state_dict_to_save, latest_ckpt_path)

        # 3. Optional periodic checkpoint if save_freq > 0
        save_freq = getattr(args, "save_freq", 0)
        if save_freq > 0 and epoch % save_freq == 0 and not args.smoke_test:
            periodic_ckpt_path = CHECKPOINTS_DIR / f"nnunet_3d_epoch_{epoch}.pt"
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

    tracker.save_json(LOGS_DIR / "nnunet_3d_metrics.json")
    tracker.save_csv(LOGS_DIR / "nnunet_3d_metrics.csv")
    logger.info("3D nnU-Net training completed.")


if __name__ == "__main__":
    main()
