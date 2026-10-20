"""
Unified 3D Supervised Segmentation Trainer.

Provides shared CLI argument registration, YAML configuration merging, DataLoader
construction, warmup+cosine learning-rate scheduling, validation evaluation, and
the AMP/GradScaler training & checkpointing loop shared across:
- `scripts/train_downstream_3d.py` (JEPA / ViT fine-tuning & probing)
- `scripts/train_unet_3d.py` (3D Residual U-Net baseline)
- `scripts/train_nnunet_3d.py` (3D DynUNet / nnU-Net baseline)
"""

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from brats_jepa_3d.config import (
    CONFIGS_DIR,
    load_yaml_config,
    merge_config_with_args,
)
from brats_jepa_3d.data import BraTS3DDataset, VolumetricAugmentations3D
from brats_jepa_3d.losses import build_segmentation_criterion, resolve_seg_loss_type
from brats_jepa_3d.metrics import validation_dice_iou
from brats_jepa_3d.utils import MetricTracker, get_autocast_context


def add_common_supervised_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Registers the shared CLI flags used by all 3D supervised segmentation runners."""
    parser.add_argument("--config", type=str, default=None, help="Path to base YAML config")
    parser.add_argument("--model_config", type=str, default=None, help="Path to model YAML config")
    parser.add_argument(
        "--dataset_config", type=str, default=None, help="Path to dataset YAML config"
    )
    parser.add_argument(
        "--exp_config",
        "--experiment_config",
        dest="exp_config",
        type=str,
        default=None,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to 3D dataset directory (default: auto-detected)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs",
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
    parser.add_argument("--deep_supervision", action="store_true", default=True)
    parser.add_argument(
        "--no_deep_supervision", "--no-deep-supervision", action="store_false", dest="deep_supervision"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="dice_bce",
        choices=["dice_bce", "tversky"],
        help="Overlap loss: symmetric Dice+BCE (default) or asymmetric Tversky(beta=0.7)+BCE",
    )
    parser.add_argument("--tversky_alpha", type=float, default=0.3)
    parser.add_argument("--tversky_beta", type=float, default=0.7)
    parser.add_argument(
        "--include_background",
        action="store_true",
        default=None,
        help="Include background in overlap loss (default: auto -> excluded for C>1, included for C=1)",
    )
    parser.add_argument(
        "--no_include_background",
        action="store_false",
        dest="include_background",
        help="Exclude background from overlap loss",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=5,
        help="Number of output segmentation classes (default: 5 for BraTS multi-region)",
    )
    return parser


def load_supervised_configs(
    args: argparse.Namespace,
    default_model_yaml: str,
) -> argparse.Namespace:
    """Merges base, model, dataset, and finetune YAML configs with CLI priority."""
    base_cfg_path = args.config or (CONFIGS_DIR / "base.yaml")
    if Path(base_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(base_cfg_path), args)

    model_cfg_path = args.model_config or (CONFIGS_DIR / "model" / default_model_yaml)
    if Path(model_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(model_cfg_path), args)

    dataset_cfg_path = args.dataset_config or (CONFIGS_DIR / "dataset" / "brats3d.yaml")
    if Path(dataset_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(dataset_cfg_path), args)

    exp_cfg_path = args.exp_config or (CONFIGS_DIR / "experiment" / "finetune.yaml")
    if Path(exp_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(exp_cfg_path), args)

    return args


def build_train_val_loaders(
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[DataLoader, DataLoader, Path | None]:
    """Constructs 3D training and validation DataLoaders with synthetic smoke-test fallback."""
    aug_tf = VolumetricAugmentations3D(
        flip_prob=getattr(args, "rand_flip_prob", getattr(args, "flip_prob", 0.5)),
        noise_prob=getattr(args, "rand_noise_prob", getattr(args, "noise_prob", 0.3)),
        noise_std=getattr(args, "noise_std", 0.05),
        modality_dropout_prob=getattr(args, "modality_dropout_prob", 0.25),
        is_training=True,
    )

    resolved_data_dir = Path(args.data_dir).resolve() if getattr(args, "data_dir", None) else None
    try:
        train_dataset = BraTS3DDataset(split="train", augmentations=aug_tf, data_dir=resolved_data_dir)
        val_dataset = BraTS3DDataset(split="val", augmentations=None, data_dir=resolved_data_dir)
    except FileNotFoundError:
        if not args.smoke_test:
            raise
        logger.warning(
            "Processed dataset not found. Generating synthetic volume dataset for smoke test verification."
        )
        train_dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": torch.randint(0, 5, (1, 128, 128, 128)).long(),
                "patient_id": f"train_{i}",
            }
            for i in range(4)
        ]
        val_dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": torch.randint(0, 5, (1, 128, 128, 128)).long(),
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
    return train_loader, val_loader, resolved_data_dir


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Constructs a linear warmup (20% of epochs, capped at 5) + cosine annealing scheduler."""
    warmup_epochs = max(1, min(5, epochs // 5)) if epochs > 1 else 0
    if epochs > 1 and warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=eta_min
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=eta_min)


def evaluate_validation_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    smoke_test: bool = False,
) -> dict[str, float]:
    """Evaluates mean BraTS macro Dice and IoU across `loader`."""
    model.eval()
    all_dices, all_ious = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validating", leave=False)):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with get_autocast_context(device, enabled=amp):
                out = model(images)
                logits = out[0] if isinstance(out, (list, tuple)) else out

            d, i = validation_dice_iou(logits, masks)
            all_dices.append(d)
            all_ious.append(i)

            if smoke_test and batch_idx >= 1:
                break

    return {
        "val_dice": float(np.mean(all_dices)) if all_dices else 0.0,
        "val_iou": float(np.mean(all_ious)) if all_ious else 0.0,
    }


def run_supervised_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    checkpoints_dir: Path,
    logs_dir: Path,
    ckpt_prefix: str,
    metrics_prefix: str,
    logger: logging.Logger,
    require_resume_exists: bool = False,
) -> MetricTracker:
    """Executes the supervised 3D segmentation training, validation, and checkpointing loop."""
    criterion = build_segmentation_criterion(
        resolve_seg_loss_type(args),
        deep_supervision=getattr(args, "deep_supervision", True),
        tversky_alpha=getattr(args, "tversky_alpha", 0.3),
        tversky_beta=getattr(args, "tversky_beta", 0.7),
        num_classes=getattr(args, "out_channels", 5),
        include_background=getattr(args, "include_background", None),
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    epochs = 1 if args.smoke_test else args.epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(
        device="cuda" if device.type == "cuda" else "cpu",
        enabled=args.amp and device.type == "cuda",
    )

    start_epoch = 1
    best_val_dice = -1.0
    tracker = MetricTracker()

    if getattr(args, "resume", None):
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            if require_resume_exists:
                raise FileNotFoundError(f"Checkpoint file not found for --resume: {resume_path}")
        else:
            logger.info(f"Resuming training from checkpoint: {resume_path}")
            try:
                resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            except TypeError:
                resume_ckpt = torch.load(resume_path, map_location=device)
            if "model_state_dict" in resume_ckpt:
                model.load_state_dict(resume_ckpt["model_state_dict"])
            if "optimizer_state_dict" in resume_ckpt and resume_ckpt["optimizer_state_dict"]:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in resume_ckpt and resume_ckpt["scheduler_state_dict"]:
                scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            if (
                "scaler_state_dict" in resume_ckpt
                and resume_ckpt["scaler_state_dict"]
                and scaler.is_enabled()
            ):
                scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
            if "epoch" in resume_ckpt:
                start_epoch = resume_ckpt["epoch"] + 1
            if "best_val_dice" in resume_ckpt:
                best_val_dice = float(resume_ckpt["best_val_dice"])
            logger.info(
                f"Resumed successfully at epoch {start_epoch} with best_val_dice: {best_val_dice:.4f}"
            )

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
        val_metrics = evaluate_validation_split(
            model, val_loader, device, amp=args.amp, smoke_test=args.smoke_test
        )
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
            best_ckpt_path = checkpoints_dir / f"{ckpt_prefix}_best.pt"
            best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict_to_save, best_ckpt_path)
            logger.info(f"New best model saved: {best_ckpt_path} (Val Dice: {best_val_dice:.4f})")

        # 2. Periodic or final epoch checkpoint (guarantees final epoch is preserved without duplicate latest.pt)
        save_freq = getattr(args, "save_freq", 0)
        is_periodic = save_freq > 0 and epoch % save_freq == 0 and not args.smoke_test
        is_final_epoch = epoch == epochs or args.smoke_test
        if is_periodic or is_final_epoch:
            periodic_ckpt_path = checkpoints_dir / f"{ckpt_prefix}_epoch_{epoch}.pt"
            periodic_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict_to_save, periodic_ckpt_path)
            logger.info(f"Checkpoint saved: {periodic_ckpt_path}")

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

    tracker.save_json(logs_dir / f"{metrics_prefix}_metrics.json")
    tracker.save_csv(logs_dir / f"{metrics_prefix}_metrics.csv")
    return tracker
