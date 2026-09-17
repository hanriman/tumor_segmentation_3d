#!/usr/bin/env python3
"""
3D Downstream Volumetric Fine-Tuning & Linear Probing Runner.
Couples pre-trained 3D JEPA encoders with Bottleneck or Multi-Scale FPN Decoders.
"""

import argparse
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
from brats_jepa_3d.losses import CombinedDiceBCELoss3D, DeepSupervisionLoss3D
from brats_jepa_3d.metrics import compute_volumetric_metrics_3d
from brats_jepa_3d.models import JEPASegmentationModel3D
from brats_jepa_3d.utils import (
    MetricTracker,
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
        "--decoder_type", type=str, default="multiscale", choices=["multiscale", "bottleneck"]
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
    parser.add_argument("--deep_supervision", action="store_true", default=False)
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
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]

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
    logger = setup_logger(
        "downstream_3d", LOGS_DIR / f"{args.model_type}_{args.decoder_type}_downstream.log"
    )
    logger.info(
        f"Starting 3D Downstream Fine-Tuning: Model={args.model_type}, Decoder={args.decoder_type}, FreezeEncoder={args.freeze_encoder}"
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

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size if not args.smoke_test else 2, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

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
    if args.pretrained_checkpoint:
        ckpt_path = Path(args.pretrained_checkpoint)
    else:
        candidates = sort_checkpoints_by_epoch(
            list(CHECKPOINTS_DIR.glob(f"{args.model_type}*epoch*.pt"))
        ) or sorted(CHECKPOINTS_DIR.glob(f"{args.model_type}*best.pt"))
        if candidates:
            ckpt_path = candidates[-1]

    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        res = model.load_pretrained_encoder(ckpt)
        logger.info(
            f"Loaded pre-trained encoder weights from {ckpt_path} ({res['loaded_keys']} keys matched)"
        )
    else:
        logger.info("No pre-trained checkpoint specified or found. Training encoder from random initialization.")

    criterion = DeepSupervisionLoss3D() if args.deep_supervision else CombinedDiceBCELoss3D()
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
            best_ckpt_path = CHECKPOINTS_DIR / f"{args.model_type}_{args.decoder_type}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": best_val_dice,
                },
                best_ckpt_path,
            )
            logger.info(f"New best model saved: {best_ckpt_path} (Val Dice: {best_val_dice:.4f})")

    tracker.save_json(LOGS_DIR / f"{args.model_type}_{args.decoder_type}_downstream_metrics.json")
    tracker.save_csv(LOGS_DIR / f"{args.model_type}_{args.decoder_type}_downstream_metrics.csv")
    logger.info("3D Downstream Fine-Tuning finished.")


if __name__ == "__main__":
    main()
