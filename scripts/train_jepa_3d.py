#!/usr/bin/env python3
"""
3D Self-Supervised JEPA Pre-training Runner.
Supports 3D I-JEPA, 3D SigReg JEPA, and 3D VisReg JEPA.
"""

import argparse
import math
from pathlib import Path

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
from brats_jepa_3d.data import (
    BraTS3DDataset,
    JEPAMaskingTransform3D,
    VolumetricAugmentations3D,
    jepa_masking_collate_fn_3d,
)
from brats_jepa_3d.losses import IJEPALoss, SigRegLoss, VisRegLoss
from brats_jepa_3d.models import IJEPA3D, SigRegJEPA3D, VisRegJEPA3D
from brats_jepa_3d.utils import (
    MetricTracker,
    get_autocast_context,
    get_device,
    set_seed,
    setup_logger,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train 3D JEPA models on BraTS 2024 GLI")
    parser.add_argument(
        "--model_type",
        type=str,
        default="sigreg_jepa",
        choices=["ijepa", "sigreg_jepa", "visreg_jepa"],
    )
    parser.add_argument("--config", type=str, default=None, help="Path to base YAML config")
    parser.add_argument("--model_config", type=str, default=None, help="Path to model YAML config")
    parser.add_argument(
        "--exp_config", type=str, default=None, help="Path to experiment YAML config"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--smoke_test", action="store_true", help="Run 1 epoch with 2 batches for fast verification"
    )
    return parser.parse_args()


def get_lr_scheduler(
    optimizer, warmup_epochs: int, total_epochs: int, base_lr: float, min_lr: float = 1e-5
):
    """Cosine learning rate schedule with linear warmup."""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()

    # Load configs if available
    base_cfg_path = args.config or (CONFIGS_DIR / "base.yaml")
    if Path(base_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(base_cfg_path), args)

    model_cfg_path = args.model_config or (CONFIGS_DIR / "model" / f"{args.model_type}_3d.yaml")
    if Path(model_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(model_cfg_path), args)

    exp_cfg_path = args.exp_config or (CONFIGS_DIR / "experiment" / "pretrain_50ep.yaml")
    if Path(exp_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(exp_cfg_path), args)

    ensure_directories()
    set_seed(args.seed)
    device = get_device()
    logger = setup_logger("pretrain_3d", LOGS_DIR / f"{args.model_type}_pretrain.log")
    logger.info(
        f"Starting 3D JEPA Pre-training: Model={args.model_type}, Device={device}, AMP={args.amp}"
    )

    # Initialize 3D Data Pipeline
    masking_tf = JEPAMaskingTransform3D(
        grid_size=(8, 8, 8),
        num_target_cuboids=4,
        target_cuboid_size=(3, 3, 3),
        context_num_patches=192,
    )
    aug_tf = VolumetricAugmentations3D(
        flip_prob=0.5,
        noise_prob=0.3,
        modality_dropout_prob=0.25,
        is_training=True,
    )

    try:
        dataset = BraTS3DDataset(
            split="train",
            masking_transform=masking_tf,
            augmentations=aug_tf,
        )
    except FileNotFoundError:
        logger.warning(
            "Processed dataset not found. Generating synthetic volume dataset for verification."
        )
        # Synthetic dataset fallback for testing
        dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": (torch.rand(1, 128, 128, 128) > 0.95).float(),
                "patient_id": f"synthetic_{i}",
                **masking_tf(),
            }
            for i in range(8)
        ]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size if not args.smoke_test else 2,
        shuffle=True,
        collate_fn=jepa_masking_collate_fn_3d,
        num_workers=0,
    )

    # Initialize Model & Loss
    if args.model_type == "ijepa":
        model = IJEPA3D(
            img_size=(128, 128, 128),
            patch_size=(16, 16, 16),
            in_channels=4,
            embed_dim=384,
            encoder_depth=8,
            predictor_depth=4,
            predictor_embed_dim=192,
        ).to(device)
        criterion = IJEPALoss(loss_type="smooth_l1")

    elif args.model_type == "sigreg_jepa":
        model = SigRegJEPA3D(
            img_size=(128, 128, 128),
            patch_size=(16, 16, 16),
            in_channels=4,
            embed_dim=384,
            proj_dim=128,
            encoder_depth=8,
            predictor_depth=4,
            predictor_embed_dim=192,
            sigreg_weight=getattr(args, "sigreg_weight", 1.0),
        ).to(device)
        criterion = SigRegLoss(
            loss_type="smooth_l1",
            sigreg_weight=getattr(args, "sigreg_weight", 1.0),
            num_projections=256,
        )

    elif args.model_type == "visreg_jepa":
        model = VisRegJEPA3D(
            img_size=(128, 128, 128),
            patch_size=(16, 16, 16),
            in_channels=4,
            embed_dim=384,
            proj_dim=128,
            encoder_depth=8,
            predictor_depth=4,
            predictor_embed_dim=192,
        ).to(device)
        criterion = VisRegLoss(
            loss_type="smooth_l1",
            center_weight=getattr(args, "center_weight", 1.0),
            scale_weight=getattr(args, "scale_weight", 1.0),
            swd_weight=getattr(args, "swd_weight", 1.0),
            num_projections=256,
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    epochs = 1 if args.smoke_test else args.epochs
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_epochs=max(1, epochs // 10),
        total_epochs=epochs,
        base_lr=args.learning_rate,
    )
    scaler = torch.amp.GradScaler(
        device="cuda" if device.type == "cuda" else "cpu",
        enabled=args.amp and device.type == "cuda",
    )

    tracker = MetricTracker()
    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(device)
            ctx_idx = batch["context_indices"].to(device)
            tgt_idx_list = [t.to(device) for t in batch["target_indices_list"]]

            optimizer.zero_grad()

            with get_autocast_context(device, enabled=args.amp):
                out = model(images, ctx_idx, tgt_idx_list)

                if args.model_type == "ijepa":
                    loss = criterion(out["predictions"], out["targets"])
                    loss_dict = {"loss": loss, "jepa_loss": loss}
                elif args.model_type == "sigreg_jepa" or args.model_type == "visreg_jepa":
                    loss_dict = criterion(
                        out["predictions"],
                        out["targets"],
                        projected_tokens=out["projected_tokens"],
                    )
                    loss = loss_dict["loss"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            # For I-JEPA: update target encoder via EMA
            if args.model_type == "ijepa":
                model.update_target_encoder()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"}
            )

            if args.smoke_test and batch_idx >= 1:
                break

        scheduler.step()
        avg_loss = epoch_loss / max(1, num_batches)
        tracker.update({"epoch": epoch, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]})
        logger.info(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

        # Save checkpoint
        if epoch % 10 == 0 or epoch == epochs or args.smoke_test:
            ckpt_path = CHECKPOINTS_DIR / f"{args.model_type}_3d_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.context_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

    # Save metrics history
    tracker.save_json(LOGS_DIR / f"{args.model_type}_pretrain_metrics.json")
    tracker.save_csv(LOGS_DIR / f"{args.model_type}_pretrain_metrics.csv")
    logger.info("3D JEPA Pre-training completed successfully.")


if __name__ == "__main__":
    main()
