#!/usr/bin/env python3
"""
3D Self-Supervised JEPA Pre-training Runner.
Supports 3D I-JEPA, 3D SigReg JEPA, and 3D VisReg JEPA.
"""

import argparse
import gc
import math
from pathlib import Path
import time

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
from brats_jepa_3d.metrics import compute_representation_collapse_metrics
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
        "--dataset_config", type=str, default=None, help="Path to dataset YAML config"
    )
    parser.add_argument(
        "--exp_config", type=str, default=None, help="Path to experiment YAML config"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", "--no-amp", action="store_false", dest="amp")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
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

    dataset_cfg_path = args.dataset_config or (CONFIGS_DIR / "dataset" / "brats3d.yaml")
    if Path(dataset_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(dataset_cfg_path), args)

    model_cfg_path = args.model_config or (CONFIGS_DIR / "model" / f"{args.model_type}_3d.yaml")
    if Path(model_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(model_cfg_path), args)

    exp_cfg_path = args.exp_config or (CONFIGS_DIR / "experiment" / "pretrain_50ep.yaml")
    if Path(exp_cfg_path).exists():
        args = merge_config_with_args(load_yaml_config(exp_cfg_path), args)

    ensure_directories()
    set_seed(args.seed)
    device = get_device()
    logger = setup_logger(f"train_{args.model_type}", LOGS_DIR / f"{args.model_type}_pretrain.log")
    logger.info(
        f"Starting 3D JEPA Pre-training: Model={args.model_type}, Device={device}, AMP={args.amp}"
    )

    # Initialize 3D Data Pipeline with config parameters
    target_cuboid_size = getattr(args, "target_cuboid_size", (3, 3, 3))
    if isinstance(target_cuboid_size, list):
        target_cuboid_size = tuple(target_cuboid_size)
    grid_size = getattr(args, "grid_size", (8, 8, 8))
    if isinstance(grid_size, list):
        grid_size = tuple(grid_size)

    masking_tf = JEPAMaskingTransform3D(
        grid_size=grid_size,
        num_target_cuboids=getattr(args, "num_target_cuboids", 4),
        target_cuboid_size=target_cuboid_size,
        max_target_overlap=getattr(args, "max_target_overlap", 0.0),
        context_num_patches=getattr(args, "context_num_patches", 192),
        connectivity=getattr(args, "context_connectivity", getattr(args, "connectivity", 26)),
    )
    aug_tf = VolumetricAugmentations3D(
        flip_prob=getattr(args, "rand_flip_prob", getattr(args, "flip_prob", 0.5)),
        noise_prob=getattr(args, "rand_noise_prob", getattr(args, "noise_prob", 0.3)),
        modality_dropout_prob=getattr(args, "modality_dropout_prob", 0.25),
        is_training=True,
    )

    try:
        train_dataset = BraTS3DDataset(
            split="train",
            masking_transform=masking_tf,
            augmentations=aug_tf,
        )
        val_dataset = BraTS3DDataset(
            split="val",
            masking_transform=masking_tf,
            augmentations=None,
        )
        if len(val_dataset) == 0:
            val_dataset = train_dataset
    except FileNotFoundError:
        logger.warning(
            "Processed dataset not found. Generating synthetic volume dataset for verification."
        )
        # Synthetic dataset fallback for testing
        train_dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": (torch.rand(1, 128, 128, 128) > 0.95).float(),
                "patient_id": f"synthetic_train_{i}",
                **masking_tf(),
            }
            for i in range(8)
        ]
        val_dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": (torch.rand(1, 128, 128, 128) > 0.95).float(),
                "patient_id": f"synthetic_val_{i}",
                **masking_tf(),
            }
            for i in range(4)
        ]

    num_workers = getattr(args, "num_workers", 2)
    use_pin_memory = getattr(args, "pin_memory", True) and (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size if not args.smoke_test else 2,
        shuffle=True,
        collate_fn=jepa_masking_collate_fn_3d,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size if not args.smoke_test else 2,
        shuffle=False,
        collate_fn=jepa_masking_collate_fn_3d,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    logger.info(
        f"Loaded {len(train_dataset)} training volumes and {len(val_dataset)} validation volumes."
    )

    spatial_shape = tuple(getattr(args, "spatial_shape", (128, 128, 128)))
    patch_size = tuple(getattr(args, "patch_size", (16, 16, 16)))
    in_channels = getattr(args, "in_channels", 4)
    embed_dim = getattr(args, "embed_dim", 384)
    encoder_depth = getattr(args, "encoder_depth", 8)
    predictor_depth = getattr(args, "predictor_depth", 4)
    predictor_embed_dim = getattr(args, "predictor_embed_dim", 192)

    # Initialize Model & Loss
    if args.model_type == "ijepa":
        model = IJEPA3D(
            img_size=spatial_shape,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            predictor_depth=predictor_depth,
            predictor_embed_dim=predictor_embed_dim,
        ).to(device)
        criterion = IJEPALoss(loss_type=getattr(args, "loss_type", "smooth_l1"))

    elif args.model_type == "sigreg_jepa":
        model = SigRegJEPA3D(
            img_size=spatial_shape,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            proj_dim=getattr(args, "proj_dim", 128),
            encoder_depth=encoder_depth,
            predictor_depth=predictor_depth,
            predictor_embed_dim=predictor_embed_dim,
            sigreg_weight=getattr(args, "sigreg_weight", 1.0),
        ).to(device)
        criterion = SigRegLoss(
            loss_type=getattr(args, "loss_type", "smooth_l1"),
            sigreg_weight=getattr(args, "sigreg_weight", 1.0),
            num_projections=getattr(args, "num_projections", 256),
            t_max=getattr(args, "t_max", 3.0),
            n_knots=getattr(args, "n_knots", 17),
            normalize_measure=getattr(args, "normalize_measure", True),
        )

    elif args.model_type == "visreg_jepa":
        model = VisRegJEPA3D(
            img_size=spatial_shape,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            proj_dim=getattr(args, "proj_dim", 128),
            encoder_depth=encoder_depth,
            predictor_depth=predictor_depth,
            predictor_embed_dim=predictor_embed_dim,
        ).to(device)
        criterion = VisRegLoss(
            loss_type=getattr(args, "loss_type", "smooth_l1"),
            center_weight=getattr(args, "center_weight", 1.0),
            scale_weight=getattr(args, "scale_weight", 1.0),
            swd_weight=getattr(args, "swd_weight", 1.0),
            num_projections=getattr(args, "num_projections", 256),
            target_std=getattr(args, "target_std", 1.0),
            swd_metric=getattr(args, "swd_metric", "mse"),
            scale_loss_type=getattr(args, "scale_loss_type", "squared"),
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

    steps_per_epoch = min(len(train_loader), 2) if args.smoke_test else len(train_loader)
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        train_sublosses: dict[str, float] = {}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
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

            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    train_sublosses[k] = train_sublosses.get(k, 0.0) + v.item()

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

            # Technical Decision (I-JEPA EMA Momentum Annealing):
            # Target encoder weights are updated via an Exponential Moving Average (EMA):
            #   \bar{\theta}_t \leftarrow m_t \bar{\theta}_{t-1} + (1 - m_t) \theta_t
            # The momentum parameter m_t is annealed via a half-period cosine schedule:
            #   m_t = m_end - (m_end - m_start) * 0.5 * (1 + cos(\pi * progress))
            # starting at m_start (default 0.996) and monotonically increasing to m_end (1.0).
            # Hyperparameters are dynamically loaded from config (configs/model/ijepa_3d.yaml).
            if args.model_type == "ijepa":
                m_start = getattr(args, "ema_start_momentum", 0.996)
                m_end = getattr(args, "ema_end_momentum", 1.0)
                anneal_epochs = getattr(args, "ema_anneal_epochs", epochs)
                anneal_steps = anneal_epochs * steps_per_epoch
                progress = min(1.0, global_step / max(1, anneal_steps))
                curr_momentum = m_end - (m_end - m_start) * 0.5 * (1.0 + math.cos(math.pi * progress))
                model.update_target_encoder(momentum=curr_momentum)

            global_step += 1
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"}
            )

            if args.smoke_test and batch_idx >= 1:
                break

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, num_batches)
        avg_train_sublosses = {
            f"train_{k}": v / max(1, num_batches) for k, v in train_sublosses.items()
        }

        # Validation evaluation
        model.eval()
        val_loss_sum = 0.0
        val_sublosses: dict[str, float] = {}
        val_batches = 0

        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(val_loader):
                if args.smoke_test and val_batch_idx >= 1:
                    break
                val_images = val_batch["images"].to(device)
                val_ctx_idx = val_batch["context_indices"].to(device)
                val_tgt_idx_list = [t.to(device) for t in val_batch["target_indices_list"]]

                with get_autocast_context(device, enabled=args.amp):
                    val_out = model(val_images, val_ctx_idx, val_tgt_idx_list)
                    if args.model_type == "ijepa":
                        v_loss = criterion(val_out["predictions"], val_out["targets"])
                        v_loss_dict = {"loss": v_loss, "jepa_loss": v_loss}
                    elif args.model_type in ("sigreg_jepa", "visreg_jepa"):
                        v_loss_dict = criterion(
                            val_out["predictions"],
                            val_out["targets"],
                            projected_tokens=val_out["projected_tokens"],
                        )
                        v_loss = v_loss_dict["loss"]

                val_loss_sum += v_loss.item()
                val_batches += 1
                for k, v in v_loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        val_sublosses[k] = val_sublosses.get(k, 0.0) + v.item()

                del val_images, val_ctx_idx, val_tgt_idx_list, val_batch, val_out

        avg_val_loss = val_loss_sum / max(1, val_batches) if val_batches > 0 else avg_train_loss
        avg_val_sublosses = (
            {f"val_{k}": v / max(1, val_batches) for k, v in val_sublosses.items()}
            if val_batches > 0
            else {}
        )
        epoch_duration = time.perf_counter() - epoch_start

        # Detailed component loss formatting matching 2D training logs
        if args.model_type == "sigreg_jepa":
            loss_detail = f" (JEPA: {avg_train_sublosses.get('train_jepa_loss', 0.0):.4f}, SigReg: {avg_train_sublosses.get('train_sigreg_loss', 0.0):.4f})"
        elif args.model_type == "visreg_jepa":
            loss_detail = (
                f" (JEPA: {avg_train_sublosses.get('train_jepa_loss', 0.0):.4f}, "
                f"Ctr: {avg_train_sublosses.get('train_center_loss', 0.0):.4f}, "
                f"Scl: {avg_train_sublosses.get('train_scale_loss', 0.0):.4f}, "
                f"Shp: {avg_train_sublosses.get('train_shape_loss', 0.0):.4f})"
            )
        else:
            loss_detail = ""

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.5f}{loss_detail} | Val Loss: {avg_val_loss:.5f} | Duration: {epoch_duration:.2f}s"
        )

        # Compute representation quality metrics at checkpoint intervals
        rep_metrics = {}
        if epoch % 10 == 0 or epoch == epochs or args.smoke_test:
            model.eval()
            with torch.no_grad():
                sample_tokens = model.context_encoder(images[:1])
                rep_metrics = compute_representation_collapse_metrics(sample_tokens)
            logger.info(
                f"Representation Quality - EffRank: {rep_metrics['effective_rank']:.2f} | "
                f"CenteredCosSim: {rep_metrics['avg_cosine_sim_centered']:.4f} | "
                f"FeatureVar: {rep_metrics['feature_variance']:.4f}"
            )
            model.train()

        epoch_metrics = {
            "epoch": epoch,
            "loss": avg_train_loss,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch_duration_sec": epoch_duration,
            "lr": scheduler.get_last_lr()[0],
            **avg_train_sublosses,
            **avg_val_sublosses,
            **rep_metrics,
        }
        tracker.update(epoch_metrics)

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = CHECKPOINTS_DIR / f"{args.model_type}_3d_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.context_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                best_ckpt_path,
            )
            logger.info(f"Best model checkpoint saved: {best_ckpt_path} (val_loss: {best_val_loss:.5f})")

        # Save periodic checkpoint
        if epoch % 10 == 0 or epoch == epochs or args.smoke_test:
            ckpt_path = CHECKPOINTS_DIR / f"{args.model_type}_3d_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.context_encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

        # Explicit CPU RAM and CUDA memory cleanup between epochs
        if "images" in locals():
            del images
        if "ctx_idx" in locals():
            del ctx_idx
        if "tgt_idx_list" in locals():
            del tgt_idx_list
        if "batch" in locals():
            del batch
        if "sample_tokens" in locals():
            del sample_tokens
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save metrics history
    tracker.save_json(LOGS_DIR / f"{args.model_type}_pretrain_metrics.json")
    tracker.save_csv(LOGS_DIR / f"{args.model_type}_pretrain_metrics.csv")
    logger.info("3D JEPA Pre-training completed successfully.")


if __name__ == "__main__":
    main()
