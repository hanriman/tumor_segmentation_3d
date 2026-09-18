#!/usr/bin/env python3
"""
3D Low-Data Label Efficiency Benchmark.
Evaluates model performance under extreme label scarcity (1% to 100% volumetric labels).
"""

import argparse
import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from brats_jepa_3d.config import (
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    METRICS_DIR,
    ensure_directories,
    load_yaml_config,
)
from brats_jepa_3d.data import BraTS3DDataset, VolumetricAugmentations3D
from brats_jepa_3d.losses import CombinedDiceBCELoss3D, DeepSupervisionLoss3D
from brats_jepa_3d.metrics import compute_volumetric_metrics_3d
from brats_jepa_3d.models import BraTS3DnnUNet, BraTS3DUNet, JEPASegmentationModel3D
from brats_jepa_3d.utils import (
    get_autocast_context,
    get_device,
    set_seed,
    setup_logger,
    sort_checkpoints_by_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="3D Label Efficiency Benchmark")
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
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
    parser.add_argument(
        "--model_type",
        type=str,
        default="all",
        help="Specific model to evaluate (visreg_jepa, sigreg_jepa, ijepa, unet_3d, nnunet_3d, or all)",
    )
    parser.add_argument("--smoke_test", action="store_true", help="Run fast verification")
    return parser.parse_args()


def train_and_eval(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int,
    amp: bool = True,
    smoke_test: bool = False,
) -> float:
    use_deep_supervision = getattr(model, "deep_supervision", False)
    criterion = DeepSupervisionLoss3D() if use_deep_supervision else CombinedDiceBCELoss3D()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4)

    warmup_epochs = max(1, min(3, epochs // 5)) if epochs > 1 else 0
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
        device="cuda" if device.type == "cuda" else "cpu", enabled=amp and device.type == "cuda"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            with get_autocast_context(device, enabled=amp):
                out = model(images)
                loss = criterion(out, masks)["loss"]

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

            if smoke_test and batch_idx >= 1:
                break

        scheduler.step()
        if smoke_test:
            break

    # Evaluate on test set
    model.eval()
    test_dices = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            with get_autocast_context(device, enabled=amp):
                out = model(images)
                logits = out[0] if isinstance(out, (list, tuple)) else out
            metrics = compute_volumetric_metrics_3d(logits, masks)
            test_dices.extend(metrics["dice_per_sample"])
            if smoke_test and batch_idx >= 1:
                break

    return float(np.mean(test_dices)) if test_dices else 0.0


def main():
    args = parse_args()
    ensure_directories()
    set_seed(args.seed)
    device = get_device()
    logger = setup_logger("low_data_3d", METRICS_DIR / "low_data_3d.log")
    logger.info("Starting 3D Low-Data Label Efficiency Benchmark...")

    fractions = [0.01, 0.10] if args.smoke_test else args.fractions
    epochs = 1 if args.smoke_test else args.epochs

    aug_tf = VolumetricAugmentations3D(is_training=True)

    try:
        test_dataset = BraTS3DDataset(split="test")
    except FileNotFoundError:
        if not args.smoke_test:
            raise
        logger.warning("BraTS3D test split not found. Using synthetic dataset for smoke test.")
        test_dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": (torch.rand(1, 128, 128, 128) > 0.95).float(),
            }
            for _ in range(2)
        ]
    num_workers = getattr(args, "num_workers", 2)
    use_pin_memory = getattr(args, "pin_memory", True) and (device.type == "cuda")

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    all_models = [
        ("3D VisReg JEPA (FPN)", "visreg_jepa"),
        ("3D SigReg JEPA (FPN)", "sigreg_jepa"),
        ("3D I-JEPA (FPN)", "ijepa"),
        ("3D nnU-Net", "nnunet_3d"),
        ("3D UNet", "unet_3d"),
    ]

    alias_map = {
        "visreg": "visreg_jepa",
        "visreg_jepa": "visreg_jepa",
        "sigreg": "sigreg_jepa",
        "sigreg_jepa": "sigreg_jepa",
        "ijepa": "ijepa",
        "unet": "unet_3d",
        "unet_3d": "unet_3d",
        "nnunet": "nnunet_3d",
        "nnunet_3d": "nnunet_3d",
        "all": "all",
    }

    req_model = alias_map.get(args.model_type.lower(), args.model_type.lower())
    if req_model != "all":
        selected_models = [m for m in all_models if m[1] == req_model]
        if not selected_models:
            logger.warning(f"Unknown model_type '{args.model_type}'. Defaulting to all models.")
            selected_models = all_models
    else:
        selected_models = all_models

    def build_model(name: str):
        jepa_prefixes = {
            "3D VisReg JEPA (FPN)": "visreg_jepa",
            "3D SigReg JEPA (FPN)": "sigreg_jepa",
            "3D I-JEPA (FPN)": "ijepa",
        }
        if name in jepa_prefixes:
            prefix = jepa_prefixes[name]
            jepa_cfg_path = CONFIGS_DIR / "model" / f"{prefix}_3d.yaml"
            jepa_cfg = load_yaml_config(jepa_cfg_path) if jepa_cfg_path.exists() else {}

            # Prioritize pre-trained SSL checkpoints (*_3d_best.pt or *_3d_epoch_*.pt), excluding downstream fine-tuned checkpoints
            ssl_best_ckpts = sorted(CHECKPOINTS_DIR.glob(f"{prefix}*_3d_best.pt"))
            ssl_epoch_ckpts = sort_checkpoints_by_epoch(list(CHECKPOINTS_DIR.glob(f"{prefix}*_epoch_*.pt")))
            if ssl_best_ckpts:
                ckpt_path = ssl_best_ckpts[-1]
            elif ssl_epoch_ckpts:
                ckpt_path = ssl_epoch_ckpts[-1]
            else:
                fallback_ckpts = [
                    p
                    for p in sorted(CHECKPOINTS_DIR.glob(f"{prefix}*.pt"))
                    if not any(dec in p.name for dec in ("multiscale", "bottleneck", "downstream"))
                ]
                ckpt_path = fallback_ckpts[-1] if fallback_ckpts else None

            if not ckpt_path or not ckpt_path.exists():
                logger.warning(
                    f"No pre-trained SSL weights found for {name} ({prefix}_3d_best.pt / {prefix}*_epoch_*.pt). "
                    f"Skipping evaluation (will not train random uninitialized encoder on low data)."
                )
                return None

            ds_flag = False
            try:
                sd_peek = torch.load(ckpt_path, map_location="cpu")
                sd_peek = sd_peek.get("model_state_dict", sd_peek)
                if any(k.startswith("decoder.ds") for k in sd_peek):
                    ds_flag = True
            except (KeyError, OSError, RuntimeError, AttributeError):
                pass

            m = JEPASegmentationModel3D(
                img_size=tuple(jepa_cfg.get("spatial_shape", (128, 128, 128))),
                patch_size=tuple(jepa_cfg.get("patch_size", (16, 16, 16))),
                in_channels=jepa_cfg.get("in_channels", 4),
                embed_dim=jepa_cfg.get("embed_dim", 384),
                encoder_depth=jepa_cfg.get("encoder_depth", 8),
                num_heads=jepa_cfg.get("num_heads", 6),
                mlp_ratio=jepa_cfg.get("mlp_ratio", 4.0),
                decoder_type="multiscale",
                deep_supervision=ds_flag,
            )
            ckpt = torch.load(ckpt_path, map_location=device)
            res = m.load_pretrained_encoder(ckpt)
            logger.info(
                f"Initialized {name} with pre-trained encoder weights from: {ckpt_path.name} ({res['loaded_keys']} keys)"
            )
            return m.to(device)

        elif name == "3D nnU-Net":
            nnunet_cfg_path = CONFIGS_DIR / "model" / "nnunet_3d.yaml"
            nnunet_cfg = load_yaml_config(nnunet_cfg_path) if nnunet_cfg_path.exists() else {}
            return BraTS3DnnUNet(
                in_channels=nnunet_cfg.get("in_channels", 4),
                out_channels=nnunet_cfg.get("out_channels", 1),
                deep_supervision=nnunet_cfg.get("deep_supervision", True),
                deep_supr_num=nnunet_cfg.get("deep_supr_num", 3),
                res_block=nnunet_cfg.get("res_block", True),
            ).to(device)

        elif name == "3D UNet":
            unet_cfg_path = CONFIGS_DIR / "model" / "unet_3d.yaml"
            unet_cfg = load_yaml_config(unet_cfg_path) if unet_cfg_path.exists() else {}
            return BraTS3DUNet(
                in_channels=unet_cfg.get("in_channels", 4),
                out_channels=unet_cfg.get("out_channels", 1),
                channels=tuple(unet_cfg.get("channels", (32, 64, 128, 256, 512))),
                strides=tuple(unet_cfg.get("strides", (2, 2, 2, 2))),
                num_res_units=unet_cfg.get("num_res_units", 2),
                dropout=unet_cfg.get("dropout", 0.1),
            ).to(device)

        else:
            raise ValueError(f"Unknown model name: {name}")

    records = []

    for frac in fractions:
        logger.info(f"\n--- Evaluating Fraction: {frac * 100:.1f}% Labels ---")
        try:
            train_dataset = BraTS3DDataset(
                split="train", fraction=frac, seed=args.seed, augmentations=aug_tf
            )
        except FileNotFoundError:
            if not args.smoke_test:
                raise
            logger.warning(
                f"BraTS3D train split not found for fraction {frac}. Using synthetic dataset for smoke test."
            )
            train_dataset = [
                {
                    "image": torch.randn(4, 128, 128, 128),
                    "mask": (torch.rand(1, 128, 128, 128) > 0.95).float(),
                }
                for _ in range(2)
            ]
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size if not args.smoke_test else 2,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        row = {"Fraction": f"{frac * 100:.1f}%"}
        for name, _ in selected_models:
            m = build_model(name)
            if m is None:
                row[name] = "N/A"
                continue
            dice = train_and_eval(
                m,
                train_loader,
                test_loader,
                device,
                epochs=epochs,
                amp=args.amp,
                smoke_test=args.smoke_test,
            )
            logger.info(f"{name} ({frac * 100:.1f}% labels) -> Test Dice: {dice * 100:.2f}%")
            row[name] = f"{dice * 100:.2f}%"
            del m
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del train_loader, train_dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        records.append(row)

    new_df = pd.DataFrame(records)
    csv_path = METRICS_DIR / "low_data_3d_summary.csv"
    md_path = METRICS_DIR / "low_data_3d_summary.md"

    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            existing_df["Fraction"] = existing_df["Fraction"].astype(str)
            new_df["Fraction"] = new_df["Fraction"].astype(str)
            for col in new_df.columns:
                if col == "Fraction":
                    continue
                new_vals = new_df.set_index("Fraction")[col]
                if not (new_vals == "N/A").all():
                    if col not in existing_df.columns:
                        existing_df[col] = "N/A"
                    for f_val, val in new_vals.items():
                        if val != "N/A":
                            existing_df.loc[existing_df["Fraction"] == f_val, col] = val
            df = existing_df
        except Exception as e:
            logger.warning(f"Could not merge with existing CSV: {e}. Writing new CSV.")
            df = new_df
    else:
        df = new_df

    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 3D Low-Data Label Efficiency Benchmark Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print("\n" + df.to_string(index=False) + "\n")
    logger.info(f"Low-data summary saved to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
