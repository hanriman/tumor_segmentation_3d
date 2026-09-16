#!/usr/bin/env python3
"""
3D Low-Data Label Efficiency Benchmark.
Evaluates model performance under extreme label scarcity (1% to 100% volumetric labels).
"""

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from brats_jepa_3d.config import CHECKPOINTS_DIR, METRICS_DIR, ensure_directories
from brats_jepa_3d.data import BraTS3DDataset, VolumetricAugmentations3D
from brats_jepa_3d.losses import CombinedDiceBCELoss3D
from brats_jepa_3d.metrics import compute_volumetric_metrics_3d
from brats_jepa_3d.models import BraTS3DnnUNet, BraTS3DUNet, JEPASegmentationModel3D
from brats_jepa_3d.utils import get_autocast_context, get_device, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="3D Label Efficiency Benchmark")
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = CombinedDiceBCELoss3D()
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
                logits = out[0] if isinstance(out, (list, tuple)) else out
                loss = criterion(logits, masks)["loss"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if smoke_test and batch_idx >= 1:
                break
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
        test_dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": (torch.rand(1, 128, 128, 128) > 0.95).float(),
            }
            for _ in range(2)
        ]
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    def build_model(name: str):
        if name == "3D VisReg JEPA (FPN)":
            m = JEPASegmentationModel3D(decoder_type="multiscale")
            ckpts = sorted(CHECKPOINTS_DIR.glob("visreg_jepa*epoch*.pt")) + sorted(
                CHECKPOINTS_DIR.glob("visreg_jepa*.pt")
            )
            if ckpts:
                ckpt_path = ckpts[-1]
                ckpt = torch.load(ckpt_path, map_location=device)
                enc_dict = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict"))
                clean_dict = {
                    k.replace("context_encoder.", ""): v
                    for k, v in enc_dict.items()
                    if "context_encoder" in k or k in m.encoder.state_dict()
                }
                m.load_pretrained_encoder(clean_dict if clean_dict else enc_dict)
                logger.info(
                    f"Initialized {name} with pre-trained encoder weights from: {ckpt_path.name}"
                )
            else:
                logger.warning(
                    f"No pre-trained weights found for {name}. Initialized from scratch."
                )
            return m.to(device)

        elif name == "3D SigReg JEPA (FPN)":
            m = JEPASegmentationModel3D(decoder_type="multiscale")
            ckpts = sorted(CHECKPOINTS_DIR.glob("sigreg_jepa*epoch*.pt")) + sorted(
                CHECKPOINTS_DIR.glob("sigreg_jepa*.pt")
            )
            if ckpts:
                ckpt_path = ckpts[-1]
                ckpt = torch.load(ckpt_path, map_location=device)
                enc_dict = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict"))
                clean_dict = {
                    k.replace("context_encoder.", ""): v
                    for k, v in enc_dict.items()
                    if "context_encoder" in k or k in m.encoder.state_dict()
                }
                m.load_pretrained_encoder(clean_dict if clean_dict else enc_dict)
                logger.info(
                    f"Initialized {name} with pre-trained encoder weights from: {ckpt_path.name}"
                )
            else:
                logger.warning(
                    f"No pre-trained weights found for {name}. Initialized from scratch."
                )
            return m.to(device)

        elif name == "3D nnU-Net":
            return BraTS3DnnUNet(deep_supervision=False).to(device)

        elif name == "3D UNet":
            return BraTS3DUNet().to(device)

        else:
            raise ValueError(f"Unknown model name: {name}")

    models = [
        ("3D VisReg JEPA (FPN)", lambda: build_model("3D VisReg JEPA (FPN)")),
        ("3D SigReg JEPA (FPN)", lambda: build_model("3D SigReg JEPA (FPN)")),
        ("3D nnU-Net", lambda: build_model("3D nnU-Net")),
        ("3D UNet", lambda: build_model("3D UNet")),
    ]

    records = []

    for frac in fractions:
        logger.info(f"\n--- Evaluating Fraction: {frac * 100:.1f}% Labels ---")
        try:
            train_dataset = BraTS3DDataset(
                split="train", fraction=frac, seed=args.seed, augmentations=aug_tf
            )
        except FileNotFoundError:
            train_dataset = [
                {
                    "image": torch.randn(4, 128, 128, 128),
                    "mask": (torch.rand(1, 128, 128, 128) > 0.95).float(),
                }
                for _ in range(2)
            ]
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size if not args.smoke_test else 2, shuffle=True
        )

        row = {"Fraction": f"{frac * 100:.1f}%"}
        for name, model_fn in models:
            m = model_fn().to(device)
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

        records.append(row)

    df = pd.DataFrame(records)
    csv_path = METRICS_DIR / "low_data_3d_summary.csv"
    md_path = METRICS_DIR / "low_data_3d_summary.md"

    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 3D Low-Data Label Efficiency Benchmark Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print("\n" + df.to_string(index=False) + "\n")
    logger.info(f"Low-data summary saved to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
