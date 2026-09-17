#!/usr/bin/env python3
r"""
3D Out-of-Distribution (OOD) Scanner Shift & Missing Modality Benchmark.

Theoretical Formulations:
-------------------------
1. 3D Rician Scanner Noise (Gudbjartsson & Patz, MRM 1995):
   Simulates MRI magnitude reconstruction from two orthogonal quadrature radiofrequency coil channels:
       M = \sqrt{(X + \eta_1)^2 + \eta_2^2}, \quad \eta_1, \eta_2 \sim \mathcal{N}(0, \sigma^2)
2. 3D B1 RF Inhomogeneity Bias Field (Sled et al., IEEE TMI 1998; Lebrun et al., 2021):
   Multiplicative smooth low-frequency polynomial spatial bias:
       X_{\text{corrupt}} = X \cdot B(z, y, x)
3. Missing Modality Zero-Padding:
   Evaluates resilience when only contrast-enhanced T1c or T2-FLAIR is accessible.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from brats_jepa_3d.config import CHECKPOINTS_DIR, METRICS_DIR, ensure_directories
from brats_jepa_3d.data import (
    BraTS3DDataset,
    apply_b1_bias_field_3d,
    apply_rician_noise_3d,
)
from brats_jepa_3d.metrics import compute_volumetric_metrics_3d
from brats_jepa_3d.models import BraTS3DnnUNet, BraTS3DUNet, JEPASegmentationModel3D
from brats_jepa_3d.utils import get_autocast_context, get_device, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="3D OOD Scanner Shift Benchmark")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", "--no-amp", action="store_false", dest="amp")
    parser.add_argument("--smoke_test", action="store_true", help="Run fast verification")
    return parser.parse_args()


def evaluate_perturbation(
    model, loader, device, perturb_fn, amp: bool = True, smoke_test: bool = False
) -> float:
    model.eval()
    dices = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if perturb_fn is not None:
                images = perturb_fn(images)

            with get_autocast_context(device, enabled=amp):
                out = model(images)
                logits = out[0] if isinstance(out, (list, tuple)) else out

            metrics = compute_volumetric_metrics_3d(logits, masks)
            dices.extend(metrics["dice_per_sample"])
            if smoke_test and batch_idx >= 1:
                break
    return float(np.mean(dices)) if dices else 0.0


def main():
    args = parse_args()
    ensure_directories()
    set_seed(args.seed)
    device = get_device()
    logger = setup_logger("evaluate_ood_3d", METRICS_DIR / "ood_3d.log")
    logger.info("Starting 3D Out-of-Distribution Scanner Shift Benchmark...")

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
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    def get_model(name: str):
        if name == "3D VisReg JEPA (FPN)":
            m = JEPASegmentationModel3D(decoder_type="multiscale")
            ms_candidates = sorted(CHECKPOINTS_DIR.glob("visreg_jepa*multiscale*best.pt"))
            candidates = ms_candidates if ms_candidates else sorted(CHECKPOINTS_DIR.glob("visreg_jepa*best.pt"))
            ckpt = candidates[-1] if candidates else None
            if ckpt and ckpt.exists():
                sd = torch.load(ckpt, map_location=device)
                sd = sd.get("model_state_dict", sd)
                missing, _unexpected = m.load_state_dict(sd, strict=False)
                matched = [k for k in m.state_dict() if k not in missing]
                logger.info(f"Loaded {name} weights from {ckpt.name} ({len(matched)} matched keys)")
            else:
                logger.warning(
                    f"Fine-tuned checkpoint for {name} not found. Evaluating randomly initialized weights."
                )
            return m.to(device)

        elif name == "3D SigReg JEPA (FPN)":
            m = JEPASegmentationModel3D(decoder_type="multiscale")
            ms_candidates = sorted(CHECKPOINTS_DIR.glob("sigreg_jepa*multiscale*best.pt"))
            candidates = ms_candidates if ms_candidates else sorted(CHECKPOINTS_DIR.glob("sigreg_jepa*best.pt"))
            ckpt = candidates[-1] if candidates else None
            if ckpt and ckpt.exists():
                sd = torch.load(ckpt, map_location=device)
                sd = sd.get("model_state_dict", sd)
                missing, _unexpected = m.load_state_dict(sd, strict=False)
                matched = [k for k in m.state_dict() if k not in missing]
                logger.info(f"Loaded {name} weights from {ckpt.name} ({len(matched)} matched keys)")
            else:
                logger.warning(
                    f"Fine-tuned checkpoint for {name} not found. Evaluating randomly initialized weights."
                )
            return m.to(device)

        elif name == "3D nnU-Net":
            m = BraTS3DnnUNet(deep_supervision=True)
            ckpt = CHECKPOINTS_DIR / "nnunet_3d_best.pt"
            if ckpt.exists():
                sd = torch.load(ckpt, map_location=device)
                sd = sd.get("model_state_dict", sd)
                missing, _unexpected = m.load_state_dict(sd, strict=False)
                matched = [k for k in m.state_dict() if k not in missing]
                logger.info(f"Loaded {name} weights from {ckpt.name} ({len(matched)} matched keys)")
            else:
                logger.warning(
                    f"Checkpoint for {name} not found at {ckpt}. Evaluating initialized weights."
                )
            return m.to(device)

        elif name == "3D UNet":
            m = BraTS3DUNet()
            ckpt = CHECKPOINTS_DIR / "unet_3d_best.pt"
            if ckpt.exists():
                sd = torch.load(ckpt, map_location=device)
                sd = sd.get("model_state_dict", sd)
                missing, _unexpected = m.load_state_dict(sd, strict=False)
                matched = [k for k in m.state_dict() if k not in missing]
                logger.info(f"Loaded {name} weights from {ckpt.name} ({len(matched)} matched keys)")
            else:
                logger.warning(
                    f"Checkpoint for {name} not found at {ckpt}. Evaluating initialized weights."
                )
            return m.to(device)
        else:
            raise ValueError(f"Unknown model name: {name}")

    model_names = [
        "3D VisReg JEPA (FPN)",
        "3D SigReg JEPA (FPN)",
        "3D nnU-Net",
        "3D UNet",
    ]
    instantiated_models = [(m_name, get_model(m_name)) for m_name in model_names]

    perturbations = [
        ("Clean Baseline", None),
        ("Rician Noise (sigma=0.08)", lambda img: apply_rician_noise_3d(img, sigma=0.08)),
        ("B1 Bias Field Inhomogeneity", lambda img: apply_b1_bias_field_3d(img, strength=0.35)),
        (
            "Missing Modalities: T1c Only",
            lambda img: (
                torch.cat(
                    [torch.zeros_like(img[:, :1]), img[:, 1:2], torch.zeros_like(img[:, 2:])], dim=1
                )
                if img.dim() == 5
                else torch.cat(
                    [torch.zeros_like(img[:1]), img[1:2], torch.zeros_like(img[2:])], dim=0
                )
            ),
        ),
        (
            "Missing Modalities: FLAIR Only",
            lambda img: (
                torch.cat([torch.zeros_like(img[:, :3]), img[:, 3:4]], dim=1)
                if img.dim() == 5
                else torch.cat([torch.zeros_like(img[:3]), img[3:4]], dim=0)
            ),
        ),
    ]

    records = []

    for p_name, p_fn in perturbations:
        logger.info(f"\n--- Evaluating Regime: {p_name} ---")
        row = {"Regime": p_name}
        for m_name, m in instantiated_models:
            dice = evaluate_perturbation(
                m, test_loader, device, p_fn, amp=args.amp, smoke_test=args.smoke_test
            )
            logger.info(f"{m_name} -> Dice: {dice * 100:.2f}%")
            row[m_name] = f"{dice * 100:.2f}%"
        records.append(row)

    df = pd.DataFrame(records)
    csv_path = METRICS_DIR / "ood_3d_summary.csv"
    md_path = METRICS_DIR / "ood_3d_summary.md"

    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 3D Out-of-Distribution & Missing Modality Robustness Benchmark\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print("\n" + df.to_string(index=False) + "\n")
    logger.info(f"OOD summary saved to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
