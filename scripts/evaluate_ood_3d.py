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

from brats_jepa_3d.config import (
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    METRICS_DIR,
    ensure_directories,
    load_yaml_config,
)
from brats_jepa_3d.data import (
    BraTS3DDataset,
    apply_b1_bias_field_3d,
    apply_rician_noise_3d,
)
from brats_jepa_3d.metrics import compute_volumetric_metrics_3d
from brats_jepa_3d.models import (
    BraTS3DnnUNet,
    BraTS3DUNet,
    JEPASegmentationModel3D,
)
from brats_jepa_3d.utils import (
    get_autocast_context,
    get_device,
    set_seed,
    setup_logger,
    sort_checkpoints_by_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="3D OOD Scanner Shift Benchmark")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", "--no-amp", action="store_false", dest="amp")
    parser.add_argument(
        "--model_type",
        type=str,
        default="all",
        help="Specific model to evaluate (visreg_jepa, sigreg_jepa, ijepa, unet_3d, nnunet_3d, or all)",
    )
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
            brain_mask = batch.get("brain_mask")
            if brain_mask is not None:
                brain_mask = brain_mask.to(device)
            if perturb_fn is not None:
                images = perturb_fn(images, brain_mask)

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
        if not args.smoke_test:
            raise
        logger.warning("BraTS3D test split not found. Using synthetic dataset for smoke test.")
        test_dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": (torch.rand(1, 128, 128, 128) > 0.95).float(),
                "brain_mask": (torch.rand(1, 128, 128, 128) > 0.5).float(),
            }
            for _ in range(2)
        ]
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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

    def get_model(name: str):
        jepa_prefixes = {
            "3D VisReg JEPA (FPN)": "visreg_jepa",
            "3D SigReg JEPA (FPN)": "sigreg_jepa",
            "3D I-JEPA (FPN)": "ijepa",
        }
        if name in jepa_prefixes:
            prefix = jepa_prefixes[name]
            jepa_cfg_path = CONFIGS_DIR / "model" / f"{prefix}_3d.yaml"
            jepa_cfg = load_yaml_config(jepa_cfg_path) if jepa_cfg_path.exists() else {}

            ms_candidates = sorted(CHECKPOINTS_DIR.glob(f"{prefix}*multiscale*best.pt"))
            candidates = ms_candidates if ms_candidates else sorted(CHECKPOINTS_DIR.glob(f"{prefix}*best.pt"))
            ckpt = candidates[-1] if candidates else None

            ds_flag = False
            if ckpt and ckpt.exists():
                try:
                    sd_peek = torch.load(ckpt, map_location="cpu")
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
            if ckpt and ckpt.exists():
                sd = torch.load(ckpt, map_location=device)
                sd = sd.get("model_state_dict", sd)
                has_downstream = any(k.startswith("encoder.") for k in sd) or any(k.startswith("decoder.") for k in sd)
                has_pretrain = any(k.startswith("context_encoder.") for k in sd)
                if has_downstream:
                    missing, _unexpected = m.load_state_dict(sd, strict=False)
                    matched = [k for k in m.state_dict() if k not in missing]
                    logger.info(f"Loaded {name} weights from {ckpt.name} ({len(matched)} matched keys)")
                elif has_pretrain:
                    res = m.load_pretrained_encoder(sd)
                    logger.warning(
                        f"Loaded pre-trained encoder weights for {name} from {ckpt.name} ({res['loaded_keys']} keys), "
                        f"but decoder is randomly initialized."
                    )
                else:
                    missing, _unexpected = m.load_state_dict(sd, strict=False)
                    matched = [k for k in m.state_dict() if k not in missing]
                    logger.info(f"Loaded {name} weights from {ckpt.name} ({len(matched)} matched keys)")
            else:
                pretrain_ckpts = sort_checkpoints_by_epoch(
                    list(CHECKPOINTS_DIR.glob(f"{prefix}*epoch*.pt"))
                )
                if pretrain_ckpts:
                    ckpt = pretrain_ckpts[-1]
                    sd = torch.load(ckpt, map_location=device)
                    sd = sd.get("model_state_dict", sd)
                    res = m.load_pretrained_encoder(sd)
                    logger.warning(
                        f"Loaded pre-trained encoder weights for {name} from {ckpt.name} ({res['loaded_keys']} keys), "
                        f"but decoder is randomly initialized."
                    )
                else:
                    logger.warning(
                        f"Checkpoint for {name} not found. Skipping OOD evaluation."
                    )
                    return None
            return m.to(device)

        elif name == "3D nnU-Net":
            nnunet_cfg_path = CONFIGS_DIR / "model" / "nnunet_3d.yaml"
            nnunet_cfg = load_yaml_config(nnunet_cfg_path) if nnunet_cfg_path.exists() else {}
            m = BraTS3DnnUNet(
                in_channels=nnunet_cfg.get("in_channels", 4),
                out_channels=nnunet_cfg.get("out_channels", 1),
                deep_supervision=nnunet_cfg.get("deep_supervision", True),
                deep_supr_num=nnunet_cfg.get("deep_supr_num", 3),
                res_block=nnunet_cfg.get("res_block", True),
            )
            ckpt = CHECKPOINTS_DIR / "nnunet_3d_best.pt"
            if ckpt.exists():
                sd = torch.load(ckpt, map_location=device)
                sd = sd.get("model_state_dict", sd)
                missing, _unexpected = m.load_state_dict(sd, strict=False)
                matched = [k for k in m.state_dict() if k not in missing]
                logger.info(f"Loaded {name} weights from {ckpt.name} ({len(matched)} matched keys)")
            else:
                logger.warning(
                    f"Checkpoint for {name} not found at {ckpt}. Skipping OOD evaluation."
                )
                return None
            return m.to(device)

        elif name == "3D UNet":
            unet_cfg_path = CONFIGS_DIR / "model" / "unet_3d.yaml"
            unet_cfg = load_yaml_config(unet_cfg_path) if unet_cfg_path.exists() else {}
            m = BraTS3DUNet(
                in_channels=unet_cfg.get("in_channels", 4),
                out_channels=unet_cfg.get("out_channels", 1),
                channels=tuple(unet_cfg.get("channels", (32, 64, 128, 256, 512))),
                strides=tuple(unet_cfg.get("strides", (2, 2, 2, 2))),
                num_res_units=unet_cfg.get("num_res_units", 2),
                dropout=unet_cfg.get("dropout", 0.1),
            )
            ckpt = CHECKPOINTS_DIR / "unet_3d_best.pt"
            if ckpt.exists():
                sd = torch.load(ckpt, map_location=device)
                sd = sd.get("model_state_dict", sd)
                missing, _unexpected = m.load_state_dict(sd, strict=False)
                matched = [k for k in m.state_dict() if k not in missing]
                logger.info(f"Loaded {name} weights from {ckpt.name} ({len(matched)} matched keys)")
            else:
                logger.warning(
                    f"Checkpoint for {name} not found at {ckpt}. Skipping OOD evaluation."
                )
                return None
            return m.to(device)
        else:
            raise ValueError(f"Unknown model name: {name}")

    instantiated_models = [(m_name, get_model(m_name)) for m_name, _ in selected_models]

    perturbations = [
        ("Clean Baseline", None),
        (
            "Rician Noise (sigma=0.08)",
            lambda img, bm: apply_rician_noise_3d(img, sigma=0.08, brain_mask=bm),
        ),
        (
            "B1 Bias Field Inhomogeneity",
            lambda img, bm: apply_b1_bias_field_3d(img, strength=0.35, brain_mask=bm),
        ),
        (
            "Missing Modalities: T1c Only",
            lambda img, bm: img * torch.tensor([0.0, 1.0, 0.0, 0.0], device=img.device).view(1, 4, 1, 1, 1)
            if img.dim() == 5
            else img * torch.tensor([0.0, 1.0, 0.0, 0.0], device=img.device).view(4, 1, 1, 1),
        ),
        (
            "Missing Modalities: FLAIR Only",
            lambda img, bm: img * torch.tensor([0.0, 0.0, 0.0, 1.0], device=img.device).view(1, 4, 1, 1, 1)
            if img.dim() == 5
            else img * torch.tensor([0.0, 0.0, 0.0, 1.0], device=img.device).view(4, 1, 1, 1),
        ),
    ]

    records = []

    for p_name, p_fn in perturbations:
        logger.info(f"\n--- Evaluating Regime: {p_name} ---")
        row = {"Regime": p_name}
        for m_name, m in instantiated_models:
            if m is None:
                row[m_name] = "N/A"
                continue
            dice = evaluate_perturbation(
                m, test_loader, device, p_fn, amp=args.amp, smoke_test=args.smoke_test
            )
            logger.info(f"{m_name} -> Dice: {dice * 100:.2f}%")
            row[m_name] = f"{dice * 100:.2f}%"
        records.append(row)

    new_df = pd.DataFrame(records)
    csv_path = METRICS_DIR / "ood_3d_summary.csv"
    md_path = METRICS_DIR / "ood_3d_summary.md"

    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            existing_df["Regime"] = existing_df["Regime"].astype(str)
            new_df["Regime"] = new_df["Regime"].astype(str)
            for col in new_df.columns:
                if col == "Regime":
                    continue
                new_vals = new_df.set_index("Regime")[col]
                if not (new_vals == "N/A").all():
                    if col not in existing_df.columns:
                        existing_df[col] = "N/A"
                    for r_val, val in new_vals.items():
                        if val != "N/A":
                            existing_df.loc[existing_df["Regime"] == r_val, col] = val
            df = existing_df
        except Exception as e:
            logger.warning(f"Could not merge with existing CSV: {e}. Writing new CSV.")
            df = new_df
    else:
        df = new_df

    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 3D Out-of-Distribution & Missing Modality Robustness Benchmark\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print("\n" + df.to_string(index=False) + "\n")
    logger.info(f"OOD summary saved to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
