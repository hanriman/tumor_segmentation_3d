#!/usr/bin/env python3
"""
3D Master Volumetric Benchmark Evaluator.
Evaluates all models on the BraTS 2024 GLI test split across:
- Volumetric 3D Dice, IoU, cKDTree 3D HD95 (Powers, 2011; Taha & Hanbury, 2015)
- Latent Space Collapse Metrics: Effective Rank (S^2), Centered Cosine Sim (Roy & Vetterli, 2007)
- Inference Latency (ms / 3D volume)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from brats_jepa_3d.config import (
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    METRICS_DIR,
    ensure_directories,
    load_yaml_config,
)
from brats_jepa_3d.data import BraTS3DDataset
from brats_jepa_3d.metrics import (
    compute_representation_collapse_metrics,
    compute_volumetric_metrics_3d,
)
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
    parser = argparse.ArgumentParser(description="Master 3D Benchmark Evaluation")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", "--no-amp", action="store_false", dest="amp")
    parser.add_argument("--smoke_test", action="store_true", help="Run fast verification")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Specific model to evaluate (visreg_jepa, sigreg_jepa, ijepa, unet, nnunet, or all)",
    )
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="multiscale",
        choices=["multiscale", "bottleneck"],
        help="Decoder type for JEPA models",
    )
    parser.add_argument(
        "--all_models",
        action="store_true",
        default=False,
        help="Evaluate all 5 benchmark models",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to an explicit checkpoint file (.pt)",
    )
    return parser.parse_args()


def benchmark_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    smoke_test: bool = False,
) -> dict[str, float]:
    model.eval()
    dices, ious, hd95s, latencies = [], [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with get_autocast_context(device, enabled=amp):
                out = model(images)
                logits = out[0] if isinstance(out, (list, tuple)) else out

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms per volume

            metrics = compute_volumetric_metrics_3d(logits, masks)
            dices.extend(metrics["dice_per_sample"])
            ious.extend(metrics["iou_per_sample"])
            hd95s.extend(metrics["hd95_per_sample"])

            if smoke_test and batch_idx >= 1:
                break

    return {
        "dice_mean": float(np.mean(dices)) if dices else 0.0,
        "dice_std": float(np.std(dices)) if dices else 0.0,
        "iou_mean": float(np.mean(ious)) if ious else 0.0,
        "iou_std": float(np.std(ious)) if ious else 0.0,
        "hd95_mean": float(np.mean(hd95s)) if hd95s else 0.0,
        "hd95_std": float(np.std(hd95s)) if hd95s else 0.0,
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


def evaluate_representation(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    smoke_test: bool = False,
) -> dict[str, float]:
    encoder.eval()
    all_tokens = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            tokens = encoder(images)  # [B, 512, D]
            all_tokens.append(tokens.cpu())
            if smoke_test and batch_idx >= 1:
                break

    if not all_tokens:
        return {"erank": 1.0, "centered_cossim": 0.0}

    cat_tokens = torch.cat(all_tokens, dim=0)  # [N_total, 512, D]
    flat_tokens = cat_tokens.reshape(-1, cat_tokens.shape[-1])  # [N*512, D]

    collapse_metrics = compute_representation_collapse_metrics(flat_tokens)
    return {
        "erank": collapse_metrics["effective_rank"],
        "centered_cossim": collapse_metrics["avg_cosine_sim_centered"],
    }


def main():
    args = parse_args()
    ensure_directories()
    set_seed(args.seed)
    device = get_device()
    logger = setup_logger("evaluate_3d", METRICS_DIR / "benchmark_3d.log")
    logger.info("Initializing Master 3D Volumetric Benchmark...")

    try:
        test_dataset = BraTS3DDataset(split="test", augmentations=None)
    except FileNotFoundError:
        logger.warning(
            "Processed dataset not found. Generating synthetic test dataset for verification."
        )
        test_dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": (torch.rand(1, 128, 128, 128) > 0.95).float(),
                "patient_id": f"test_{i}",
            }
            for i in range(2)
        ]

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    all_models_list = [
        ("3D SigReg JEPA (FPN)", "sigreg_jepa", args.decoder_type or "multiscale"),
        ("3D VisReg JEPA (FPN)", "visreg_jepa", args.decoder_type or "multiscale"),
        ("3D I-JEPA (FPN)", "ijepa", args.decoder_type or "multiscale"),
        ("3D Residual UNet", "unet_3d", None),
        ("3D nnU-Net (DynUNet)", "nnunet_3d", None),
    ]

    # Filter if user specified a specific model_type
    req_model = args.model_type.lower() if args.model_type else None
    if req_model and not args.all_models and req_model not in ("all", "all_models"):
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
        }
        canonical = alias_map.get(req_model, req_model)
        models_to_evaluate = [m for m in all_models_list if m[1] == canonical]
        if not models_to_evaluate:
            logger.warning(f"Unknown model_type '{req_model}'. Defaulting to all models.")
            models_to_evaluate = all_models_list
    else:
        models_to_evaluate = all_models_list

    results = []

    for label, model_type, decoder_type in models_to_evaluate:
        logger.info(f"Evaluating: {label}")

        if args.checkpoint and len(models_to_evaluate) == 1:
            ckpt_file = Path(args.checkpoint)
        else:
            if args.checkpoint and len(models_to_evaluate) > 1:
                logger.warning(
                    f"--checkpoint was specified ({args.checkpoint}), but multiple models are being evaluated. "
                    f"Using default checkpoint path for {label}."
                )
            if model_type == "unet_3d":
                ckpt_file = CHECKPOINTS_DIR / "unet_3d_best.pt"
            elif model_type == "nnunet_3d":
                ckpt_file = CHECKPOINTS_DIR / "nnunet_3d_best.pt"
            else:
                downstream_ckpts = sorted(CHECKPOINTS_DIR.glob(f"{model_type}_{decoder_type}_best.pt")) or sorted(
                    CHECKPOINTS_DIR.glob(f"{model_type}*best.pt")
                )
                pretrain_ckpts = sort_checkpoints_by_epoch(
                    list(CHECKPOINTS_DIR.glob(f"{model_type}*epoch*.pt"))
                )
                if downstream_ckpts:
                    ckpt_file = downstream_ckpts[-1]
                elif pretrain_ckpts:
                    ckpt_file = pretrain_ckpts[-1]
                else:
                    ckpt_file = CHECKPOINTS_DIR / f"{model_type}_{decoder_type}_best.pt"

        if model_type == "unet_3d":
            unet_cfg_path = CONFIGS_DIR / "model" / "unet_3d.yaml"
            unet_cfg = load_yaml_config(unet_cfg_path) if unet_cfg_path.exists() else {}
            model = BraTS3DUNet(
                in_channels=unet_cfg.get("in_channels", 4),
                out_channels=unet_cfg.get("out_channels", 1),
                channels=tuple(unet_cfg.get("channels", (32, 64, 128, 256, 512))),
                strides=tuple(unet_cfg.get("strides", (2, 2, 2, 2))),
                num_res_units=unet_cfg.get("num_res_units", 2),
                dropout=unet_cfg.get("dropout", 0.1),
            ).to(device)
            if ckpt_file.exists():
                sd = torch.load(ckpt_file, map_location=device)
                sd = sd.get("model_state_dict", sd)
                missing, _unexpected = model.load_state_dict(sd, strict=False)
                matched = [k for k in model.state_dict() if k not in missing]
                logger.info(f"Loaded {label} weights from {ckpt_file.name} ({len(matched)} matched keys)")
            else:
                logger.warning(
                    f"Checkpoint for {label} not found at {ckpt_file}. Evaluating initialized weights."
                )
            erank, cossim = "-", "-"

        elif model_type == "nnunet_3d":
            nnunet_cfg_path = CONFIGS_DIR / "model" / "nnunet_3d.yaml"
            nnunet_cfg = load_yaml_config(nnunet_cfg_path) if nnunet_cfg_path.exists() else {}
            model = BraTS3DnnUNet(
                in_channels=nnunet_cfg.get("in_channels", 4),
                out_channels=nnunet_cfg.get("out_channels", 1),
                deep_supervision=nnunet_cfg.get("deep_supervision", True),
                deep_supr_num=nnunet_cfg.get("deep_supr_num", 3),
                res_block=nnunet_cfg.get("res_block", True),
            ).to(device)
            if ckpt_file.exists():
                sd = torch.load(ckpt_file, map_location=device)
                sd = sd.get("model_state_dict", sd)
                missing, _unexpected = model.load_state_dict(sd, strict=False)
                matched = [k for k in model.state_dict() if k not in missing]
                logger.info(f"Loaded {label} weights from {ckpt_file.name} ({len(matched)} matched keys)")
            else:
                logger.warning(
                    f"Checkpoint for {label} not found at {ckpt_file}. Evaluating initialized weights."
                )
            erank, cossim = "-", "-"

        else:
            # JEPA Downstream Model
            jepa_cfg_path = CONFIGS_DIR / "model" / f"{model_type}_3d.yaml"
            jepa_cfg = load_yaml_config(jepa_cfg_path) if jepa_cfg_path.exists() else {}

            # Detect whether checkpoint has deep supervision heads
            ds_flag = getattr(args, "deep_supervision", False)
            if ckpt_file and ckpt_file.exists():
                try:
                    sd_peek = torch.load(ckpt_file, map_location="cpu")
                    sd_peek = sd_peek.get("model_state_dict", sd_peek)
                    if any(k.startswith("decoder.ds") for k in sd_peek):
                        ds_flag = True
                except (KeyError, OSError, RuntimeError, AttributeError):
                    pass

            model = JEPASegmentationModel3D(
                img_size=tuple(jepa_cfg.get("spatial_shape", (128, 128, 128))),
                patch_size=tuple(jepa_cfg.get("patch_size", (16, 16, 16))),
                in_channels=jepa_cfg.get("in_channels", 4),
                embed_dim=jepa_cfg.get("embed_dim", 384),
                encoder_depth=jepa_cfg.get("encoder_depth", 8),
                num_heads=jepa_cfg.get("num_heads", 6),
                mlp_ratio=jepa_cfg.get("mlp_ratio", 4.0),
                out_channels=1,
                decoder_type=decoder_type or "multiscale",
                deep_supervision=ds_flag,
            ).to(device)
            if ckpt_file and ckpt_file.exists():
                sd = torch.load(ckpt_file, map_location=device)
                sd = sd.get("model_state_dict", sd)
                has_downstream = any(k.startswith("encoder.") for k in sd) or any(k.startswith("decoder.") for k in sd)
                has_pretrain = any(k.startswith("context_encoder.") for k in sd)
                if has_downstream:
                    missing, _unexpected = model.load_state_dict(sd, strict=False)
                    matched = [k for k in model.state_dict() if k not in missing]
                    logger.info(f"Loaded downstream fine-tuned weights from {ckpt_file.name} ({len(matched)} matched keys)")
                elif has_pretrain:
                    res = model.load_pretrained_encoder(sd)
                    logger.warning(
                        f"Loaded pre-trained encoder weights from {ckpt_file.name} ({res['loaded_keys']} keys), "
                        f"but decoder is randomly initialized."
                    )
                else:
                    missing, _unexpected = model.load_state_dict(sd, strict=False)
                    matched = [k for k in model.state_dict() if k not in missing]
                    logger.info(f"Loaded weights from {ckpt_file.name} ({len(matched)} matched keys)")
            else:
                logger.warning(
                    f"Checkpoint for {label} not found at {ckpt_file}. Evaluating initialized weights."
                )

            # Evaluate representation diagnostics
            rep_stats = evaluate_representation(
                model.encoder, test_loader, device, smoke_test=args.smoke_test
            )
            erank_val = rep_stats["erank"]
            embed_dim = getattr(model.encoder, "embed_dim", 384)
            erank = f"{erank_val:.2f} ({erank_val / embed_dim * 100:.1f}%)"
            cossim = f"{rep_stats['centered_cossim']:.4f}"

        seg_stats = benchmark_model(
            model, test_loader, device, amp=args.amp, smoke_test=args.smoke_test
        )

        results.append(
            {
                "Model": label,
                "Dice (%)": f"{seg_stats['dice_mean'] * 100:.2f} ± {seg_stats['dice_std'] * 100:.2f}",
                "IoU (%)": f"{seg_stats['iou_mean'] * 100:.2f} ± {seg_stats['iou_std'] * 100:.2f}",
                "HD95 (mm)": f"{seg_stats['hd95_mean']:.2f} ± {seg_stats['hd95_std']:.2f}",
                "Latency (ms)": f"{seg_stats['latency_ms']:.2f}",
                "EffRank (S²)": erank,
                "Centered CosSim": cossim,
            }
        )

    df_new = pd.DataFrame(results)
    csv_path = METRICS_DIR / "benchmark_3d_summary.csv"
    master_csv_path = METRICS_DIR / "master_3d_benchmark.csv"
    md_path = METRICS_DIR / "benchmark_3d_summary.md"
    master_md_path = METRICS_DIR / "master_3d_benchmark.md"

    if csv_path.exists() and len(models_to_evaluate) < len(all_models_list):
        try:
            df_old = pd.read_csv(csv_path)
            for _, row in df_new.iterrows():
                mask = df_old["Model"] == row["Model"]
                if mask.any():
                    for col in df_new.columns:
                        df_old.loc[mask, col] = row[col]
                else:
                    df_old = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
            df = df_old
        except (ValueError, KeyError, OSError, pd.errors.EmptyDataError):
            df = df_new
    else:
        df = df_new

    df = df.dropna(subset=["Model"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    df.to_csv(master_csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 3D Volumetric BraTS 2024 GLI Master Benchmark Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    with open(master_md_path, "w", encoding="utf-8") as f:
        f.write("# 3D Volumetric BraTS 2024 GLI Master Benchmark Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print("\n" + df.to_string(index=False) + "\n")
    logger.info(f"Summary saved to {csv_path} and {master_csv_path}")


if __name__ == "__main__":
    main()
