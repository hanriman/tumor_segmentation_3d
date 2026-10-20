#!/usr/bin/env python3
"""
3D Low-Data Label Efficiency Benchmark.
Evaluates model performance under extreme label scarcity (1% to 100% volumetric labels).
"""

import argparse
import gc
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from brats_jepa_3d.config import (
    OUTPUTS_DIR,
    ensure_directories,
    resolve_default_output_dir,
)
from brats_jepa_3d.data import BraTS3DDataset, VolumetricAugmentations3D
from brats_jepa_3d.losses import build_segmentation_criterion, resolve_seg_loss_type
from brats_jepa_3d.metrics import compute_brats_regions_3d
from brats_jepa_3d.models import build_segmentation_model_for_training, resolve_model_alias
from brats_jepa_3d.training import build_warmup_cosine_scheduler, evaluate_validation_split
from brats_jepa_3d.utils import (
    get_autocast_context,
    get_device,
    set_seed,
    setup_logger,
    tta_predict_logits,
)


def parse_args():
    parser = argparse.ArgumentParser(description="3D Label Efficiency Benchmark")
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--min_steps",
        type=int,
        default=0,
        help="Minimum gradient update steps per low-data fraction (default: 0, strictly preserving --epochs budget; set >0 for step-floor training)",
    )
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
        help="Specific model to evaluate (visreg_jepa, sigreg_jepa, unet_3d, nnunet_3d, vit_scratch, or all)",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        default=False,
        help="Evaluate 3D ViT trained from random initialization (no pre-trained checkpoint)",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=False,
        help="Evaluate 3D ViT model trained with a frozen encoder (probe on low data)",
    )
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="multiscale",
        choices=["multiscale", "unetr_hybrid"],
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
        "--tta",
        action="store_true",
        default=False,
        help="4-fold orthogonal-reflection test-time augmentation on test eval",
    )
    parser.add_argument("--include_background", action="store_true", default=None)
    parser.add_argument("--no_include_background", action="store_false", dest="include_background")
    parser.add_argument(
        "--deep_supervision",
        action="store_true",
        default=True,
        help="Train with deep supervision multi-scale loss heads (default: True)",
    )
    parser.add_argument(
        "--no_deep_supervision",
        action="store_false",
        dest="deep_supervision",
        help="Disable deep supervision heads",
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
        help="Output directory for checkpoints and metrics",
    )
    parser.add_argument("--smoke_test", action="store_true", help="Run fast verification")
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default=None,
        help="Explicit path to pre-trained SSL encoder checkpoint (e.g. visreg_jepa_3d_best.pt)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs="+",
        default=None,
        help="Voxel patch size (e.g. 8 8 8 or 16 16 16, default: from model config)",
    )
    return parser.parse_args()


def train_and_eval(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int,
    amp: bool = True,
    smoke_test: bool = False,
    loss_type: str = "dice_bce",
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    num_classes: int | None = None,
    tta: bool = False,
    include_background: bool | None = None,
    val_loader: DataLoader | None = None,
    min_steps: int = 0,
) -> dict[str, float]:
    """Trains on the fraction loader (optionally enforcing a minimum gradient step floor via `min_steps`),
    selects the best checkpoint on `val_loader` (if provided), and scores `test_loader`
    across canonical BraTS regions.
    Returns per-region {"WT", "TC", "ET", "mean"} macro Dice.
    """
    use_deep_supervision = getattr(model, "deep_supervision", False)
    criterion = build_segmentation_criterion(
        loss_type,
        use_deep_supervision,
        tversky_alpha,
        tversky_beta,
        num_classes=num_classes,
        include_background=include_background,
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4)

    steps_per_epoch = max(1, len(train_loader))
    if smoke_test:
        effective_epochs = 1
    elif min_steps > 0:
        effective_epochs = max(epochs, math.ceil(min_steps / steps_per_epoch))
    else:
        effective_epochs = epochs
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs=effective_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(
        device="cuda" if device.type == "cuda" else "cpu", enabled=amp and device.type == "cuda"
    )

    best_val_dice = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    val_interval = max(1, effective_epochs // 5)

    for epoch in range(1, effective_epochs + 1):
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
        if val_loader is not None and (
            epoch % val_interval == 0 or epoch == effective_epochs or smoke_test
        ):
            val_dice = evaluate_validation_split(
                model, val_loader, device, amp=amp, smoke_test=smoke_test
            )["val_dice"]
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if smoke_test:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Evaluate on test set across canonical BraTS regions (WT, TC, ET)
    model.eval()
    region_dices = {"WT": [], "TC": [], "ET": []}
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            with get_autocast_context(device, enabled=amp):
                if tta:
                    logits = tta_predict_logits(model, images)
                else:
                    out = model(images)
                    logits = out[0] if isinstance(out, (list, tuple)) else out
            res = compute_brats_regions_3d(logits, masks, compute_hd95=False)
            for rname in ("WT", "TC", "ET"):
                region_dices[rname].extend(res[rname]["dice_per_sample"])
            if smoke_test and batch_idx >= 1:
                break

    out_scores = {r: float(np.nanmean(v)) if v else 0.0 for r, v in region_dices.items()}
    out_scores["mean"] = float(sum(out_scores[r] for r in ("WT", "TC", "ET")) / 3.0)
    return out_scores


def main():
    args = parse_args()
    if getattr(args, "from_scratch", False):
        exp_name = f"scratch_{args.decoder_type}_outputs"
    elif getattr(args, "freeze_encoder", False):
        exp_name = f"frozen_{args.model_type}_{args.decoder_type}_outputs"
    elif args.model_type in ["unet_3d", "nnunet_3d"]:
        exp_name = f"{args.model_type}_outputs"
    else:
        exp_name = f"{args.model_type}_{args.decoder_type}_outputs"

    out_dir = resolve_default_output_dir(
        args.output_dir,
        experiment_name=exp_name,
        is_smoke_test=getattr(args, "smoke_test", False),
    )
    checkpoints_dir = (out_dir / "checkpoints") if (out_dir / "checkpoints").exists() else out_dir
    metrics_dir = out_dir / "metrics"
    ensure_directories(out_dir)
    set_seed(args.seed)
    device = get_device()
    logger = setup_logger("low_data_3d", metrics_dir / "low_data_3d.log")
    logger.info("Starting 3D Low-Data Label Efficiency Benchmark...")

    fractions = [0.01, 0.10] if args.smoke_test else args.fractions
    epochs = 1 if args.smoke_test else args.epochs

    aug_tf = VolumetricAugmentations3D(is_training=True)

    resolved_data_dir = Path(args.data_dir).resolve() if getattr(args, "data_dir", None) else None
    try:
        val_dataset = BraTS3DDataset(split="val", data_dir=resolved_data_dir)
        test_dataset = BraTS3DDataset(split="test", data_dir=resolved_data_dir)
    except FileNotFoundError:
        if not args.smoke_test:
            raise
        logger.warning("BraTS3D val/test split not found. Using synthetic dataset for smoke test.")
        val_dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": torch.randint(0, 5, (1, 128, 128, 128), dtype=torch.float32),
            }
            for _ in range(2)
        ]
        test_dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": torch.randint(0, 5, (1, 128, 128, 128), dtype=torch.float32),
            }
            for _ in range(2)
        ]
    num_workers = getattr(args, "num_workers", 2)
    use_pin_memory = getattr(args, "pin_memory", True) and (device.type == "cuda")

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
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
        ("3D nnU-Net", "nnunet_3d"),
        ("3D UNet", "unet_3d"),
    ]

    req_model = resolve_model_alias(args.model_type)
    if args.from_scratch or req_model == "vit_scratch":
        scratch_label = (
            "3D ViT-UNETR (From Scratch)"
            if args.decoder_type == "unetr_hybrid"
            else "3D ViT-FPN (From Scratch)"
        )
        selected_models = [(scratch_label, "vit_scratch")]
    elif getattr(args, "freeze_encoder", False) or req_model == "vit_frozen":
        frozen_label = (
            "3D ViT-UNETR (Frozen)"
            if args.decoder_type == "unetr_hybrid"
            else "3D ViT-FPN (Frozen)"
        )
        selected_models = [(frozen_label, "vit_frozen")]
    elif req_model != "all":
        selected_models = [m for m in all_models if m[1] == req_model]
        if not selected_models:
            logger.warning(f"Unknown model_type '{args.model_type}'. Defaulting to all models.")
            selected_models = all_models
    else:
        selected_models = all_models

    records = []

    def _display_name(name: str) -> str:
        dec = getattr(args, "decoder_type", "multiscale")
        if dec == "unetr_hybrid":
            return name.replace("(FPN)", "(UNETR)").replace("ViT-FPN", "ViT-UNETR")
        if dec != "multiscale" and ("JEPA" in name or "ViT-FPN" in name):
            return f"{name} [{dec}]"
        return name

    for frac in fractions:
        logger.info(f"\n--- Evaluating Fraction: {frac * 100:.1f}% Labels ---")
        try:
            train_dataset = BraTS3DDataset(
                split="train", fraction=frac, seed=args.seed, augmentations=aug_tf, data_dir=resolved_data_dir
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
                    "mask": torch.randint(0, 5, (1, 128, 128, 128)).long(),
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
        for name, m_type in selected_models:
            is_scratch = m_type == "vit_scratch" or args.from_scratch
            is_frozen = m_type == "vit_frozen" or getattr(args, "freeze_encoder", False)
            built = build_segmentation_model_for_training(
                model_type=m_type,
                device=device,
                decoder_type=args.decoder_type,
                from_scratch=is_scratch,
                freeze_encoder=is_frozen,
                deep_supervision=args.deep_supervision,
                patch_size=getattr(args, "patch_size", None),
                checkpoints_dir=checkpoints_dir,
                pretrained_checkpoint=getattr(args, "pretrained_checkpoint", None),
                resolved_data_dir=resolved_data_dir,
                logger=logger,
                require_ssl_for_pretrain=not is_scratch,
                label=name,
            )
            if built is None:
                row[_display_name(name)] = "N/A"
                continue
            m, n_classes = built
            scores = train_and_eval(
                m,
                train_loader,
                test_loader,
                device,
                epochs=epochs,
                amp=args.amp,
                smoke_test=args.smoke_test,
                loss_type=resolve_seg_loss_type(args),
                tversky_alpha=getattr(args, "tversky_alpha", 0.3),
                tversky_beta=getattr(args, "tversky_beta", 0.7),
                num_classes=n_classes,
                tta=args.tta,
                include_background=getattr(args, "include_background", None),
                val_loader=val_loader,
                min_steps=getattr(args, "min_steps", 0),
            )
            dname = _display_name(name)
            logger.info(
                f"{name} ({frac * 100:.1f}% labels) -> "
                f"WT {scores['WT'] * 100:.2f}% / TC {scores['TC'] * 100:.2f}% / "
                f"ET {scores['ET'] * 100:.2f}% (mean {scores['mean'] * 100:.2f}%)"
            )
            row[dname] = f"{scores['mean'] * 100:.2f}%"
            for r in ("WT", "TC", "ET", "mean"):
                row[f"{dname} [{r}]"] = f"{scores[r] * 100:.2f}%"
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
    csv_path = metrics_dir / "low_data_3d_summary.csv"
    md_path = metrics_dir / "low_data_3d_summary.md"

    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            existing_df["Fraction"] = existing_df["Fraction"].astype(str).str.strip()
            new_df["Fraction"] = new_df["Fraction"].astype(str).str.strip()

            new_cols = [c for c in new_df.columns if c not in existing_df.columns]
            overlap_cols = [c for c in new_df.columns if c != "Fraction" and c in existing_df.columns]
            if new_cols:
                merged = pd.merge(existing_df, new_df[["Fraction"] + new_cols], on="Fraction", how="outer")
            else:
                merged = pd.merge(existing_df, new_df[["Fraction"]], on="Fraction", how="outer")
            if overlap_cols:
                new_indexed = new_df.set_index("Fraction")
                for col in overlap_cols:
                    for frac, val in new_indexed[col].items():
                        if pd.notna(val) and str(val) != "N/A":
                            merged.loc[merged["Fraction"] == frac, col] = val

            def frac_key(v: str) -> float:
                m = re.search(r"(\d+(?:\.\d+)?)", str(v))
                return float(m.group(1)) if m else 0.0

            merged["_sort"] = merged["Fraction"].apply(frac_key)
            df = merged.sort_values(by="_sort").drop(columns=["_sort"]).reset_index(drop=True).fillna("N/A")
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
