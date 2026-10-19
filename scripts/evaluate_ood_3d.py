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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from brats_jepa_3d.config import OUTPUTS_DIR, ensure_directories
from brats_jepa_3d.data import (
    BraTS3DDataset,
    apply_b1_bias_field_3d,
    apply_rician_noise_3d,
)
from brats_jepa_3d.metrics import compute_brats_regions_3d
from brats_jepa_3d.models import load_segmentation_model_for_eval, resolve_model_alias
from brats_jepa_3d.utils import (
    get_autocast_context,
    get_device,
    set_seed,
    setup_logger,
    tta_predict_logits,
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
        help="Evaluate 3D ViT model trained with a frozen encoder (probe)",
    )
    parser.add_argument("--smoke_test", action="store_true", help="Run fast verification")
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="multiscale",
        choices=["multiscale", "unetr_hybrid"],
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
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="4-fold orthogonal-reflection test-time augmentation",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs="+",
        default=None,
        help="Voxel patch size (e.g. 8 8 8 or 16 16 16, default: from model config)",
    )
    return parser.parse_args()


def evaluate_perturbation(
    model,
    loader,
    device,
    perturb_fn,
    amp: bool = True,
    smoke_test: bool = False,
    tta: bool = False,
) -> dict[str, float]:
    """Scores one perturbation regime.

    Returns per-region {"WT", "TC", "ET", "mean"} macro Dice so regime tables resolve
    the T1c/FLAIR asymmetry per region (e.g. ET preserved T1c-only).
    """
    model.eval()
    region_dices = {"WT": [], "TC": [], "ET": []}
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
    out_dir = Path(args.output_dir).resolve() if getattr(args, "output_dir", None) else OUTPUTS_DIR
    checkpoints_dir = (out_dir / "checkpoints") if (out_dir / "checkpoints").exists() else out_dir
    metrics_dir = out_dir / "metrics"
    ensure_directories(out_dir)
    set_seed(args.seed)
    device = get_device()
    logger = setup_logger("evaluate_ood_3d", metrics_dir / "ood_3d.log")
    logger.info("Starting 3D Out-of-Distribution Scanner Shift Benchmark...")

    resolved_data_dir = Path(args.data_dir).resolve() if getattr(args, "data_dir", None) else None
    try:
        test_dataset = BraTS3DDataset(split="test", data_dir=resolved_data_dir)
    except FileNotFoundError:
        if not args.smoke_test:
            raise
        logger.warning("BraTS3D test split not found. Using synthetic dataset for smoke test.")
        test_dataset = [
            {
                "image": torch.randn(4, 128, 128, 128),
                "mask": torch.randint(0, 5, (1, 128, 128, 128)).long(),
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

    instantiated_models = []
    for m_name, m_type in selected_models:
        is_scratch = m_type == "vit_scratch" or args.from_scratch
        is_frozen = m_type == "vit_frozen" or getattr(args, "freeze_encoder", False)
        m = load_segmentation_model_for_eval(
            model_type=m_type,
            label=m_name,
            device=device,
            decoder_type=args.decoder_type,
            is_scratch=is_scratch,
            is_frozen=is_frozen,
            patch_size=getattr(args, "patch_size", None),
            checkpoints_dir=checkpoints_dir,
            resolved_data_dir=resolved_data_dir,
            logger=logger,
        )
        instantiated_models.append((m_name, m))

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
            lambda img, bm: (
                img * torch.tensor([0.0, 1.0, 0.0, 0.0], device=img.device).view(1, 4, 1, 1, 1)
                if img.dim() == 5
                else img * torch.tensor([0.0, 1.0, 0.0, 0.0], device=img.device).view(4, 1, 1, 1)
            ),
        ),
        (
            "Missing Modalities: FLAIR Only",
            lambda img, bm: (
                img * torch.tensor([0.0, 0.0, 0.0, 1.0], device=img.device).view(1, 4, 1, 1, 1)
                if img.dim() == 5
                else img * torch.tensor([0.0, 0.0, 0.0, 1.0], device=img.device).view(4, 1, 1, 1)
            ),
        ),
    ]

    records = []

    def _display_name(m_name: str) -> str:
        dec = getattr(args, "decoder_type", "multiscale")
        if dec != "multiscale" and ("JEPA" in m_name or "ViT-FPN" in m_name):
            return f"{m_name} [{dec}]"
        return m_name

    for p_name, p_fn in perturbations:
        logger.info(f"\n--- Evaluating Regime: {p_name} ---")
        row = {"Regime": p_name}
        for m_name, m in instantiated_models:
            if m is None:
                row[_display_name(m_name)] = "N/A"
                continue
            scores = evaluate_perturbation(
                m, test_loader, device, p_fn, amp=args.amp, smoke_test=args.smoke_test,
                tta=args.tta,
            )
            dname = _display_name(m_name)
            logger.info(
                f"{m_name} -> WT {scores['WT'] * 100:.2f}% / "
                f"TC {scores['TC'] * 100:.2f}% / ET {scores['ET'] * 100:.2f}% "
                f"(mean {scores['mean'] * 100:.2f}%)"
            )
            row[dname] = f"{scores['mean'] * 100:.2f}%"
            for r in ("WT", "TC", "ET", "mean"):
                row[f"{dname} [{r}]"] = f"{scores[r] * 100:.2f}%"
        records.append(row)

    new_df = pd.DataFrame(records)
    csv_path = metrics_dir / "ood_3d_summary.csv"
    md_path = metrics_dir / "ood_3d_summary.md"

    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            existing_df["Regime"] = existing_df["Regime"].astype(str).str.strip()
            new_df["Regime"] = new_df["Regime"].astype(str).str.strip()

            new_cols = [c for c in new_df.columns if c not in existing_df.columns]
            overlap_cols = [c for c in new_df.columns if c != "Regime" and c in existing_df.columns]
            if new_cols:
                merged = pd.merge(existing_df, new_df[["Regime"] + new_cols], on="Regime", how="outer")
            else:
                merged = pd.merge(existing_df, new_df[["Regime"]], on="Regime", how="outer")
            if overlap_cols:
                new_indexed = new_df.set_index("Regime")
                for col in overlap_cols:
                    for r_val, val in new_indexed[col].items():
                        if pd.notna(val) and str(val) != "N/A":
                            merged.loc[merged["Regime"] == r_val, col] = val
            df = merged.fillna("N/A")
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
