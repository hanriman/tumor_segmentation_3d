#!/usr/bin/env python3
"""Interpolation baseline for 3D VisReg JEPA pretraining (diagnostic, no training).

Compares the trained JEPA predictor against a non-learned spatial-smoothing
baseline (inverse-distance weighting over context embeddings) in the same
embedding space, on the same masks drawn by the repo's own sampler.

For each val volume:
  true target embeddings  = frozen encoder, full-grid forward, gathered at targets
  JEPA prediction         = frozen predictor(context tokens, target positions)
  IDW prediction          = weighted average of context embeddings (1/d^2 in patch-grid units)

Reports mean MSE ratio  E_interp / E_jepa  overall and split by tissue tag
(background / boundary / interior, from the integer mask), plus effective rank
at random init vs trained checkpoint (epoch-0 datum for the flat-74 question).

Usage:
    uv run python scripts/interpolation_baseline_jepa_3d.py --n_vols 24 --seed 0
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brats_jepa_3d.data.dataset import BraTS3DDataset  # noqa: E402
from brats_jepa_3d.data.masking import JEPAMaskingTransform3D  # noqa: E402
from brats_jepa_3d.metrics.probing_metrics import (  # noqa: E402
    compute_effective_rank,
    compute_representation_collapse_metrics,
)
from brats_jepa_3d.models.visreg_jepa_3d import VisRegJEPA3D  # noqa: E402

_canonical_ckpt = PROJECT_ROOT / "outputs/checkpoints/visreg_jepa_3d_best.pt"
_mr_ckpt = (
    PROJECT_ROOT
    / "outputs/multiregion/visreg_pretrain_outputs"
    / "checkpoints/visreg_jepa_3d_best.pt"
)
DEFAULT_PRETRAIN_CKPT = _canonical_ckpt if _canonical_ckpt.exists() else _mr_ckpt
GRID = (8, 8, 8)
PATCH = 16


def parse_args():
    p = argparse.ArgumentParser(description="JEPA vs interpolation diagnostic")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_PRETRAIN_CKPT))
    p.add_argument("--n_vols", type=int, default=24)
    p.add_argument("--masks_per_vol", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--data_dir", type=str, default=str(
        PROJECT_ROOT / "data/processed/brats_gli_3d_v2"))
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def patch_coords(indices: torch.Tensor) -> torch.Tensor:
    """Flat patch indices [N] -> 3D grid coords [N, 3] on the 8^3 lattice."""
    idx = indices.long()
    gz, gy, gx = GRID
    z = idx // (gy * gx)
    y = (idx % (gy * gx)) // gx
    x = idx % gx
    return torch.stack([z, y, x], dim=-1).float()


def idw_predict(
    ctx_embs: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor, power: float = 2.0
) -> torch.Tensor:
    """Inverse-distance weighting: [N_tgt, D] from [N_ctx, D] in grid units."""
    c = patch_coords(ctx_idx)  # [C, 3]
    t = patch_coords(tgt_idx)  # [T, 3]
    d2 = ((t[:, None, :] - c[None, :, :]) ** 2).sum(-1)  # [T, C]
    w = 1.0 / (d2.pow(power / 2.0) + 1e-8)
    w = w / w.sum(dim=1, keepdim=True)
    return w @ ctx_embs


def token_tissue_tags(mask: torch.Tensor) -> torch.Tensor:
    """Per-patch tag from integer mask [1,128,128,128]: 0=bg, 1=boundary, 2=interior.

    Boundary = cuboid straddles tumor edge (0 < tumor_frac < 1).
    """
    m = (mask[0] > 0).float()  # [128,128,128]
    # tumor fraction per 16^3 patch -> [8,8,8]
    frac = F.avg_pool3d(m[None, None], kernel_size=PATCH, stride=PATCH)[0, 0]
    tags = torch.zeros_like(frac, dtype=torch.long)
    tags[(frac > 0) & (frac < 1)] = 1
    tags[frac == 1] = 2
    return tags.reshape(-1)  # [512]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = BraTS3DDataset(
        split=args.split,
        data_dir=args.data_dir,
        augmentations=None,  # no augmentation: diagnostic on clean volumes
    )
    n = min(args.n_vols, len(ds))
    print(f"val pool: {len(ds)} volumes, using {n}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)

    def build():
        return VisRegJEPA3D(
            img_size=(128, 128, 128), patch_size=(8, 8, 8),
            in_channels=4, embed_dim=384, encoder_depth=8,
            predictor_depth=4, predictor_embed_dim=192,
        )

    trained = build()
    missing, unexpected = trained.load_state_dict(sd, strict=False)
    print(f"trained ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}")
    trained.eval()

    torch.manual_seed(args.seed)  # identical init for the random twin
    random_model = build()
    random_model.eval()

    sampler = JEPAMaskingTransform3D()
    err = defaultdict(lambda: {"interp": 0.0, "jepa": 0.0, "n": 0})
    all_trained_tokens, all_init_tokens = [], []

    for vi in range(n):
        sample = ds[vi]
        image = sample["image"].float().unsqueeze(0)  # [1,4,128,128,128]
        mask = sample["mask"]
        brain = sample.get("brain_mask", (image[0].abs().sum(0, keepdim=True) > 0))
        # token brain fractions like the training pipeline
        pooled = F.avg_pool3d(brain.float().unsqueeze(0), kernel_size=PATCH, stride=PATCH)
        token_brain_frac = pooled.reshape(-1)
        tags = token_tissue_tags(mask)

        with torch.no_grad():
            full_trained = trained.context_encoder(image)  # [1,512,384]
            full_init = random_model.context_encoder(image)
        all_trained_tokens.append(full_trained[0])
        all_init_tokens.append(full_init[0])

        for rep in range(args.masks_per_vol):
            gen = torch.Generator().manual_seed(args.seed * 100003 + vi * 101 + rep)
            m = sampler(token_brain_frac=token_brain_frac, generator=gen)
            ctx_idx = m["context_indices"]
            with torch.no_grad():
                ctx_tokens = trained.context_encoder(image, patch_indices=ctx_idx[None])[0]
                for tgt_idx in m["target_indices_list"]:
                    tgt_idx = tgt_idx
                    true = full_trained[0][tgt_idx]  # [T,384]
                    pred = trained.predictor(
                        ctx_tokens[None], ctx_idx[None], tgt_idx[None]
                    )[0]
                    base = idw_predict(ctx_tokens, ctx_idx, tgt_idx)
                    se_j = ((pred - true) ** 2).mean(-1)
                    se_i = ((base - true) ** 2).mean(-1)
                    for k in range(tgt_idx.numel()):
                        tag = ("background", "boundary", "interior")[int(tags[int(tgt_idx[k])])]
                        err[tag]["interp"] += float(se_i[k])
                        err[tag]["jepa"] += float(se_j[k])
                        err[tag]["n"] += 1
        if (vi + 1) % 6 == 0:
            print(f"  ... {vi + 1}/{n} volumes")

    rows = {}
    for tag, d in err.items():
        mi = d["interp"] / d["n"]
        mj = d["jepa"] / d["n"]
        rows[tag] = {"mse_interp": mi, "mse_jepa": mj, "ratio": mi / mj, "n_tokens": d["n"]}
    tot_i = sum(d["interp"] for d in err.values())
    tot_j = sum(d["jepa"] for d in err.values())
    tot_n = sum(d["n"] for d in err.values())
    rows["overall"] = {
        "mse_interp": tot_i / tot_n, "mse_jepa": tot_j / tot_n,
        "ratio": tot_i / tot_j, "n_tokens": tot_n,
    }

    stacked_trained = torch.stack(all_trained_tokens).reshape(-1, 384)
    stacked_init = torch.stack(all_init_tokens).reshape(-1, 384)
    rank = {
        "init_eff_rank": compute_effective_rank(stacked_init),
        "trained_eff_rank": compute_effective_rank(stacked_trained),
    }
    rank.update({
        f"trained_{k}": v
        for k, v in compute_representation_collapse_metrics(
            stacked_trained, generator=torch.Generator().manual_seed(args.seed)
        ).items()
    })

    out = {
        "checkpoint": str(args.checkpoint), "ckpt_epoch": ckpt.get("epoch"),
        "n_vols": n, "masks_per_vol": args.masks_per_vol, "seed": args.seed,
        "mse_by_tissue": rows, "rank": rank,
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
