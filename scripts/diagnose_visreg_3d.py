#!/usr/bin/env python3
"""Real-data 3D JEPA diagnostics for brain-aware masking and VISReg regularization.

D1: mask air-fraction + all-air block rate using brain_mask.
D2: air-vs-tissue target prediction loss split.
D3: VISReg center/scale/shape on all context tokens vs tissue-only tokens.

Usage:
    python scripts/diagnose_visreg_3d.py --num_volumes 200 [--data_dir PATH] [--checkpoint PATH]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from brats_jepa_3d.config import get_dataset_dir
from brats_jepa_3d.data import BraTS3DDataset
from brats_jepa_3d.data.masking import JEPAMaskingTransform3D
from brats_jepa_3d.losses import VisRegLoss
from brats_jepa_3d.models import SingleEncoderJEPA3D


def token_brain_frac(brain_mask: torch.Tensor, grid: tuple[int, int, int] = (16, 16, 16)) -> torch.Tensor:
    bm = brain_mask[0] if brain_mask.dim() == 4 else brain_mask  # [D, H, W]
    pooled = F.adaptive_avg_pool3d(bm.float().unsqueeze(0).unsqueeze(0), grid)
    return pooled.flatten()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=None)
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--num_volumes", type=int, default=200)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    ps = int(args.patch_size)
    grid = (128 // ps, 128 // ps, 128 // ps)
    data_dir = args.data_dir or get_dataset_dir()

    ds = BraTS3DDataset(data_dir=data_dir, split="train", masking_transform=None, augmentations=None)
    print(f"dataset: {data_dir} | n={len(ds)} | patch_size={ps}^3 | grid={grid}")
    T = JEPAMaskingTransform3D(grid_size=grid)
    n = min(args.num_volumes, len(ds))
    ctx_air, tgt_air, air_blocks, n_blocks = [], [], 0, 0
    for i in range(n):
        s = ds[i]
        bm = s["brain_mask"] if "brain_mask" in s else (s["image"] != 0).any(0, keepdim=True).float()
        fr = token_brain_frac(bm, grid=grid)
        out = T(token_brain_frac=fr)
        ctx = out["context_indices"].tolist()
        tgts = [t.tolist() for t in out["target_indices_list"]]
        ctx_air.append(sum(1 for c in ctx if fr[c] < 0.10) / len(ctx))
        flat = [t for g in tgts for t in g]
        tgt_air.append(sum(1 for t in flat if fr[t] < 0.10) / len(flat))
        for g in tgts:
            n_blocks += 1
            if all(fr[t] < 0.10 for t in g):
                air_blocks += 1
    print(
        f"[D1] ctx air mean={sum(ctx_air)/n:.3f} | tgt air mean={sum(tgt_air)/n:.3f} | "
        f"all-air tgt blocks={air_blocks}/{n_blocks}={air_blocks/max(1,n_blocks):.3f}"
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = SingleEncoderJEPA3D(patch_size=(ps, ps, ps))
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_matched = len(sd) - len([k for k in sd if k in missing])
        if n_matched == 0:
            pref = {
                k[len("context_encoder."):]: v
                for k, v in sd.items()
                if k.startswith("context_encoder.")
            }
            model.context_encoder.load_state_dict(pref, strict=False)
            n_matched = len(pref)
            print(f"Loaded encoder-only weights ({n_matched} keys); predictor random.")
        else:
            print(
                f"Loaded full SSL weights ({n_matched} keys matched, "
                f"{len(missing)} missing, {len(unexpected)} unexpected)."
            )
        model.eval()
        crit = VisRegLoss()
        air_losses, tis_losses, s_all, s_tis, w_all, w_tis = [], [], [], [], [], []
        with torch.no_grad():
            for i in range(min(n, 50)):
                s = ds[i]
                img = s["image"].unsqueeze(0).float()
                bm = s["brain_mask"] if "brain_mask" in s else (img[0] != 0).any(0, keepdim=True).float()
                fr = token_brain_frac(bm, grid=grid)
                out = T(token_brain_frac=fr)
                ctx = out["context_indices"].unsqueeze(0)
                tgt = [t.unsqueeze(0) for t in out["target_indices_list"]]
                fwd = model(img, ctx, tgt)
                ctx_tokens = model.context_encoder(img, patch_indices=ctx)
                for p, t, tb in zip(fwd["predictions"], fwd["targets"], tgt):
                    tn = F.layer_norm(t.detach(), (t.shape[-1],))
                    l = F.smooth_l1_loss(p, tn, reduction="none").mean(-1)[0]
                    is_air = torch.tensor([fr[x] < 0.10 for x in tb[0].tolist()])
                    air_losses.append(l[is_air].mean().item() if is_air.any() else float("nan"))
                    tis_losses.append(l[~is_air].mean().item() if (~is_air).any() else float("nan"))
                z = ctx_tokens.reshape(-1, ctx_tokens.shape[-1])
                s_all.append(crit._scale_loss(z).item())
                w_all.append(crit._sliced_wasserstein_distance(z).item())
                zt = fwd["tissue_tokens"]
                if len(zt) >= 2:
                    s_tis.append(crit._scale_loss(zt).item())
                    w_tis.append(crit._sliced_wasserstein_distance(zt).item())
        print(f"[D2] air-tgt loss={np.nanmean(air_losses):.4f} tissue-tgt loss={np.nanmean(tis_losses):.4f}")
        print(
            f"[D3] scale all={np.mean(s_all):.4f} tissue-only={np.mean(s_tis):.4f} | "
            f"shape all={np.mean(w_all):.4f} tissue-only={np.mean(w_tis):.4f}"
        )


if __name__ == "__main__":
    main()
