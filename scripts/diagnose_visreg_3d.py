#!/usr/bin/env python3
"""Real-data VisReg-3D diagnostics (run on Kaggle where BraTS data exists).

D1: mask air-fraction + fallback rate using brain_mask.
D2: air-vs-tissue target prediction loss split (frozen encoder).
D3: VisReg center/scale/shape on all tokens vs tissue-only tokens.

Usage:
    python scripts/diagnose_visreg_3d.py --num_volumes 200 [--checkpoint outputs/checkpoints/<pretrain>.pt]
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from brats_jepa_3d.config import get_dataset_dir
from brats_jepa_3d.data import BraTS3DDataset
from brats_jepa_3d.data.masking import JEPAMaskingTransform3D
from brats_jepa_3d.losses import VisRegLoss
from brats_jepa_3d.models import VisRegJEPA3D


def token_brain_frac(brain_mask, grid=(8, 8, 8), ps=16):
    bm = brain_mask[0] if brain_mask.dim() == 4 else brain_mask  # [D,H,W]
    fracs = torch.zeros(grid[0] * grid[1] * grid[2])
    for z in range(grid[0]):
        for y in range(grid[1]):
            for x in range(grid[2]):
                blk = bm[z*ps:(z+1)*ps, y*ps:(y+1)*ps, x*ps:(x+1)*ps]
                fracs[z*64 + y*8 + x] = blk.float().mean().item()
    return fracs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_volumes", type=int, default=200)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    ds = BraTS3DDataset(split="train", masking_transform=None, augmentations=None)
    print(f"dataset: {get_dataset_dir()} n={len(ds)}")
    T = JEPAMaskingTransform3D()
    n = min(args.num_volumes, len(ds))
    ctx_air, tgt_air, air_blocks, n_blocks = [], [], 0, 0
    for i in range(n):
        s = ds[i]
        fr = token_brain_frac(s["brain_mask"] if "brain_mask" in s else (s["image"] != 0).any(0, keepdim=True).float())
        out = T()
        ctx = out["context_indices"].tolist()
        tgts = [t.tolist() for t in out["target_indices_list"]]
        ctx_air.append(sum(1 for c in ctx if fr[c] < 0.10) / len(ctx))
        flat = [t for g in tgts for t in g]
        tgt_air.append(sum(1 for t in flat if fr[t] < 0.10) / len(flat))
        for g in tgts:
            n_blocks += 1
            if all(fr[t] < 0.10 for t in g):
                air_blocks += 1
    print(f"[D1] ctx air mean={sum(ctx_air)/n:.3f} | tgt air mean={sum(tgt_air)/n:.3f} | "
          f"all-air tgt blocks={air_blocks}/{n_blocks}={air_blocks/max(1,n_blocks):.3f}")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model = VisRegJEPA3D()
        sd = ckpt.get("model_state_dict", ckpt)
        # Full SSL state dict (context_encoder + projector + predictor) when
        # available: D3 probes projected tokens, so the trained projector matters.
        # Falls back to encoder-only loading for encoder-only checkpoints.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_matched = len(sd) - len([k for k in sd if k in missing])
        if n_matched == 0:
            pref = {k[len("context_encoder."):]: v for k, v in sd.items()
                    if k.startswith("context_encoder.")}
            model.context_encoder.load_state_dict(pref, strict=False)
            n_matched = len(pref)
            print(f"Loaded encoder-only weights ({n_matched} keys); projector/predictor random.")
        else:
            print(f"Loaded full SSL weights ({n_matched} keys matched, "
                  f"{len(missing)} missing, {len(unexpected)} unexpected).")
        model.eval()
        crit = VisRegLoss()
        air_losses, tis_losses, s_all, s_tis, w_all, w_tis = [], [], [], [], [], []
        with torch.no_grad():
            for i in range(min(n, 50)):
                s = ds[i]
                img = s["image"].unsqueeze(0).float()
                fr = token_brain_frac(s["brain_mask"] if "brain_mask" in s else (img[0] != 0).any(0, keepdim=True).float())
                out = T()
                ctx = out["context_indices"].unsqueeze(0)
                tgt = [t.unsqueeze(0) for t in out["target_indices_list"]]
                fwd = model(img, ctx, tgt)
                for p, t, tb in zip(fwd["predictions"], fwd["targets"], tgt):
                    tn = F.layer_norm(t.detach(), (t.shape[-1],))
                    l = F.smooth_l1_loss(p, tn, reduction="none").mean(-1)[0]
                    is_air = torch.tensor([fr[x] < 0.10 for x in tb[0].tolist()])
                    air_losses.append(l[is_air].mean().item() if is_air.any() else float("nan"))
                    tis_losses.append(l[~is_air].mean().item() if (~is_air).any() else float("nan"))
                z = fwd["projected_tokens"].reshape(-1, fwd["projected_tokens"].shape[-1])
                s_all.append(crit._scale_loss(z).item()); w_all.append(crit._sliced_wasserstein_distance(z).item())
                keep = torch.tensor([fr[c] >= 0.10 for c in ctx[0].tolist()])
                zt = fwd["projected_tokens"][0][keep]
                if len(zt):
                    s_tis.append(crit._scale_loss(zt).item()); w_tis.append(crit._sliced_wasserstein_distance(zt).item())
        import numpy as np
        print(f"[D2] air-tgt loss={np.nanmean(air_losses):.4f} tissue-tgt loss={np.nanmean(tis_losses):.4f}")
        print(f"[D3] scale all={np.mean(s_all):.4f} tissue-only={np.mean(s_tis):.4f} | "
              f"shape all={np.mean(w_all):.4f} tissue-only={np.mean(w_tis):.4f}")


if __name__ == "__main__":
    main()
