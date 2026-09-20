# Rationale: Brain-Aware Masking and the SWD Air-Spike Problem

**Status:** Implemented (`masking.py`, `dataset.py`, `visreg_jepa_3d.py`), verified on proxy
+ local control (200 real volumes, paired) + Kaggle D1 (full train split).
**Applies to:** 3D VisReg JEPA pretraining (`train_jepa_3d.py --model_type visreg_jepa`).

---

## 1. The anatomical fact

The 128³ training cube is mostly zero padding: skull-stripped brains occupy ~27–40%
of cube voxels, and 316/512 patch tokens contain <10% brain. Uniform sampling over
the 8×8×8 token grid therefore spends most of the JEPA task on air:
context 54.3% air / targets 35.8% air on the proxy ellipsoid (only ~29%
tissue→tissue pairs); 34.8% / 22.4% on real data (real brains fill more of the cube).
The 2D slice setting never suffered this at scale (axial slices are mostly brain),
so uniform sampling — inherited from the 2D recipe — silently underperformed in 3D.

## 2. Why air targets waste the prediction loss

Air patches are constant zeros → constant tokens → trivially predictable targets that
contribute near-zero prediction error (measured: air-target loss 0.0201 vs tissue
0.1309). Every air target is gradient budget spent learning nothing. The fix steers
target sampling toward tissue via per-volume token brain-fractions pooled from the
(unsupervised, non-tumor) `brain_mask`: rejection sampling accepts 3³ cuboids with
≥14/27 tissue tokens (20 attempts, best-attempt fallback preserving the overlap
control), keeping spatial diversity — a stricter threshold was rejected after
measurement showed targets clustering centrally.

## 3. The SWD air-spike problem (the larger lever)

VisReg's shape penalty projects batch tokens onto 256 random directions, sorts each
projection, and matches standard-normal quantiles. Identical air-constant tokens form
a sharp **spike** beside the tissue cloud; the penalty then spends gradient forcing
the spike into bell-curve shape instead of organizing tissue features. Measured on
controlled synthetic features (identical tissue cloud ± air constants, random
projector): scale loss 0.0011 → 0.0253 (**23×**), shape loss 0.0043 → 0.0811
(**19×**). The fix filters `projected_tokens` to tissue rows before `VisRegLoss`
(<32 tissue tokens falls back to the full batch), so the spike never enters the
computation. Context air (~55–59%) is deliberately tolerated: context is model
*input*, not loss, and 192 requested tokens cannot fit into ~140 available tissue
tokens — filtering the regularization is sufficient.

## 4. Before / after mechanics

| Quantity | Before (uniform) | After (brain-aware + tissue-only SWD) | Evidence |
|---|---|---|---|
| Target air fraction | 35.8% proxy / 22.4% real | 25.5% proxy / 16.6% real (−26%, p<1e-16, 0/200 worsened) | proxy sweep + paired real control |
| Tissue→tissue pairs | ~29% / 50.6% | ~majority / 54.8% | same |
| SWD input | 192 tokens incl. air spike | ~76 tissue tokens | verified shapes |
| SWD scale/shape on air mix | 0.0253 / 0.0811 | 0.0011 / 0.0043 (tissue-only) | synthetic mechanism test |
| Mask stream | frozen across epochs (see `stochastic_masking_freshness_rationale.md`) | fresh per epoch, deterministic per config | verified |
| Kaggle D1 (full train split) | — | ctx 33.6% / tgt 23.4%, 0 all-air blocks | `05_diagnose_visreg_3d.ipynb` |

(Proxy numbers overstate the problem — real anatomy is kinder. Cite the real-control
column in the thesis, not the proxy column.)

## 5. Fairness position

`brain_mask` derives from nonzero voxels: unsupervised, tumor-agnostic, and already
used for parenchyma-only noise in `transforms.py`. Masking affects SSL pretraining
only; downstream splits, augmentations, and losses are identical across baselines —
the same category as the 2D recipe's candidate-window BFS, which likewise biased
training toward tissue without supervision implications.

## 6. What this does and does not prove

Proven: the training *signal* is cleaner (less air in targets, no air in SWD, fresh
masks). Not proven by these diagnostics: downstream Dice gains — that requires the
100-epoch pretraining run and the old-vs-new checkpoint tissue-loss comparison
(D2 rerun). Report mechanism evidence as justification for the run, never as a
performance claim.
