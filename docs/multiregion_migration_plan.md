# Plan: Migrate from Binary Whole-Tumor to BraTS Three-Region Segmentation (WT/TC/ET)

**Status:** planned, not started. Tracked on branch `feat/multiregion-wt-tc-et` (cut after committing the 01/07/08 notebook split on main).
**Motivation:** the current pipeline segments binary whole-tumor (tumor yes/no); the BraTS community standard scores three overlapping regions (WT/TC/ET). See the Task Scope discussion: WT-only blocks leaderboard comparability and hides ET-specific failure modes.
**Representativeness note:** this is a supervised-side-only migration. SSL pretraining is label-free and untouched; the 100-epoch encoder investment is preserved.

---

## Branching & Data-Version Convention

- **Branch:** all work below happens on `feat/multiregion-wt-tc-et`. Main stays runnable for WT-protocol runs (including the in-flight JEPA pretrain and the 07/08 downstream notebooks). If a shared-file bugfix lands on main mid-migration, rebase this branch instead of patching twice.
- **Data versions (independent of git — record per run, per table):**
  - **v1** = current Kaggle `brats-3d-full` (binary masks, WT-only protocol). All existing results (UNet, nnU-Net, scratch ablations, in-flight JEPA pretrain) are v1.
  - **v2** = reprocessed upload with integer labels 0-4 preserved as-is (0:bg, 1:NCR, 2:ED, 3:NET, 4:ET) + `has_et` metadata (label-4 present, Phase 0). All region-protocol runs are v2.
- **Labeling rule:** every results table and figure caption states its data version and protocol (v1/WT-only vs v2/WT-TC-ET). The two are never mixed or compared across the boundary.

---

## Design Decision

Train a **5-class softmax** (background, NCR, ED, NET, ET) and **derive the three overlapping BraTS regions by union** at evaluation:

- **WT** (whole tumor) = {NCR, ED, NET, ET} = labels {1, 2, 3, 4}
- **TC** (tumor core) = {NCR, NET, ET} = labels {1, 3, 4}
- **ET** (enhancing tumor) = labels {4}

This is the community-standard recipe. Consequences:

- **SSL pretraining is untouched**: no changes to masking, JEPA losses, or `scripts/train_jepa_3d.py`.
- **Only the supervised side changes**: data labels, decoder heads (`out_channels` 1 → 5), loss calls, metrics, eval scripts, TTA.
- The loss stack already contains dormant multi-class paths (softmax Dice+CE, Tversky, deep supervision) — this plan activates them rather than writing new math.

---

## Phase 0 — Data: Preserve Integer Labels

**Files:** `scripts/prepare_data_3d.py`, `src/brats_jepa_3d/data/dataset.py`

- [ ] In `process_patient`, save the resampled segmentation with **original integer labels** (0/1/2/3/4, nearest-neighbor — already the case) instead of collapsing to `(seg > 0)`. Keep the field name `mask`; every binary consumer uses `(mask > 0)`, so old code keeps working on new files (backward compatible by construction).
- [ ] Verify the BraTS-2024 label convention in the raw data (expect exactly {0,1,2,3,4} with 1=NCR, 2=ED, 3=NET, 4=ET — confirmed on 1,350 vols, Sep 2026) and fail loud on any other value.
- [ ] Extend `metadata.csv`: keep `num_voxels_tumor` (WT, preserves stratification), add `has_et` / `num_voxels_et` — some cases lack enhancing tumor, and evaluation must know that.
- [ ] Re-run `prepare_data_3d.py` (full pool, not `--limit`). Cost: one-time, CPU-only, minutes with workers. Upload the result as a **new version (v2) of the existing `brats-3d-full` Kaggle dataset** — do not create a second dataset. In Kaggle notebooks, set the dataset attachment to latest (attachments pin versions; a stale pin silently mounts v1 binary masks).
- [ ] Regression test: integer masks round-trip through `BraTS3DDataset` with values ⊆ {0, 1, 2, 3}; `(mask > 0)` reproduces the old binary masks exactly. Upload v2 only after these pass.

---

## Phase 1 — Losses & Metrics: Multi-Class Paths Live

**Files:** `src/brats_jepa_3d/losses/*`, `src/brats_jepa_3d/metrics/volumetric_metrics.py`, `src/brats_jepa_3d/utils/tta.py`, `tests/`

- [ ] **Losses**: activate and test the existing multi-class branches — `VolumetricDiceLoss` with explicit `include_background=False`, `CombinedDiceBCELoss3D` multi-class CE path, `VolumetricTverskyLoss` per-class path. Decide the Tversky-BCE wrapper: extend to multi-class CE or keep it binary-only (document either way; do not leave a silent middle ground).
- [ ] **Region derivation helper** (new, small, pure function): `(logits[B,4,...]) → {WT, TC, ET}` binary masks via `argmax → union`. Unit-test against hand-built volumes.
- [ ] **Metrics**: keep `compute_volumetric_metrics_3d` binary-only (correct and tested); add a thin `compute_brats_regions_3d` wrapper that calls it **once per region** — this inherits all guarded NaN semantics, HD95, and `nanmean` aggregation for free, including ET-missing cohorts.
- [ ] **TTA**: add a multiclass path alongside `predict_with_tta_3d` (average softmax probabilities across flips, then back to logits). The current sigmoid roundtrip is binary-only and must not silently run on 4-channel outputs; assert on channel count.
- [ ] Tests: region derivation, per-region NaN on ET-free cohorts, Tversky/Dice equivalence on binarized inputs, multiclass TTA shape contract.

---

## Phase 2 — One Model End-to-End First (UNet, Cheapest)

**Files:** `scripts/train_downstream_3d.py` (plus UNet/nnU-Net trainers), `configs/`, `scripts/evaluate_*.py`

- [ ] Thread `out_channels=5` (via config, not hardcoded) through `JEPASegmentationModel3D`, `BraTS3DUNet`, `BraTS3DnnUNet` — deep-supervision heads adapt automatically (1×1×1 convolutions). Old C=1 checkpoints will not load into C=5 heads: expected, document it, no compatibility shim.
- [ ] Rewire the three eval scripts to region evaluation (WT/TC/ET tables instead of a single Dice). Keep CSV shapes aggregation-friendly: one row per (model, region) or region-suffixed columns — decide once, since `combine_and_generate_paper_artifacts.py` must parse it.
- [ ] Train **UNet only**, full data, and confirm: sane WT (≈ current 87%), TC lower, ET lowest with higher variance. If ET collapses (tiny-structure instability), add region-weighted Dice or per-region Tversky *before* touching other arms.
- [ ] Gate: UNet 3-region results plausible → proceed; otherwise fix loss weighting, not architectures.

---

## Phase 3 — Retrain All Arms + Re-benchmark

- [ ] VisReg multiscale (notebook 07) and hybrid (notebook 08) finetunes on the frozen Phase-1 recipe — same encoder, same hyperparameters, only `out_channels` differs. This preserves the controlled decoder ablation.
- [ ] nnU-Net baseline in 5-class mode.
- [ ] Re-run the full battery per arm: test split, low-data tiers, OOD regimes — now ×3 regions. **Budget warning**: this roughly triples remaining eval compute. Sequence it (UNet → 07 → 08 → nnU-Net) so partial results are already citable if quota dies mid-way. Label every table/figure with data version + protocol per the convention above.
- [ ] Update aggregation + LaTeX export for region-dimensioned tables; paper `main.tex` results section.

---

## Phase 4 — Paper & Docs

- [ ] Methods "Task definition" paragraph flips from scope-justification to standard BraTS protocol; Limitations paragraph shrinks to the remaining items (single-center splits, no prospective data).
- [ ] README §2/§5/§6.3, `kaggle_guide.md`, notebooks 07/08 headers: WT-only caveats → region reporting. Notebooks need only text edits (commands gain no new flags if `out_channels` lives in config).
- [ ] Never mix: old WT-only tables and new region tables must be labeled as different protocols — they are not comparable.

---

## Explicitly Out of Scope

SSL pretraining, masking, OOD corruption models, encoder checkpoint formats, and the aggregation framework's overall shape. The migration is supervised-side-only by design — that is what keeps it tractable.

## Relation to Other Docs

- `roadmap_improvements_ablation_dataset.md` — engineering roadmap; this migration is its candidate journal-version upgrade.
- `remediation_plan_audit_2026-09-21.md` — audit trail; Phase 1–2 work here should follow the same test-gated discipline.
- `kaggle_guide.md` — update Job 1b/1c and archive lists when Phase 3 runs.
