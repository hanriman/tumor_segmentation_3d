# Remediation Plan — Post-Audit Fixes (2026-09-21)

**Source:** forensic audit of `src/brats_jepa_3d/` + `scripts/train_jepa_3d.py` (2026-09-21).
**Scope:** fix-only. No new architectures, no `8³` token resolution (Phase 4 stays deferred), no hyperparameter retuning beyond restoring mathematical intent.
**How to use:** execute Phase 0 → P0 → P1 → P2 → P3 in order. Each step lists files, concrete edits, regression test, and done-criteria. Check the box only when the test passes on `pytest tests/ -q` plus the step's own check.

---

## Phase 0 — Baseline & Safety Harness (do first, ~1h)

- [ ] **0.1 Pin baseline.** Record `pytest tests/ -q` result, `git status`, `git log --oneline -3`. Save to top of implementation log.
- [ ] **0.2 Add failing-first regression tests** (TDD for P0/P1). New file `tests/test_audit_remediation_2026_09_21.py` with skipped/expected-fail tests for:
  - modality-dropout expectation preservation,
  - torch-only determinism under fixed seed,
  - SigReg tissue filtering parity with VisReg,
  - per-sample tissue fallback,
  - Rician air noise non-zero + non-negativity,
  - random B1 field varies per sample,
  - masking zero-overlap hard guarantee,
  - `hd95_tumor_only is nan` when `compute_hd95=False` / no-tumor cohort.
- [ ] **0.3 Smoke gate.** Define gate: `train_jepa_3d.py --smoke_test` (all 3 model types) + full `pytest` must pass before merging any phase.

Files touched: `tests/test_audit_remediation_2026_09_21.py` (new).

---

## P0 — Mathematical Correctness Bugs (train/test shift + OOD validity)

### Step 1 — Modality dropout: restore inverted-dropout scaling [P0]
**Problem:** `src/brats_jepa_3d/data/transforms.py:48-57` zeros channels without rescaling → ~25% expected-magnitude drop at train vs test.
- [ ] 1.1 Scale kept channels by `1 / keep_prob` where `keep_prob = mask.sum()/C` per sample (fallback single-channel → scale = `C`). Preserve `mask` semantics (0/1) then multiply by scale factor.
- [ ] 1.2 Keep eval path (`self.training==False` or `p_drop<=0`) identity.
- [ ] 1.3 Unit test: seeded batch, `E[output|train] ≈ input` mean over channels within tolerance; eval identity; all-drop fallback still yields exactly 1 active scaled channel.
- [ ] 1.4 Update class docstring (remove stale "no rescale" implication, state inverted scaling).

Done when: expectation test passes; downstream smoke loss finite.

### Step 2 — Deterministic RNG: eliminate `random.random` [P0]
**Problem:** `transforms.py:99,107` uses Python `random`, breaking seeded worker reproducibility.
- [ ] 2.1 Replace with `torch.rand((), generator=...)` or `torch.rand(1).item()` on `image.device`; thread an optional `torch.Generator` through `VolumetricAugmentations3D.__call__` (default `None` → derive from `torch.initial_seed()` like `masking.py:89-100`).
- [ ] 2.2 Same-seed → bitwise-identical `(image, mask, brain_mask)` across two calls; different-seed → differs with high probability.
- [ ] 2.3 Keep `RandomModalityDropout3D` on torch PRNG (already ok); document generator contract.

Done when: determinism test passes twice in a row under `torch.manual_seed(0)`.

### Step 3 — SigReg tissue parity (air-dilution fix) [P0]
**Problem:** `models/sigreg_jepa_3d.py:97` regularizes all 192 context tokens; VisReg filters tissue (`visreg_jepa_3d.py:91-105`); trainer only plumbs mask for VisReg (`scripts/train_jepa_3d.py:330-336,413-417`).
- [ ] 3.1 Add `context_tissue_mask: Tensor|None = None` to `SigRegJEPA3D.forward`, replicate VisReg filtering logic (share a helper, e.g. `models/_tissue_filter.py` or static method) with same `<32 → fallback` rule as interim (refined in Step 4).
- [ ] 3.2 Update `scripts/train_jepa_3d.py` train + val loops to pass mask for `sigreg_jepa` identically to `visreg_jepa`.
- [ ] 3.3 Update `SigRegLoss` docstring + README §2 row (tissue-only now both).
- [ ] 3.4 Test: synthetic batch with known air rows → `projected_tokens` excludes air; smoke pretrain finite `sigreg_loss`.

Done when: SigReg and VisReg produce identical filtering on same mask fixture.

### Step 4 — Per-sample (not batch-wide) tissue fallback [P0]
**Problem:** `visreg_jepa_3d.py:96` `(counts>=32).all()` poisons whole batch on one thin volume.
- [ ] 4.1 Replace with per-sample filtering: for each `b`, if `count_b>=32` keep `full_projected[b][mask[b]]` else keep all `N_ctx` rows of `b`; `torch.cat` across `b`. Apply via shared helper from Step 3 (both models).
- [ ] 4.2 Preserve 2D output `[N_kept, proj_dim]` contract for loss flattening.
- [ ] 4.3 Test: B=2, counts `[100, 5]` → kept rows `100+192`; B=2 both `≥32` → `sum(counts)`.

Done when: per-sample test passes; batch-all fallback code removed.

### Step 5 — Rician OOD: air noise + principled non-negativity [P0]
**Problem:** `transforms.py:142-149` shift-by-min hack, background forced to zero (no Rayleigh air).
- [ ] 5.1 Apply Rician magnitude to **all voxels** (or at minimum: tissue Rician + background Rayleigh `sqrt(eta1²+eta2²)`), then `where(brain, noisy_tissue, noisy_air)` instead of zeros. Keep dtype/device.
- [ ] 5.2 Replace shift-back trick with documented choice: operate on raw magnitude assumption; if input has negatives (Z-scored), clamp shifted baseline explicitly and note approximation in docstring (do not silently re-add `min_val` to mimic raw MRI).
- [ ] 5.3 Guard `sigma<=0` → identity; empty-mask → identity (keep existing).
- [ ] 5.4 Tests: background voxels non-zero after call with `sigma>0`; non-negativity where applicable; `sigma=0` identity.

Done when: OOD smoke eval runs and air-noise test passes. Note: this changes OOD numbers — re-baseline expected.

### Step 6 — B1 field: random coefficients per sample [P0] — DONE (per-sample follow-up)
**Problem:** `transforms.py:183-185` single fixed field for all volumes (interim fix: one random field broadcast across the batch).
- [x] 6.1 Sample `c_ijk ~ N(0, 1)` per sample in the batch (`[B, 10]` via `einsum` over the 10-term 2nd-order polynomial), scaled by `strength`. Accepts optional `generator` for determinism (CPU-sampled, then moved to device for cross-device reproducibility).
- [x] 6.2 Keep coordinate grid `[-1,1]³`, multiplicative `(1 + strength*field)`, field clamped to `[-0.5, 0.5]` so gain stays in `[1-0.5*strength, 1+0.5*strength]` (docstring gain-range `[0.5, 1.5]` corrected); preserve background-zeroing semantics (gain on signal; air contrast vs Rician documented).
- [x] 6.3 Tests: identical inputs in one batch now get different fields; same seed → identical; explicit generator → identical; README §2 B1 row + §6.3 caution updated.

Done when: randomness + determinism tests pass. Note: this changes OOD numbers again — re-baseline required (prior per-call-broadcast tables not comparable).

---

## P1 — Guarantee / Contract Violations

### Step 7 — Hard zero-overlap masking guarantee [P1]
**Problem:** `data/masking.py:130-159` best-effort fallback can violate `max_target_overlap=0.0`.
- [ ] 7.1 Enforce hard constraint: if no zero-overlap placement found in 20 attempts, shrink/relocate deterministically (e.g. exhaustive scan of valid `z0,y0,x0` grid via generator order) or raise informative `RuntimeError` naming attempts — never silently accept overlap when `max_target_overlap==0`.
- [ ] 7.2 Keep tissue-preference (`≥14/27`) as soft preference, overlap as hard constraint (overlap wins ties).
- [ ] 7.3 Tests: 500 synthetic masks, `pairwise target intersections == 0` when `max_overlap=0`; nonzero `max_overlap` still respected.

### Step 8 — `hd95_tumor_only` NaN semantics [P1]
**Problem:** `metrics/volumetric_metrics.py:263-267` returns `0.0` (perfect) when HD95 not computed or no-tumor cohort.
- [ ] 8.1 Return `float("nan")` for `hd95_tumor_only` when `compute_hd95==False` or `tumor_indices` empty; keep `hd95` as `nan` consistently.
- [ ] 8.2 Audit callers (`scripts/evaluate_*.py`, notebooks aggregation) for `nanmean` handling so CSVs show `n/a` not `0.0`.
- [ ] 8.3 Tests: `compute_hd95=False` → both `nan`; all-empty cohort with `True` → `dice==1.0`, `hd95_tumor_only is nan`.

### Step 9 — Dice-background contract alignment [P1]
**Problem:** default `include_background=True` (`losses/dice_bce_loss_3d.py:87,99`) contradicts README "Dice excludes background".
- [ ] 9.1 Decide + document: binary `C=1` unaffected; for multi-class set explicit `include_background=False` at all trainer call sites OR change default to `False` with migration note. Do one, not both silently.
- [ ] 9.2 Verify `CombinedDiceBCELoss3D` CE still supervises all voxels (`ignore_index=-100`) — keep that asymmetry intentional and documented.
- [ ] 9.3 Test: 4-class synthetic logits → Dice loss differs between `True/False`; default matches documented choice.

### Step 10 — Representation diagnostics on val, not stale train batch [P1]
**Problem:** `scripts/train_jepa_3d.py:469` uses leftover train `images`.
- [ ] 10.1 Capture first val batch tokens (`val_images[:1]` via `model.context_encoder`) for `compute_representation_collapse_metrics`; guard when val loader empty.
- [ ] 10.2 Log split source (`val EffRank`) to avoid misreading.

---

## P2 — Fragility / Fairness / Reporting

### Step 11 — Remove encoder mode-toggle side effect [P2] — DONE (dropout-disabled context)
**Problem:** `sigreg_jepa_3d.py`, `visreg_jepa_3d.py` flipped `train()/eval()` inside `forward`.
- [x] 11.1 Gradient-free target pass that does **not** mutate mode: `with torch.no_grad(), dropout_disabled(self.context_encoder)` (`models/vision_transformer_3d.py`). Zeroes `nn.Dropout.p` + `nn.MultiheadAttention.dropout` and restores on exit; thread/DDP-safe, no caller-visible side effect. Encoder input still includes air by design (only the projector regularizer is tissue-filtered).
- [x] 11.2 Test: `model.train()` before forward → still `training==True` after; `model.eval()` → stays `False`; grads flow only to context path (`test_p2_step11_no_mode_side_effect` green).

### Step 12 — Freeze sincos positional embeddings [P2]
**Problem:** `vision_transformer_3d.py:178` learnable `pos_embed` lets metric topology drift.
- [ ] 12.1 Set `requires_grad=False` (or `register_buffer`) for `pos_embed` in encoder **and** predictor; keep weight-loading compat (`strict=False` + prefix strip already handles it).
- [ ] 12.2 Note in docstring: fixed Vaswani/Feichtenhofer topology from step 0.
- [ ] 12.3 Test: `pos_embed.requires_grad==False`; optimizer param list excludes it; smoke run finite.

### Step 13 — Aggregation + LaTeX hygiene [P2] — DONE (variance reporting)
**Problem:** `utils/aggregation.py` best-Dice dedup hides variance; hardcoded `N=271`.
- [x] 13.1 Keep-best policy retained but with `n_runs` + `dice_std` columns and dropped-row `INFO` log — no run disappears silently.
- [x] 13.2 LaTeX caption `N=` parameterized via `test_n=` (no hardcoded 271).
- [x] 13.3 Tests: duplicate-model CSVs → explicitly summarized; caption contains injected `N`.

### Step 14 — Harden UNet DS hooks [P2] — DONE (bounded + version-recorded)
**Problem:** `models/unet_3d.py` permanent hooks, version-sensitive.
- [x] 14.1 Bound once in `__init__` with removable handles + explicit `close()`; type hint `list[tuple[str,Tensor]]`; `monai_version` recorded on the module for drift diagnosis.
- [x] 14.2 `RuntimeError` on missing shape/channel mismatch kept.
- [x] 14.3 Test: two consecutive forwards identical outputs; no handle leak; eval returns single tensor (`test_p2_step14_unet_hooks_bounded` green).

---

## P3 — Dead Code & Clarity (do last)

### Step 15 — Deduplicate tissue filter + freeze branches [P3]
- [ ] 15.1 Extract `filter_tissue_tokens(full_projected, mask, min_tokens=32)` helper; both JEPA models call it.
- [x] 15.2 Collapse `JEPASegmentationModel3D._encode` freeze duplication into a single `torch.no_grad()` / `nullcontext`-guarded encoder call.
- [ ] 15.3 Unify double RNG counters (`dataset.py` + `masking.py`) — keep per-sample `initial_seed+idx` scheme, remove one layer, document worker-divergence intent.
- [x] 15.4 Remove `resolve_seg_loss_type` `sys.argv` sniffing; legacy direct calls without an explicit-flag record or `cli_args` now log a deprecation warning and fall back to the accepted-value check (no `sys.argv` read).

### Step 16 — Docs sync [P3] — DONE (follow-up batch)
- [x] 16.1 Update README §2 rows: modality-dropout rescaling, torch-only RNG, SigReg tissue-only, per-sample random B1 (+ corrected gain range), Rician air, fixed sincos, NaN HD95 semantics, Dice-background default; Grid Resampling row + preprocessing §5.1 now state the bbox-to-cube stretch (aspect not preserved).
- [x] 16.2 Update `docs/roadmap_improvements_ablation_dataset.md` Phase 8: per-sample B1 done, OOD re-baseline required again.
- [x] 16.3 Note OOD number break: prior Rician/B1 tables not comparable post-Step 5/6 (README §6.3 caution extended to per-sample B1).

---

## Execution Order & Gates

1. Phase 0 (harness) → 2. P0 Steps 1–6 (one PR each, smoke + targeted test) → 3. P1 Steps 7–10 → 4. P2 Steps 11–14 → 5. P3 Steps 15–16.
2. After each P0/P1 step: `pytest tests/test_audit_remediation_2026_09_21.py tests/test_data_3d.py tests/test_losses_3d.py tests/test_models_3d.py tests/test_metrics_3d.py -q`.
3. Final gate: full `pytest tests/ -q` + `train_jepa_3d --smoke_test` ×3 model types + `evaluate_3d --smoke_test` (or `--compute_hd95` off) green.
4. Do **not** start downstream re-benchmarks until P0 done — OOD/low-data numbers will shift.

## Out of Scope (explicitly deferred)

- Finer `8³`/4096-token ViT, SDPA/FlashAttention-2 (Phase 4).
- New loss weights (`center/scale/swd`, `sigreg_weight`) tuning.
- Full-pool Kaggle re-run (needs dataset upload + GPU session).
- Context-air content (`~55%` air in encoder input by design) — only regularization filtered; encoder-input air removal is a research change, not a bug fix.

---

## Residual Fix Log (2026-09-21, post-audit follow-up)

All four residual items from the re-audit are fixed. Tests: `test_audit_remediation_2026_09_21.py + test_metrics_3d.py` **23 passed**; `test_losses + test_models + test_data` **24 passed**.

- [x] **R1 — Tumor-only NaN triad.** `dice_tumor_only`/`iou_tumor_only` returned `1.0` on tumor-free cohorts while `hd95_tumor_only` was NaN. Now all three return NaN (`metrics/volumetric_metrics.py`); per-sample guarded `1.0` on empty volumes unchanged. Eval scripts consume only `*_per_sample` lists, so no script changes needed. Test `test_p1_step8_*` extended to all three keys.
- [x] **R2 — Air-model contrast sentence.** B1 docstring now states the deliberate difference from Rician (gain on signal vs. magnitude noise floor); Rician + B1 docstrings carry the 2026-09-21 re-baseline note. README §2 B1 row and §6.3 OOD bullet carry the same notes.
- [x] **R3 — Re-baseline note.** Covered by R2 (code docstrings + README caution block). Prior OOD tables must be re-run via `evaluate_ood_3d.py` before citing.
- [x] **R4 — Loud shape contract.** `filter_tissue_tokens` raises `ValueError` on non-3D input; `SigRegLoss`/`VisRegLoss` validate non-empty 2D `[N, D]` after flatten. New test `test_residual_tissue_filter_rejects_non3d`.

Notebooks (01–06, kaggle_runner): verified — no hardcoded metric semantics. All OOD/eval cells delegate to `scripts/evaluate_ood_3d.py` / `evaluate_3d.py`, which aggregate `*_per_sample` lists and are unaffected by the tumor-only NaN change. No `.ipynb` edits required.

## Follow-Up Fix Log (post-remediation audit batch)

Code fixes (all backward-compatible; full `pytest tests/` green — 134 passed):

- [x] **F1 — B1 per-sample fields.** `apply_b1_bias_field_3d` samples `[B, 10]` coefficients (`einsum`, CPU-sampled for generator/device safety) so every volume gets its own field; gain-range docstring corrected to `[1-0.5*strength, 1+0.5*strength]`. Verified: identical batch inputs diverge; same seed / same generator reproduce.
- [x] **F2 — Seeded regularizer projections.** `SigRegLoss.forward` / `VisRegLoss.forward` (+ `_sliced_wasserstein_distance`) accept optional `generator` (CPU-sampled, moved to device); default path unchanged.
- [x] **F3 — IJEPA loss in fp32.** `IJEPALoss` casts block losses to float32 before accumulation (was prediction dtype → fp16 under AMP).
- [x] **F4 — Explicit Tversky binary guard.** `CombinedTverskyBCEWithLogitsLoss3D.forward` raises `ValueError` on `C != 1` instead of relying on a broadcast error.
- [x] **F5 — `_encode` dedup + logging hygiene.** Single `no_grad`/`nullcontext` encoder path; `aggregation.py` / `export.py` `print()` → `logging` (notebook `export_artifacts(..., verbose=True)` callers unaffected — return-dict contract unchanged).
- [x] **F6 — Aspect-ratio honesty.** `prepare_data_3d.py` docstring + README §2 Grid Resampling row + §5.1 state the bbox-to-cube stretch (aspect not preserved).

Docs/notebooks synced: README §2 B1/SigReg/VisReg/baseline rows, §5.1, §6.3 caution; roadmap Phase 8 status; this plan (Steps 6 / 15.2 / 15.4 / 16); notebook OOD cells carry the per-sample re-baseline pointer. OOD tables must be re-run via `evaluate_ood_3d.py` before citing (per-sample B1 break).

## Fix Log — F1–F13 audit batch (2026-09-21, second round)

Code fixes (targeted `47 passed`; rest of suite `120 passed`; smoke asserts green):

- [x] **F1 — Mode-toggle removed.** `dropout_disabled()` context (`models/vision_transformer_3d.py`, exported) zeroes `Dropout.p` + `MultiheadAttention.dropout` and restores; both JEPA models use `torch.no_grad() + dropout_disabled` with no `.eval()/.train()` flip.
- [x] **F2 — VisReg mu detached.** Shape term detaches both `mu` and `std`; center handled only by `_center_loss`.
- [x] **F3 — Sign-preserving OOD.** Rician tissue: true Rician for `X>=0`, additive Gaussian for Z-scored negatives, Rayleigh air. B1: gain on original voxels (negatives preserved), zero air. Second OOD number break — re-run `evaluate_ood_3d.py` before citing.
- [x] **F4 — Air scope documented.** Encoder input includes air by design; only the projector regularizer is tissue-filtered (model docstrings).
- [x] **F5 — HD95 per-volume seed.** Content-derived RNG seed replaces fixed `42/42`; same pair reproducible, different volumes differ.
- [x] **F6/F7 — Dataset/masking hygiene.** Pooling failures log warnings (no silent `None`); `random` fallback and function-local logging import removed.
- [x] **F8 — UNet version record.** `monai_version` stored on module; loud `RuntimeError` guards kept.
- [x] **F9 — Aggregation variance.** `dice_std` column + dropped-row `INFO` log alongside `n_runs`.
- [x] **F10 — Tversky background flag.** `VolumetricTverskyLoss(include_background=True)` default behavior-preserving; `False` excludes class 0 for multi-class parity with Dice.
- [x] **F11 — Bottleneck fast path.** `_encode(x, return_intermediate=False)` skips intermediate list for bottleneck decoder.
- [x] **F12 — README scope.** §6.2 relabeled unvalidated targets with do-not-cite caution.
- [x] **F13 — Calibration status.** SigReg/VisReg docstrings marked uncalibrated with sweep guidance; EP truncation noted. Weight tuning itself remains deferred (Out of Scope).

Docs/notebooks synced: README §2 VisReg/Rician/B1/Tversky rows + §6.3 second-break caution; roadmap Phase 8 status; notebook OOD cells carry the sign-preserving re-baseline pointer. OOD tables must be re-run via `evaluate_ood_3d.py` before citing.
