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
  - random B1 field varies per call,
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

### Step 6 — B1 field: random coefficients per sample [P0]
**Problem:** `transforms.py:183-185` single fixed field for all volumes.
- [ ] 6.1 Sample `c_ijk ~ N(0, σ²)` per forward (per-sample in batch) for the 10-term 2nd-order polynomial (`i+j+k≤2`), scaled by `strength`. Accept optional `generator` for determinism.
- [ ] 6.2 Keep coordinate grid `[-1,1]³`, multiplicative `(1+field)`, clamp field to e.g. `[-0.5,0.5]` to avoid sign flip; preserve background-zeroing semantics (or air handling consistent with Step 5 decision).
- [ ] 6.3 Tests: two calls same seed → identical field; different seed → different; field smooth (low-freq by construction).

Done when: randomness + determinism tests pass.

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

### Step 11 — Remove encoder mode-toggle side effect [P2]
**Problem:** `sigreg_jepa_3d.py:85-89`, `visreg_jepa_3d.py:75-80` flip `train()/eval()` inside `forward`.
- [ ] 11.1 Replace with gradient-free target pass that does **not** mutate mode: `with torch.no_grad():` + dropout-disabled context. Options: (a) `torch.no_grad` + temporarily set `requires_grad=False` only, relying on `model.eval()` externally for val; or (b) functional dropout-off via `self.context_encoder.eval()` snapshot restored in `finally` (minimal change) + comment. Prefer (b) as surgical fix, (a) as follow-up research change — document choice.
- [ ] 11.2 Test: `model.train()` before forward → still `training==True` after; `model.eval()` → stays `False`; grads flow only to context path.

### Step 12 — Freeze sincos positional embeddings [P2]
**Problem:** `vision_transformer_3d.py:178` learnable `pos_embed` lets metric topology drift.
- [ ] 12.1 Set `requires_grad=False` (or `register_buffer`) for `pos_embed` in encoder **and** predictor; keep weight-loading compat (`strict=False` + prefix strip already handles it).
- [ ] 12.2 Note in docstring: fixed Vaswani/Feichtenhofer topology from step 0.
- [ ] 12.3 Test: `pos_embed.requires_grad==False`; optimizer param list excludes it; smoke run finite.

### Step 13 — Aggregation + LaTeX hygiene [P2]
**Problem:** `utils/aggregation.py:87-93` best-Dice dedup hides variance; `251` hardcodes `N=271`.
- [ ] 13.1 Dedup policy: keep all runs + add `run_id` column, OR keep best but add `n_runs` + `std` columns and log dropped rows at `INFO`. Implement one explicitly.
- [ ] 13.2 Parameterize LaTeX caption `N=` (derive from metadata or pass `test_n=`); remove hardcoded 271.
- [ ] 13.3 Tests: duplicate-model CSVs → both preserved or explicitly summarized; caption contains injected `N`.

### Step 14 — Harden UNet DS hooks [P2]
**Problem:** `models/unet_3d.py:86-100` permanent hooks, version-sensitive.
- [ ] 14.1 Use `with torch.no_grad()`-safe local hook handles + `try/finally` removal per forward, or bind once with `removable` handles and explicit `close()`; fix type hint (`list[tuple[str,Tensor]]`).
- [ ] 14.2 Keep `RuntimeError` on missing shape/channel mismatch (already good).
- [ ] 14.3 Test: two consecutive forwards identical outputs; no handle leak (`len(model.unet._forward_hooks)` bounded); eval returns single tensor.

---

## P3 — Dead Code & Clarity (do last)

### Step 15 — Deduplicate tissue filter + freeze branches [P3]
- [ ] 15.1 Extract `filter_tissue_tokens(full_projected, mask, min_tokens=32)` helper; both JEPA models call it.
- [ ] 15.2 Collapse `JEPASegmentationModel3D.forward` freeze duplication (`segmentation_head_3d.py:505-509,512-516`) into single `torch.no_grad()`-guarded encoder call.
- [ ] 15.3 Unify double RNG counters (`dataset.py` + `masking.py`) — keep per-sample `initial_seed+idx` scheme, remove one layer, document worker-divergence intent.
- [ ] 15.4 Replace `resolve_seg_loss_type` `sys.argv` sniffing with explicit `loss_type` pass-through from trainers; keep backward-compat shim + deprecation log.

### Step 16 — Docs sync [P3]
- [ ] 16.1 Update README §2 rows: modality-dropout rescaling, torch-only RNG, SigReg tissue-only, random B1, Rician air, fixed sincos, NaN HD95 semantics, Dice-background default.
- [ ] 16.2 Update `docs/roadmap_improvements_ablation_dataset.md` Phase 8: mark SigReg parity done, per-sample fallback done, new OOD re-baseline required.
- [ ] 16.3 Note OOD number break: prior Rician/B1 tables not comparable post-Step 5/6.

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
