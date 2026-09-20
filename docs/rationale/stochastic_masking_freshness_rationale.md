# Rationale: Stochastic Mask Freshness Across Epochs (Counter-Seeded Determinism)

**Status:** Bug found by audit, fixed, regression-tested (`tests/test_data_3d.py::test_brain_aware_masking_varies_with_counter`).
**Applies to:** `src/brats_jepa_3d/data/masking.py`, `src/brats_jepa_3d/data/dataset.py`.

---

## 1. Why masks must vary across epochs

Stochastic patch masking promises a *different* context/target puzzle every time a
volume is shown: epoch 1 predicts region A from region B, epoch 2 predicts C from D.
Over a 100-epoch pretraining run, each volume therefore yields ~100 distinct
prediction tasks. This diversity is load-bearing for JEPA: the encoder must organize
a representation space that supports *arbitrary* spatial queries, not memorize one
fixed split per volume.

## 2. The failure mode: epoch-frozen masks

When the masking RNG was made DataLoader-worker-safe, the per-sample generator was
seeded exclusively on `(worker_seed, volume_index)` — both constant across epochs.
Consequence, verified empirically:

```
epoch1_mask == epoch2_mask  →  True   (identical context/target sets, all 100 epochs)
```

Each volume presented the same puzzle 100 times: stochastic masking degraded to a
fixed 37.5%/21% split, silently shrinking the effective pretraining diversity by two
orders of magnitude. The legacy code never had this bug (global `random` state
advanced on every call); it was introduced alongside the worker-safety fix.

## 3. Design constraints on the fix

1. **Worker safety:** `torch.utils.data` workers are forked processes; Python's
   global `random` state is identical in every worker unless reseeded, producing
   correlated masks across the batch. All mask randomness must flow through an
   explicit `torch.Generator` (see `JEPAMaskingTransform3D.__call__(generator=...)`).
2. **Determinism:** same seed + same config must reproduce the same training run
   (thesis reproducibility). Pure entropy (`torch.seed()`, wall-clock) is forbidden.
3. **Freshness:** same `(worker, volume)` pair must yield different masks on
   successive epochs.

Constraints 2 and 3 conflict for any seed that is a pure function of `(worker, idx)`.
The resolution is a **monotonic call counter**: the dataset holds
`self._mask_counter`, incremented on every `__getitem__` that samples a mask, mixed
into the generator seed:

```
seed = torch.initial_seed() + idx * 7919 + counter * 104729   (mod 2**32)
```

- Freshness: the counter advances every sample, so epoch N+1 never repeats epoch N
  (for identical access order; with shuffling, pairing differs too — still fresh).
- Determinism: the counter is a pure function of the sample-access sequence, which
  is itself deterministic given seed + config. Same seed replays the same mask
  stream.
- Worker locality: each forked worker owns an independent counter copy, so streams
  diverge across workers (intended — correlated masks across a batch would
  reintroduce the original worker-safety bug).

## 4. Verification

- Unit: `test_brain_aware_masking_varies_with_counter` — same seed/idx across three
  counter values yields three disjoint mask sets, all satisfying the
  context∩target=∅ guarantee.
- Integration: `BraTS3DDataset` smoke + full pytest suite (116 passed).
- Residual risk: changing `num_workers` or shuffle order changes the mask stream
  (accepted: reproducibility is per-config, as with all DataLoader RNG).

## 5. Rule of thumb

Any RNG state derived from a *static* sample key (`idx`, `patient_id`) is frozen
across epochs by construction. Stochasticity-per-epoch requires a *dynamic* component
(step, counter, epoch number) in the seed. When reviewing masking/augmentation code,
check: "if I reset the epoch loop, does this draw repeat?" — if yes, it is a bug.
