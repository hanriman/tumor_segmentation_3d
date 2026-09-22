# Multiregion notebooks (v2 protocol)

**Protocol: BraTS 2024 WT/TC/ET on the v2 dataset version.** 5-class outputs
(bg/NCR/ED/NET/ET); eval emits one row per `(model, region)`. Never mix these
tables with v1 binary whole-tumor results.

## Files

| Notebook | Purpose |
|---|---|
| `01_train_visreg_3d.ipynb` | SSL pretraining (label-free; identical on either pool — prefer v2 attachment so fingerprints match downstream). |
| `02_train_nnunet_3d.ipynb` | nnU-Net v2 baseline (5-class via config). |
| `03_train_unet_3d.ipynb` | Phase 2 gate: UNet full-data on v2 (sane WT → lower TC → lowest ET). ET-collapse fallback: `--loss_type tversky`. |
| `04_train_vit_from_scratch_ablation_3d.ipynb` | From-scratch multiscale ablation (v2). |
| `06_vit_unetr_hybrid_ablation_3d.ipynb` | From-scratch hybrid ablation (v2). |
| `07_finetune_visreg_multiscale_3d.ipynb` | VisReg FPN finetune on the shared SSL encoder + full battery. |
| `08_finetune_visreg_unetr_3d.ipynb` | VisReg hybrid finetune on the shared SSL encoder + full battery. |

## Battery reporting (v2 regions protocol)

Test (`evaluate_3d`), low-data (`evaluate_low_data_3d`), and OOD
(`evaluate_ood_3d`) all emit one column/row per `(model, region)`:
`Model [WT]`, `[TC]`, `[ET]`, plus `[mean]` (average of the three, for
tier/regime comparability with v1 tables). Binary (C=1) runs keep the legacy
single-column shape. Aggregation merges on column names, so region columns
flow through `aggregate_low_data_summaries` / `aggregate_ood_summaries`
untouched.

Deliberately absent: 05 (diagnose ran on v1; SSL-side, rerun only if masking code changes) and `kaggle_runner_3d` (v1 orchestrator; multiregion runs per-notebook until the battery shape is known).

## Requirements

* Attach **`brats-3d-full` dataset VERSION 2** (integer masks 0–4, `has_et`
  column). Kaggle attachments pin versions — a stale v1 pin silently mounts
  binary masks and voids every number.
* Code: branch `feat/multiregion-wt-tc-et` (configs default `out_channels=5`).
