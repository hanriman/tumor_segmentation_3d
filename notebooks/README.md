# Research Notebook Suite (BraTS 2024 GLI Multiregion Protocol)

**Protocol: BraTS 2024 WT/TC/ET on the v2 dataset version.** 5-class outputs
(bg/NCR/ED/NET/ET); eval emits one row per `(model, region)`. Never mix these
tables with legacy v1 binary whole-tumor results.

## Files

| Notebook | Purpose |
|---|---|
| `01_train_visreg_3d.ipynb` | SSL pre-training (label-free; exports `visreg_pretrain_outputs.zip` encoder checkpoint). |
| `02_diagnose_visreg_3d.ipynb` | Label-free SSL masking diagnostics (D1 air fractions; D2/D3 with a VisReg checkpoint). |
| `03_train_nnunet_3d.ipynb` | 3D nnU-Net baseline (DynUNet with multi-scale deep supervision). |
| `04_train_unet_3d.ipynb` | 3D Residual UNet baseline (MONAI ResUNet). |
| `05_train_vit_from_scratch_ablation_3d.ipynb` | From-scratch ViT + Multi-Scale FPN ablation (random initialization control). |
| `06_vit_unetr_hybrid_ablation_3d.ipynb` | From-scratch ViT + Hybrid UNETR ablation (random initialization control). |
| `07_finetune_visreg_multiscale_3d.ipynb` | VisReg + Multi-Scale FPN downstream fine-tuning on shared SSL encoder. |
| `08_finetune_visreg_unetr_3d.ipynb` | VisReg + Hybrid UNETR downstream fine-tuning on shared SSL encoder. |


## Battery reporting (v2 regions protocol)

Test (`evaluate_3d`), low-data (`evaluate_low_data_3d`), and OOD
(`evaluate_ood_3d`) all emit one column/row per `(model, region)`:
`Model [WT]`, `[TC]`, `[ET]`, plus `[mean]` (average of the three, for
tier/regime comparability with v1 tables). Binary (C=1) WT-only paths have
been removed from the codebase. Aggregation merges on column names, so region columns
flow through `aggregate_low_data_summaries` / `aggregate_ood_summaries`
untouched.

The v1 binary whole-tumor notebooks and the v1 `kaggle_runner_3d` orchestrator
were removed with the multiregion-only cleanup (recoverable from git history).

## Requirements

* Attach **`brats-3d-full` dataset VERSION 2** (integer masks 0–4, `has_et`
  column). Kaggle attachments pin versions — a stale v1 pin silently mounts
  binary masks and voids every number.
* Code: branch `main` (configs default `out_channels=5`).
