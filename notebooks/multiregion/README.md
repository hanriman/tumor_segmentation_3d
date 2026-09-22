# Multiregion notebooks (v2 protocol)

**Protocol: BraTS 2024 WT/TC/ET on the v2 dataset version.** 5-class outputs
(bg/NCR/ED/NET/ET); eval emits one row per `(model, region)`. Never mix these
tables with v1 binary whole-tumor results.

## Files

| Notebook | Purpose |
|---|---|
| `03_train_unet_3d.ipynb` | Phase 2 gate: UNet full-data on v2 (sane WT → lower TC → lowest ET). ET-collapse fallback: `--loss_type tversky`. |
| `07_finetune_visreg_multiscale_3d.ipynb` | VisReg FPN finetune on the shared SSL encoder + full battery. |
| `08_finetune_visreg_unetr_3d.ipynb` | VisReg hybrid finetune on the shared SSL encoder + full battery. |

Deliberately absent: 02 (nnU-Net follows the gate), 04/06 (scratch ablations join
the Phase 3 battery), 01/05 (SSL pretraining is label-free — no v2 copy needed),
runner (deferred until the battery shape is known).

## Requirements

* Attach **`brats-3d-full` dataset VERSION 2** (integer masks 0–4, `has_et`
  column). Kaggle attachments pin versions — a stale v1 pin silently mounts
  binary masks and voids every number.
* Code: branch `feat/multiregion-wt-tc-et` (configs default `out_channels=5`).
