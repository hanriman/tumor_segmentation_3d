# Research Notebook Suite (BraTS 2024 GLI Multiregion Protocol)

**Protocol: BraTS 2024 WT/TC/ET on the v2 dataset version.** 5-class outputs
(bg/NCR/ED/NET/ET); eval emits one row per `(model, region)`. Never mix these
tables with legacy v1 binary whole-tumor results.

## Files

| Notebook | Purpose |
|---|---|
| `01_train_visreg_3d.ipynb` | SSL pre-training (label-free; Option B 3D target jitter [97 tokens across 6 cuboids], dual cosine schedules [LR warmup + regularizer annealing]; exports `visreg_pretrain_outputs.zip` encoder checkpoint). |
| `02_diagnose_visreg_3d.ipynb` | Label-free SSL masking diagnostics (D1 air fractions; D2/D3 with a VisReg checkpoint). |
| `03_train_nnunet_3d.ipynb` | 3D nnU-Net baseline (DynUNet with multi-scale deep supervision). |
| `04_train_unet_3d.ipynb` | 3D Residual UNet baseline (MONAI ResUNet). |
| `05_train_vit_from_scratch_ablation_3d.ipynb` | From-scratch ViT + Multi-Scale FPN ablation (random initialization control). |
| `06_vit_unetr_hybrid_ablation_3d.ipynb` | From-scratch ViT + Hybrid UNETR ablation (random initialization control). |
| `07_finetune_visreg_multiscale_3d.ipynb` | VisReg + Multi-Scale FPN downstream fine-tuning on shared SSL encoder. |
| `08_finetune_visreg_unetr_3d.ipynb` | VisReg + Hybrid UNETR downstream fine-tuning on shared SSL encoder. |
| `colab_a100_visreg_pretrain.ipynb` | Dedicated NVIDIA A100 (Google Colab Web UI) high-resolution 8³ voxel patch pre-training with private Kaggle API download and Google Drive checkpoint persistence. |


## Unified Configuration & Reproducibility Schema

All notebooks (`01` through `08` and `colab_a100_visreg_pretrain`) share a standardized top-level configuration block (Cell 3 / Cell 6) eliminating hardcoded CLI magic numbers:

```python
SEED = 42                 # Deterministic seed for weight init, augmentations, and data splits
NUM_WORKERS = 4          # Parallel CPU data loading workers (Kaggle T4 provides 4 vCPUs)
PATCH_SIZE = 16          # 16³ voxel patch (512 token grid) on T4; 8³ voxel patch (4096 token grid) on A100
BATCH_SIZE = 2           # Volumetric downstream batch size (nnU-Net: ~7.4 GB VRAM at B=2 with DS)
EPOCHS = 30              # Supervised training / fine-tuning epochs (pre-training: 50–100)
LOW_DATA_EPOCHS = 15     # Low-data label efficiency evaluation epochs
LEARNING_RATE = 3e-4     # Base AdamW learning rate (pre-training: 5e-4 with cosine schedule)
WEIGHT_DECAY = 1e-4      # AdamW weight decay (pre-training: 0.05)
```

CLI execution cells uniformly interpolate these parameters:
```bash
!python scripts/train_downstream_3d.py \
    --data_dir {data_dir} \
    --patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE} \
    --seed {SEED} \
    --epochs {EPOCHS} \
    --batch_size {BATCH_SIZE} \
    --num_workers {NUM_WORKERS} \
    --learning_rate {LEARNING_RATE} \
    --weight_decay {WEIGHT_DECAY} \
    --amp
```

## Resolution-Aware JEPA Masking

Masking configuration is decoupled from the dataset contract ([configs/dataset/brats3d.yaml](file:///Users/hanriman/Documents/master/thesis/thesis_3d/configs/dataset/brats3d.yaml)) and managed by the self-supervised schedule ([configs/experiment/pretrain.yaml](file:///Users/hanriman/Documents/master/thesis/thesis_3d/configs/experiment/pretrain.yaml)):
* **Context Budget:** Always fixed to `context_fraction: 0.375` (37.5% of total volume: 192 tokens on $8^3$ grid, 1,536 tokens on $16^3$ grid).
* **Option B Cuboid Target Shapes:**
  * **$8^3$ Token Grid ($16^3$ patch size, Kaggle T4):** $\mathcal{S} = [(3,3,3), (2,3,3), (3,2,3), (3,3,2), (2,2,2), (2,2,2)]$ ($97\text{ tokens total}$).
  * **$16^3$ Token Grid ($8^3$ patch size, Colab A100):** $\mathcal{S} = [(6,6,6), (4,6,6), (6,4,6), (6,6,4), (4,4,4), (4,4,4)]$ ($776\text{ tokens total}$).
  * **Physical millimeter invariance:** In both grid resolutions, target cuboids correspond to the exact same anatomical volume ($48\text{ mm}$ macro cube, $32 \times 48 \times 48\text{ mm}$ planar slabs, and $32\text{ mm}$ regional meso cubes).

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

