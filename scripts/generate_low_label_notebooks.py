#!/usr/bin/env python3
"""
scripts/generate_low_label_notebooks.py
------------------------------------
Generates the 8 parallel, cell-by-cell Google Colab notebooks for low-label benchmarking:
- Monotonic nested sampling (1%, 5%, 10%, 25%, 50%) adding up cumulative patient volumes.
- Early stopping with patience=5, min_epochs=5, max_epochs=50.
- Underscore naming convention (frac_0_01, <model>_0_01_best.pt, <model>_0_01_epoch_10.pt).
- Dedicated Train Cell and Held-Out Test Eval Cell per percentage.
- Aggregation into Table 2 format and curve plotting.
- 1-click zip export to Google Drive.
"""

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "notebooks" / "low_label"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRACTIONS = [
    ("0_01", "1%", 0.01, 12, "1% Annotation Budget (12 Patient Volumes)"),
    ("0_05", "5%", 0.05, 58, "5% Annotation Budget (58 Patient Volumes = 12 from 1% + 46 new)"),
    ("0_10", "10%", 0.10, 115, "10% Annotation Budget (115 Patient Volumes = 58 from 5% + 57 new)"),
    ("0_25", "25%", 0.25, 286, "25% Annotation Budget (286 Patient Volumes = 115 from 10% + 171 new)"),
    ("0_50", "50%", 0.50, 572, "50% Annotation Budget (572 Patient Volumes = 286 from 25% + 286 new)"),
]

MODELS = [
    {
        "filename": "01_low_label_visreg_multiscale.ipynb",
        "title": "VisReg Multi-Scale FPN Fine-Tuned",
        "role": "3D VisReg JEPA + Multi-Scale FPN Downstream Fine-Tuning (Primary Proposed)",
        "model_alias": "visreg_multiscale",
        "target_drive_folder": "thesis/low_label/visreg_multiscale",
        "zip_name": "visreg_multiscale_low_data.zip",
        "batch_size": 4,
        "needs_pretrain": True,
        "train_script": "scripts/train_downstream_3d.py",
        "train_flags": (
            '--model_type visreg_jepa --decoder_type multiscale '
            '--pretrained_checkpoint "{pretrained_ckpt}" '
            '--patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE} '
            '--deep_supervision'
        ),
        "eval_flags": (
            '--model_type visreg_jepa --decoder_type multiscale '
            '--patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE}'
        ),
    },
    {
        "filename": "02_low_label_visreg_unetr.ipynb",
        "title": "VisReg Hybrid UNETR Fine-Tuned",
        "role": "3D VisReg JEPA + Hybrid UNETR Downstream Fine-Tuning (Ablation Decoder)",
        "model_alias": "visreg_unetr",
        "target_drive_folder": "thesis/low_label/visreg_unetr",
        "zip_name": "visreg_unetr_low_data.zip",
        "batch_size": 4,
        "needs_pretrain": True,
        "train_script": "scripts/train_downstream_3d.py",
        "train_flags": (
            '--model_type visreg_jepa --decoder_type unetr_hybrid '
            '--pretrained_checkpoint "{pretrained_ckpt}" '
            '--patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE} '
            '--no_deep_supervision'
        ),
        "eval_flags": (
            '--model_type visreg_jepa --decoder_type unetr_hybrid '
            '--patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE}'
        ),
    },
    {
        "filename": "03_low_label_nnunet.ipynb",
        "title": "3D nnU-Net DynUNet Supervised Baseline",
        "role": "3D nnU-Net (DynUNet) Baseline with Deep Supervision (Supervised Benchmark)",
        "model_alias": "nnunet_3d",
        "target_drive_folder": "thesis/low_label/nnunet",
        "zip_name": "nnunet_low_data.zip",
        "batch_size": 2,
        "needs_pretrain": False,
        "train_script": "scripts/train_nnunet_3d.py",
        "train_flags": '--deep_supervision',
        "eval_flags": '--model_type nnunet_3d',
    },
    {
        "filename": "04_low_label_unet.ipynb",
        "title": "3D Residual U-Net Supervised Baseline",
        "role": "3D Residual Conv3D U-Net Baseline with Deep Supervision (Supervised Benchmark)",
        "model_alias": "unet_3d",
        "target_drive_folder": "thesis/low_label/unet",
        "zip_name": "unet_low_data.zip",
        "batch_size": 4,
        "needs_pretrain": False,
        "train_script": "scripts/train_unet_3d.py",
        "train_flags": '--deep_supervision',
        "eval_flags": '--model_type unet_3d',
    },
    {
        "filename": "05_low_label_scratch_multiscale.ipynb",
        "title": "3D ViT Multi-Scale FPN From Scratch",
        "role": "3D ViT + Multi-Scale FPN Trained From Scratch (Zero Pre-training Baseline)",
        "model_alias": "scratch_multiscale",
        "target_drive_folder": "thesis/low_label/scratch_multiscale",
        "zip_name": "scratch_multiscale_low_data.zip",
        "batch_size": 4,
        "needs_pretrain": False,
        "train_script": "scripts/train_downstream_3d.py",
        "train_flags": (
            '--model_type visreg_jepa --decoder_type multiscale --from_scratch '
            '--patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE} '
            '--deep_supervision'
        ),
        "eval_flags": (
            '--model_type visreg_jepa --decoder_type multiscale --from_scratch '
            '--patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE}'
        ),
    },
    {
        "filename": "06_low_label_scratch_unetr.ipynb",
        "title": "3D ViT Hybrid UNETR From Scratch",
        "role": "3D ViT + Hybrid UNETR Trained From Scratch (Zero Pre-training Baseline)",
        "model_alias": "scratch_unetr",
        "target_drive_folder": "thesis/low_label/scratch_unetr",
        "zip_name": "scratch_unetr_low_data.zip",
        "batch_size": 4,
        "needs_pretrain": False,
        "train_script": "scripts/train_downstream_3d.py",
        "train_flags": (
            '--model_type visreg_jepa --decoder_type unetr_hybrid --from_scratch '
            '--patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE} '
            '--no_deep_supervision'
        ),
        "eval_flags": (
            '--model_type visreg_jepa --decoder_type unetr_hybrid --from_scratch '
            '--patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE}'
        ),
    },
    {
        "filename": "07_low_label_frozen_visreg_multiscale.ipynb",
        "title": "VisReg Multi-Scale FPN Frozen Encoder Probe",
        "role": "3D VisReg JEPA + Multi-Scale FPN Linear / Non-Linear Probe (Frozen Encoder)",
        "model_alias": "frozen_visreg_multiscale",
        "target_drive_folder": "thesis/low_label/frozen_visreg_multiscale",
        "zip_name": "frozen_visreg_multiscale_low_data.zip",
        "batch_size": 4,
        "needs_pretrain": True,
        "train_script": "scripts/train_downstream_3d.py",
        "train_flags": (
            '--model_type visreg_jepa --decoder_type multiscale '
            '--pretrained_checkpoint "{pretrained_ckpt}" --freeze_encoder '
            '--patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE} '
            '--deep_supervision'
        ),
        "eval_flags": (
            '--model_type visreg_jepa --decoder_type multiscale --freeze_encoder '
            '--patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE}'
        ),
    },
    {
        "filename": "08_low_label_frozen_visreg_unetr.ipynb",
        "title": "VisReg Hybrid UNETR Frozen Encoder Probe",
        "role": "3D VisReg JEPA + Hybrid UNETR Linear / Non-Linear Probe (Frozen Encoder)",
        "model_alias": "frozen_visreg_unetr",
        "target_drive_folder": "thesis/low_label/frozen_visreg_unetr",
        "zip_name": "frozen_visreg_unetr_low_data.zip",
        "batch_size": 4,
        "needs_pretrain": True,
        "train_script": "scripts/train_downstream_3d.py",
        "train_flags": (
            '--model_type visreg_jepa --decoder_type unetr_hybrid '
            '--pretrained_checkpoint "{pretrained_ckpt}" --freeze_encoder '
            '--patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE} '
            '--no_deep_supervision'
        ),
        "eval_flags": (
            '--model_type visreg_jepa --decoder_type unetr_hybrid --freeze_encoder '
            '--patch_size {PATCH_SIZE} {PATCH_SIZE} {PATCH_SIZE}'
        ),
    },
]


def make_md_cell(content: str) -> dict:
    lines = [f"{line}\n" for line in content.splitlines()]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def make_code_cell(content: str) -> dict:
    lines = [f"{line}\n" for line in content.splitlines()]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def build_notebook(m: dict) -> dict:
    cells = []

    # Title & Overview
    header_md = f"""# 🔬 Colab GPU: Low-Label Benchmark – {m['title']}
### Protocol: BraTS 2024 GLI Multiregion Low-Data Benchmark
**Architecture Role**: {m['role']}

This notebook evaluates label efficiency across **1%, 5%, 10%, 25%, and 50%** annotation budgets using a **modular cell-by-cell execution pipeline**:
1. **Strict Monotonic Nested Sampling**: Every percentage strictly adds up cumulative patient scans ($\\mathcal{{S}}_{{1\\%}} \\subset \\mathcal{{S}}_{{5\\%}} \\subset \\mathcal{{S}}_{{10\\%}} \\subset \\mathcal{{S}}_{{25\\%}} \\subset \\mathcal{{S}}_{{50\\%}}$) via deterministic quantile ranking (`seed=42`).
2. **Dedicated Train Cell**: Trains with Early Stopping (`patience=5`, `min_epochs=5`, `max_epochs=50`, `save_freq=10`), saving checkpoints every 10 epochs and the best model (`_best.pt`) to Google Drive.
3. **Dedicated Held-Out Evaluation Cell**: Evaluates on all 242 held-out test volumes, printing WT, TC, ET Dice and HD95 live in the output.
4. **Summary & Curve Plot**: Aggregates all fractions into master Table 2 format and plots the efficiency curve.
5. **1-Click Package**: Archives all fraction outputs to Google Drive.
"""
    cells.append(make_md_cell(header_md))

    # Cell 1: Hardware & Drive Mount
    cells.append(make_md_cell("## 1. Hardware, CUDA & Google Drive Mounting\nVerifies GPU accelerator and mounts Google Drive."))
    cells.append(make_code_cell(
"""import datetime
import time
from pathlib import Path
import torch

print('=' * 60)
print('GPU Accelerator Status:')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('WARNING: CUDA not detected! Check Colab Runtime -> Change runtime type -> GPU.')
print('=' * 60)

from google.colab import drive
drive_mounted = Path('/content/drive/MyDrive').exists()
if not drive_mounted:
    drive.mount('/content/drive', force_remount=False)
print(f'Google Drive mounted: {Path("/content/drive/MyDrive").exists()}')"""
    ))

    # Cell 2: Configuration
    cells.append(make_md_cell("## 2. Standardized Configuration & Early Stopping Parameters\nDefines deterministic SEED=42 (guaranteeing nested cumulative subsets), patience=5, and Drive destinations."))
    cells.append(make_code_cell(
f"""# =========================================================================
# ⚙️ Standardized Low-Label Experiment Configuration
# =========================================================================
SEED = 42                 # Deterministic seed guaranteeing monotonic nested sampling
NUM_WORKERS = 4          # Parallel DataLoader workers
BATCH_SIZE = {m['batch_size']}          # Volumetric downstream batch size
PATCH_SIZE = 8           # 8³ voxel patch (4,096 tokens)
MAX_EPOCHS = 50          # Maximum training epochs ceiling
PATIENCE = 5             # Early stopping patience (epochs without validation improvement)
MIN_EPOCHS = 5           # Burn-in epochs before early stopping can trigger
MIN_STEPS = 250          # Step floor to prevent undertraining on tiny fractions
SAVE_FREQ = 10           # Periodic checkpoint cadence (saves _epoch_10.pt, _epoch_20.pt, etc.)

TARGET_DRIVE_FOLDER = '{m['target_drive_folder']}'
ZIP_NAME = '{m['zip_name']}'
MODEL_ALIAS = '{m['model_alias']}'

output_dir = Path(f'/content/drive/MyDrive/{{TARGET_DRIVE_FOLDER}}')
output_dir.mkdir(parents=True, exist_ok=True)

print(f'Model Alias:         {{MODEL_ALIAS}}')
print(f'Target Drive Folder: {{output_dir}}')
print(f'Early Stopping:      patience={{PATIENCE}}, min_epochs={{MIN_EPOCHS}}, max_epochs={{MAX_EPOCHS}}')
print(f'Checkpoint Cadence:  save_freq={{SAVE_FREQ}}, seed={{SEED}}')"""
    ))

    # Cell 3: Dependencies
    cells.append(make_md_cell("## 3. Dependencies Installation\nInstalls core medical imaging dependencies."))
    cells.append(make_code_cell(
"""!pip install -q --no-cache-dir monai nibabel tabulate matplotlib pandas
import monai
import nibabel as nib
print(f'MONAI: {monai.__version__} | NiBabel: {nib.__version__}')"""
    ))

    # Cell 4: Codebase Setup
    cells.append(make_md_cell("## 4. Codebase Setup & Editable Installation\nDiscovers repository and performs editable install."))
    cells.append(make_code_cell(
"""import os
import shutil
import sys
from pathlib import Path

code_candidates = [
    Path('/content/thesis_3d'),
    Path('/content/drive/MyDrive/thesis/thesis_3d'),
    Path.cwd(),
]
repo_root = None
for cand in code_candidates:
    if (cand / 'src' / 'brats_jepa_3d').exists():
        repo_root = cand
        break

if repo_root is None or not (repo_root / 'src' / 'brats_jepa_3d').exists():
    drive_repo = Path('/content/drive/MyDrive/thesis/thesis_3d')
    if drive_repo.exists():
        repo_root = Path('/content/thesis_3d')
        if not repo_root.exists():
            shutil.copytree(drive_repo, repo_root)
    else:
        repo_root = Path('/content/thesis_3d')

os.chdir(str(repo_root))
print(f'Working Directory: {Path.cwd()}')
!pip install -q -e ."""
    ))

    # Cell 5: Dataset Discovery
    cells.append(make_md_cell("## 5. Dataset Ingestion & Discovery\nResolves preprocessed BraTS 3D volumes on fast local NVMe SSD or Google Drive."))
    cells.append(make_code_cell(
"""import os
from pathlib import Path

data_candidates = [
    Path('/content/data/brats_gli_3d_v2'),
    Path('/content/data'),
    Path('/content/drive/MyDrive/thesis/data/brats_gli_3d_v2'),
    Path('/content/drive/MyDrive/thesis/data'),
    Path('data/processed/brats_gli_3d_v2'),
]
data_dir = None
for cand in data_candidates:
    if cand.exists() and (cand / 'metadata.csv').exists():
        data_dir = cand
        break

if data_dir is None:
    drive_zip = Path('/content/drive/MyDrive/thesis/data/brats_gli_3d_v2.zip')
    if drive_zip.exists():
        print(f'Unpacking {drive_zip} to /content/data/ ...')
        !unzip -q -n "{drive_zip}" -d /content/data/
        data_dir = Path('/content/data/brats_gli_3d_v2')

print(f'Dataset directory resolved: {data_dir}')
assert data_dir is not None and data_dir.exists(), 'Dataset not found!'"""
    ))

    # Cell 6: Pretrain intake (if applicable)
    if m["needs_pretrain"]:
        cells.append(make_md_cell("## 6. Pre-trained Checkpoint Intake\nLocates the pre-trained 3D VisReg JEPA encoder checkpoint from Google Drive."))
        cells.append(make_code_cell(
"""from pathlib import Path
from brats_jepa_3d.config import CHECKPOINTS_DIR
from brats_jepa_3d.utils import sort_checkpoints_by_epoch

pretrain_dir = Path('/content/drive/MyDrive/thesis/pretrain model')
candidates = []

if pretrain_dir.exists():
    for folder in [pretrain_dir / 'checkpoints', pretrain_dir]:
        if folder.is_dir():
            best = folder / 'visreg_jepa_3d_best.pt'
            if best.is_file():
                candidates.append(best)
            epochs = sort_checkpoints_by_epoch([p for p in folder.glob('visreg_jepa*_epoch_*.pt') if p.is_file()])
            if epochs:
                candidates.append(epochs[-1])
            general = [p for p in sorted(folder.glob('visreg_jepa*.pt')) if p.is_file() and 'scratch' not in p.name and 'frozen' not in p.name]
            candidates.extend(general)

if not candidates:
    for folder in [CHECKPOINTS_DIR, Path('outputs/checkpoints')]:
        if folder.is_dir():
            best = folder / 'visreg_jepa_3d_best.pt'
            if best.is_file():
                candidates.append(best)

pretrained_ckpt = str(candidates[0]) if candidates else None
assert pretrained_ckpt, (
    f'❌ Pre-trained checkpoint not found in /content/drive/MyDrive/thesis/pretrain model/!\\n'
    f'Please verify your pre-trained .pt file exists in Google Drive under thesis/pretrain model/.'
)
print(f'✓ Found pre-trained encoder checkpoint: {pretrained_ckpt}')"""
        ))

    # Cells for each fraction
    sec_num = 7 if m["needs_pretrain"] else 6
    for f_suffix, f_pct, f_float, n_vols, desc in FRACTIONS:
        # Train Section
        cells.append(make_md_cell(
f"""## {sec_num}. Annotation Budget: {f_pct} ({n_vols} Patient Volumes)
**Monotonic Subset**: {desc}

### {sec_num}.1 Train Cell ({f_pct})
Trains with Early Stopping (`patience=5`, `min_epochs=5`, `max_epochs=50`, `save_freq=10`).
Periodic checkpoints are saved every 10 epochs, and the best model is saved to `{m['model_alias']}_{f_suffix}_best.pt`."""
        ))

        train_cmd = (
f"""# Train Cell ({f_pct} labels, {n_vols} volumes)
frac_dir = output_dir / 'frac_{f_suffix}'
frac_dir.mkdir(parents=True, exist_ok=True)
print(f'Training {f_pct} in: {{frac_dir}}')

!python {m['train_script']} \\
    --data_dir "{{data_dir}}" \\
    --output_dir "{{frac_dir}}" \\
    --fraction {f_float} \\
    --seed {{SEED}} \\
    --epochs {{MAX_EPOCHS}} \\
    --patience {{PATIENCE}} \\
    --min_epochs {{MIN_EPOCHS}} \\
    --min_steps {{MIN_STEPS}} \\
    --save_freq {{SAVE_FREQ}} \\
    --ckpt_prefix "{{MODEL_ALIAS}}_{f_suffix}" \\
    --batch_size {{BATCH_SIZE}} \\
    --num_workers {{NUM_WORKERS}} \\
    {m['train_flags']} \\
    --amp"""
        )
        cells.append(make_code_cell(train_cmd))

        # Eval Section
        cells.append(make_md_cell(
f"""### {sec_num}.2 Held-Out Test Evaluation Cell ({f_pct})
Evaluates the saved `{m['model_alias']}_{f_suffix}_best.pt` across all **242 held-out test volumes**."""
        ))

        eval_cmd = (
f"""# Eval Cell ({f_pct} best checkpoint -> held-out test split)
frac_dir = output_dir / 'frac_{f_suffix}'
eval_dir = frac_dir / 'eval'
ckpt_path = frac_dir / 'checkpoints' / f'{{MODEL_ALIAS}}_{f_suffix}_best.pt'
assert ckpt_path.exists(), f'Checkpoint not found: {{ckpt_path}}'

print(f'Evaluating checkpoint: {{ckpt_path}}')
!python scripts/evaluate_3d.py \\
    --data_dir "{{data_dir}}" \\
    --output_dir "{{eval_dir}}" \\
    --checkpoint "{{ckpt_path}}" \\
    {m['eval_flags']} \\
    --batch_size 1 \\
    --num_workers {{NUM_WORKERS}} \\
    --amp"""
        )
        cells.append(make_code_cell(eval_cmd))
        sec_num += 1

    # Summary & Curve Plot Cell
    cells.append(make_md_cell(
f"""## {sec_num}. Master Table 2 Summary & Efficiency Curve
Compiles held-out test metrics across all fractions into the canonical Table 2 format, saves `low_data_3d_summary.csv` and `.md`, and renders the efficiency curve."""
    ))
    cells.append(make_code_cell(
"""import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

fractions_info = [
    ('0_01', '1%', 12),
    ('0_05', '5%', 58),
    ('0_10', '10%', 115),
    ('0_25', '25%', 286),
    ('0_50', '50%', 572),
]

summary_rows = []
for frac_suffix, pct_label, n_vols in fractions_info:
    eval_csv = output_dir / f'frac_{frac_suffix}' / 'eval' / 'metrics' / 'benchmark_3d_summary.csv'
    if eval_csv.exists():
        df_eval = pd.read_csv(eval_csv)
        row_dict = {
            'Fraction': pct_label,
            'Volumes': n_vols,
            'WT Dice (%)': 'n/a',
            'TC Dice (%)': 'n/a',
            'ET Dice (%)': 'n/a',
            'Mean Dice (%)': 'n/a',
            'Mean HD95 (voxels)': 'n/a',
        }
        dice_vals = []
        hd95_vals = []
        for rname in ['WT', 'TC', 'ET']:
            sub = df_eval[df_eval['Model'].str.contains(f'\\\\[{rname}\\\\]', na=False)]
            if not sub.empty:
                d_str = str(sub.iloc[0]['Dice (%)'])
                row_dict[f'{rname} Dice (%)'] = d_str
                try:
                    val = float(d_str.split('±')[0].strip())
                    dice_vals.append(val)
                except Exception:
                    pass
                if 'HD95 (voxels)' in sub.columns:
                    h_str = str(sub.iloc[0]['HD95 (voxels)'])
                    try:
                        h_val = float(h_str.split('±')[0].strip())
                        hd95_vals.append(h_val)
                    except Exception:
                        pass
        if dice_vals:
            row_dict['Mean Dice (%)'] = f'{sum(dice_vals) / len(dice_vals):.2f}'
        if hd95_vals:
            row_dict['Mean HD95 (voxels)'] = f'{sum(hd95_vals) / len(hd95_vals):.2f}'
        summary_rows.append(row_dict)

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    metrics_dir = output_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = metrics_dir / 'low_data_3d_summary.csv'
    summary_md = metrics_dir / 'low_data_3d_summary.md'
    summary_df.to_csv(summary_csv, index=False)
    try:
        summary_df.to_markdown(summary_md, index=False)
    except Exception:
        with open(summary_md, 'w') as f:
            f.write(summary_df.to_string(index=False))

    print('\\n' + '=' * 70)
    print(f'📊 LOW-DATA LABEL EFFICIENCY SUMMARY: {MODEL_ALIAS}')
    print('=' * 70)
    print(summary_df.to_string(index=False))
    print('=' * 70)
    print(f'✓ Saved summary CSV to: {summary_csv}')
    print(f'✓ Saved summary Markdown to: {summary_md}')

    valid_pts = []
    for _, r in summary_df.iterrows():
        try:
            m_val = float(r['Mean Dice (%)'])
            valid_pts.append((r['Fraction'], r['Volumes'], m_val))
        except (ValueError, TypeError):
            pass
    if valid_pts:
        labels, vols, y_vals = zip(*valid_pts)
        plt.figure(figsize=(8, 5))
        plt.plot(labels, y_vals, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label=MODEL_ALIAS)
        for lbl, y in zip(labels, y_vals):
            plt.annotate(f'{y:.2f}%', (lbl, y), textcoords='offset points', xytext=(0, 10), ha='center', fontweight='bold')
        plt.title(f'Label Efficiency: {MODEL_ALIAS} (BraTS 2024 Held-Out Test)', fontsize=13, fontweight='bold')
        plt.xlabel('Annotation Fraction (Patient Volume Budget)', fontsize=11)
        plt.ylabel('Macro Test Dice (%) [WT/TC/ET]', fontsize=11)
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plot_path = metrics_dir / 'low_data_curve.png'
        plt.savefig(plot_path, dpi=300)
        print(f'✓ Saved efficiency curve plot to: {plot_path}')
        plt.show()
else:
    print('⚠️ No evaluation results found yet. Execute the Train and Eval cells above!')"""
    ))
    sec_num += 1

    # Zip Package Cell
    cells.append(make_md_cell(
f"""## {sec_num}. 1-Click Package & Export to Google Drive
Archives all fraction checkpoints, logs, and evaluation reports into a standalone zip file in Google Drive."""
    ))
    cells.append(make_code_cell(
f"""import shutil
from pathlib import Path

zip_path = Path('/content/drive/MyDrive/thesis/low_label') / ZIP_NAME
zip_path.parent.mkdir(parents=True, exist_ok=True)

print(f'Creating archive {{zip_path}} from {{output_dir}} ...')
shutil.make_archive(str(zip_path.with_suffix('')), 'zip', output_dir)
print(f'✓ Export completed: {{zip_path}} ({{zip_path.stat().st_size / 1e6:.2f}} MB)')"""
    ))

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "A100",
                "provenance": []
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    return nb


def main():
    print(f"Generating 8 low-label notebooks in {OUTPUT_DIR} ...")
    for m in MODELS:
        nb = build_notebook(m)
        nb_path = OUTPUT_DIR / m["filename"]
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        print(f"✓ Generated {m['filename']} ({len(nb['cells'])} cells)")
    print("\n✓ Successfully generated all 8 parallel low-label Colab notebooks!")


if __name__ == "__main__":
    main()
