# Documentation Directory

Welcome to the `thesis_3d` documentation suite. The documentation is organized into four authoritative, focused guides:

| Document | Title & Focus | Scope |
| :--- | :--- | :--- |
| [**`01_visreg_jepa_architecture_and_pipeline.md`**](01_visreg_jepa_architecture_and_pipeline.md) | **Architecture & End-to-End Pipeline** | Full dataflow: preprocessing, $16^3$ patch embedding, masking geometry, 8-layer ViT encoder, 4-layer predictor, VisReg/SigReg loss formulations, and downstream decoders (FPN & UNETR). |
| [**`02_forensic_audit_and_tier2_upgrades.md`**](02_forensic_audit_and_tier2_upgrades.md) | **Forensic Audit & Tier 2 Upgrades** | Audit remediations (0% longitudinal patient leakage with grouped `base_id`, composite stratification, deep supervision parity, Simpson's paradox fix) and algorithmic upgrades (Option B 3D target jitter, FIFO token memory queue, cosine regularizer annealing). |
| [**`03_scientific_rationales_and_benchmarks.md`**](03_scientific_rationales_and_benchmarks.md) | **Scientific Rationales & Benchmarks** | Theoretical justifications ([`AGENTS.md`](../AGENTS.md)): brain-aware masking & air-spike rejection, HD95 validation decoupling, full fine-tuning vs. frozen probing, benchmark fairness, and official 6-model multi-region (WT/TC/ET) findings. |
| [**`04_cloud_and_local_execution_guide.md`**](04_cloud_and_local_execution_guide.md) | **Cloud & Local Execution Guide** | Operational walkthrough for running pre-training and downstream segmentation on Kaggle GPU (T4/P100) and Google Colab Pro (A100), plus automated 1-click paper benchmark aggregation. |

---

*Note: For scientific research guidelines and paper critique standards, refer to [`AGENTS.md`](../AGENTS.md) at the repository root.*
