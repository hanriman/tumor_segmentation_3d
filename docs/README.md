# Documentation Directory

Welcome to the `thesis_3d` documentation suite. The documentation is organized into four authoritative, focused guides:

| Document | Title & Focus | Scope |
| :--- | :--- | :--- |
| Document | Title & Focus | Scope |
| :--- | :--- | :--- |
| [**`01_visreg_jepa_architecture_and_pipeline.md`**](01_visreg_jepa_architecture_and_pipeline.md) | **Architecture & End-to-End Pipeline** | Full dataflow: preprocessing, canonical $8^3$ (4,096 tokens) & lightweight $16^3$ patch embedding, masking geometry (anisotropic target jitter: 776 tokens on $16^3$ grid / 97 tokens on $8^3$ grid across 6 cuboids), 8-layer ViT encoder, 4-layer predictor, dual cosine schedules (LR linear warmup + regularizer annealing), VisReg/SigReg loss formulations, and scale-adaptive downstream decoders (FPN & UNETR). |
| [**`02_forensic_audit_and_tier2_upgrades.md`**](02_forensic_audit_and_tier2_upgrades.md) | **Forensic Audit & Upgrades** | Complete three-phase audit & remediation log: Phase 1 (0% patient leakage with grouped `base_id`, composite stratification, deep supervision parity, Simpson's paradox fix), Phase 2 (anisotropic 3D target jitter, dense token regularization, dual cosine schedules), and Phase 3 (patch size alignment to $8^3$, hybrid probing disentanglement, missing modality energy scaling, tissue-filtered effective rank, test output isolation). Covers all 169 passed regression tests. |
| [**`03_scientific_rationales_and_benchmarks.md`**](03_scientific_rationales_and_benchmarks.md) | **Scientific Rationales & Benchmarks** | Theoretical justifications ([`AGENTS.md`](../AGENTS.md)): brain-aware masking & air-spike rejection, HD95 validation decoupling, full fine-tuning vs. pure latent probing (ViT-FPN) vs. frozen hybrid probing (ViT-UNETR), canonical $8^3$ patch scaling, missing modality energy analysis ($4\times$ inverted dropout deficit), and official multi-region (WT/TC/ET) findings. |
| [**`04_cloud_and_local_execution_guide.md`**](04_cloud_and_local_execution_guide.md) | **Cloud & Local Execution Guide** | Operational walkthrough for running pre-training and downstream segmentation on Kaggle GPU (T4/P100) and Google Colab (A100 SXM4 suite with `thesis/` Drive hierarchy, fast local NVMe extraction & Drive mirroring), `--rescale_missing_modalities` CLI usage, plus automated 1-click paper benchmark aggregation. |

---

*Note: For scientific research guidelines and paper critique standards, refer to [`AGENTS.md`](../AGENTS.md) at the repository root.*
