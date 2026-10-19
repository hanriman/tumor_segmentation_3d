# Documentation Directory

Welcome to the `thesis_3d` documentation suite. The documentation is organized into four authoritative, focused guides:

| Document | Title & Focus | Scope |
| :--- | :--- | :--- |
| [**`01_visreg_jepa_architecture_and_pipeline.md`**](01_visreg_jepa_architecture_and_pipeline.md) | **Architecture & End-to-End Pipeline** | Full dataflow: preprocessing, canonical $8^3$ ($4,096$ tokens) & lightweight $16^3$ patch embedding, masking geometry (anisotropic target jitter: 776 tokens on $16^3$ grid / 97 tokens on $8^3$ grid across 6 cuboids), 8-layer ViT encoder, 4-layer predictor, unified `SingleEncoderJEPA3D` with direct encoder-space VISReg/SigReg regularization ($M=1024$ rays), and scale-adaptive downstream decoders (FPN & UNETR). |
| [**`02_forensic_audit_and_tier2_upgrades.md`**](02_forensic_audit_and_tier2_upgrades.md) | **Forensic Audit & Architectural Upgrades** | Complete audit & remediation log: patient-grouped splits, composite stratification, deep supervision parity, anisotropic 3D target jitter, paper-faithful direct encoder-space VISReg (`arXiv:2602.11401`) & LeJEPA/SigReg (`arXiv:2511.08544`) regularization, zero-imputed modality dropout parity (`rescale=False`), validation-selected low-data evaluation with `--min_steps 250`, and central `builder.py` / `supervised_trainer.py` unification (172 passed tests). |
| [**`03_scientific_rationales_and_benchmarks.md`**](03_scientific_rationales_and_benchmarks.md) | **Scientific Rationales & Benchmarks** | Theoretical justifications ([`AGENTS.md`](../AGENTS.md)): brain-aware masking & air-spike rejection, HD95 validation decoupling, full fine-tuning vs. pure latent probing (ViT-FPN) vs. frozen hybrid probing (ViT-UNETR), canonical $8^3$ patch scaling, missing-modality analysis, and empirical multi-region (WT/TC/ET) findings. |
| [**`04_cloud_and_local_execution_guide.md`**](04_cloud_and_local_execution_guide.md) | **Cloud & Local Execution Guide** | Operational walkthrough for running pre-training and downstream segmentation on Google Colab (`notebooks/colab/01..09_*.ipynb` with `My Drive/thesis/` hierarchy, fast local NVMe extraction & Drive mirroring), local CLI execution, and automated 1-click paper benchmark aggregation. |

---

*Note: For scientific research guidelines and paper critique standards, refer to [`AGENTS.md`](../AGENTS.md) at the repository root.*
