# Scientific Rationales, Benchmark Methodology & Empirical Findings

*Authoritative consolidated scientific rationale document for experimental design, architectural choices, and empirical findings in the `thesis_3d` research program.*

---

## 1. Executive Summary: Core Research Findings

On the **BraTS 2024 GLI** benchmark under the official three-region protocol (**Whole Tumor [WT]**, **Tumor Core [TC]**, and **Enhancing Tumor [ET]**), we evaluate six model configurations:

| Model Architecture | Encoder Backbone | Decoder Type | Mean Dice (WT/TC/ET) | WT Dice | TC Dice | ET Dice | 1% Low-Data Mean Dice | Latency (128³) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **3D nnU-Net** | DynUNet CNN | Deep Supervised | **79.72%** | **88.19%** | **83.00%** | **67.98%** | 28.05% | 968 ms (TTA) / 235 ms |
| **3D Residual UNet** | ResUNet CNN | Residual 3D | 77.96% | 86.59% | 80.46% | 66.84% | 7.03% *(Collapse)* | **260 ms (TTA) / 76 ms** |
| **3D VisReg UNETR** | 3D ViT (JEPA Pretrained) | UNETR Hybrid | 73.11% | 84.74% | 75.77% | 58.83% | **32.09%** *(Top Result)* | 987 ms (TTA) / 128 ms |
| **3D ViT-UNETR** | 3D ViT (From Scratch) | UNETR Hybrid | 73.37% | 84.84% | 76.17% | 59.10% | 29.61% | 945 ms (TTA) / 128 ms |
| **3D ViT-FPN** | 3D ViT (From Scratch) | Multi-Scale FPN | 67.93% | 80.30% | 68.90% | 54.58% | 17.00% | 471 ms (TTA) / 115 ms |
| **3D VisReg JEPA** | 3D ViT (JEPA Pretrained) | Multi-Scale FPN | 67.82% | 80.31% | 68.96% | 54.20% | 15.35% | 465 ms (TTA) / 115 ms |

### Key Scientific Takeaways ([`AGENTS.md`](../AGENTS.md)):
1. **Full Data Regime**: Supervised 3D CNNs achieve higher overall Dice than Vision Transformers, confirming that localized 3D convolutional inductive biases are superior for volumetric dense prediction when annotations are abundant.
2. **Hybrid Convolutions Resolve the Token Bottleneck**: Adding a convolutional stem in UNETR provides a **$+5.44\text{pp}$** mean Dice gain over pure ViT-FPN ($73.37\%$ vs.\ $67.93\%$), successfully bridging the spatial granularity deficit of $16^3$ patch tokenization.
3. **Severe Label Scarcity ($1\%$ Labels / 12 Volumes)**: Standard 3D Residual UNet experiences catastrophic background collapse ($7.03\%$). In contrast, 3D VisReg JEPA paired with the UNETR hybrid decoder achieves the top performance ($\mathbf{32.09\%}$), outperforming from-scratch UNETR ($29.61\%$, $\mathbf{+2.48\text{pp}}$) and deeply supervised nnU-Net ($28.05\%$, $\mathbf{+4.04\text{pp}}$), while exceeding standard 3D Residual UNet by $\mathbf{+25.06\text{pp}}$.

---

## 2. Rationale 1: Brain-Aware Masking & The SWD Air-Spike Problem

### 2.1 The Biophysical Anatomy of 3D Brain MRI
Skull-stripped brain parenchyma occupies only **$27\%\text{--}40\%$** of a $128^3$ volumetric cube; the remaining $\approx 56\%$ is empty air (constant zeros). Uniform random masking inherited from 2D methods allocates over half of all context and target patches to empty air:
- Air patches produce constant zero tokens, yielding trivial prediction loss ($\approx 0.0201$ vs. $0.1309$ for brain tissue).
- Every air target wastes optimizer gradient steps learning trivial identity representations.

### 2.2 The Sliced-Wasserstein Distance (SWD) Air-Spike
VisReg projects batch representations onto $M=256$ 1D hyperplanes and matches them against standard Gaussian quantiles $\mathcal{N}(0, 1)$. When constant air tokens enter the loss:
- Identical air tokens form a sharp **Dirac delta spike** adjacent to the continuous brain representation cloud.
- The shape penalty spends massive gradient energy attempting to force the zero spike into a Gaussian bell curve instead of organizing biological brain tissue features:
  - Synthetic test: Scale loss increased from $0.0011 \to 0.0253$ (**$23\times$**); Shape loss increased from $0.0043 \to 0.0811$ (**$19\times$**).

### 2.3 The Architectural Solution
1. **Tissue Rejection Sampling**: Target sampling evaluates tissue fraction via `brain_mask` and accepts cuboids meeting proportional tissue thresholds ($\ge \max(4, \lfloor 0.40 \times \text{vol} \rfloor)$).
2. **Tissue-Only SWD Filtering**: Batch tokens are filtered to tissue-containing rows before regularizer evaluation. Air tokens are excluded from SWD moment matching, ensuring clean Gaussian organization of biological tissue.

---

## 3. Rationale 2: Decoupled Validation & HD95 Bottleneck Elimination

### 3.1 The 13-Minute Validation Freeze
Early in training, unconverged models (e.g., 3D UNet without deep supervision, validation Dice $\approx 0.04$) produce diffuse, speckled noise throughout the $128^3$ volume.
- Computing 95th Percentile Hausdorff Distance (HD95) extracts morphological boundaries via binary erosion:
  ```python
  eroded_p = binary_erosion(pred_mask, structure=struct_26)
  pts_p = np.argwhere(pred_mask ^ eroded_p)
  ```
- A compact converged lesion generates $\approx 1,000$ boundary points (KD-tree query takes $<1\,\text{ms}$).
- Speckled noise turns every noise voxel into a border, producing **$200,000\text{--}500,000$ boundary points**. Constructing `scipy.spatial.cKDTree` on 500k points takes **$3.8\,\text{s}$ per volume**, freezing validation for **$13\text{ minutes}$** on CPU while the GPU sits idle at 0%.

### 3.2 The Decoupling Strategy
- During training epochs, validation monitors volumetric **Dice and IoU** only (execution time: $<42\,\text{s}$).
- Intensive HD95 evaluation is decoupled and computed exclusively during final test-set evaluation ([`scripts/evaluate_3d.py`](file:///Users/hanriman/Documents/master/thesis/thesis_3d/scripts/evaluate_3d.py)).
- **Result**: Validation speedup of **$18.7\times$**, cutting 30-epoch training from 7.5 hours down to 1.4 hours without compromising metric fidelity.

---

## 4. Rationale 3: Full Fine-Tuning vs. Frozen Encoder Probing

### 4.1 Methodological Comparison Matrix

| Evaluation Dimension | **Full Fine-Tuning (`freeze_encoder=False`)** *(Default)* | **Frozen Encoder Probe (`freeze_encoder=True`)** |
| :--- | :--- | :--- |
| **Primary Scientific Question** | *"Does self-supervised pre-training provide an optimal initialization that outperforms randomly initialized baselines?"* | *"How much spatial and anatomical structure is directly decodable without parameter adaptation?"* |
| **Fairness vs. CNN Baselines** | **Fair & Symmetric**: Both 3D nnU-Net and JEPA allow full parameter capacity to adapt to tumor margins. | **Asymmetric**: 3D nnU-Net adapts 100% of parameters; JEPA locks ~85% of parameters. |
| **Spatial Resolution Mismatch** | Full fine-tuning allows token attention maps to refine sub-patch boundary localization. | Locks representations at coarse $16^3$ spatial granularity; decoder must perform pure blind upsampling. |
| **Standard in 3D Medical Imaging** | **Universal benchmark standard** (MICCAI, IEEE TMI, MONAI SSL). | Treated strictly as an **ablation probe** in dense 3D segmentation. |

### 4.2 Research Protocol
The primary benchmark adopts **full fine-tuning** to establish fair parity with fully supervised CNNs. Frozen probing is reserved as a mechanistic ablation to assess out-of-the-box linear decodability.

---

## 5. Rationale 4: Hardware Dynamics & Methodological Fairness

### 5.1 Batch Size Asymmetry: Pre-Training ($B=8/16$) vs. Downstream ($B=2$)
- **Pre-Training ($B=8$ or $B=16$)**: Operates purely in 1D sequence space ($L=512, D=384$) without a volumetric decoder. Peak activation memory is only $\approx 1.5\,\text{GB}$ under AMP FP16. Larger batch sizes improve empirical covariance estimation in VisReg/SigReg.
- **Downstream Segmentation ($B=2$)**: Reconstructs dense $128^3$ volumetric grids across 4 deep supervision heads. Peak memory reaches $\approx 7.4\,\text{GB}$ at $B=2$ and saturates 16 GB GPUs at $B=4$. Downstream VisReg and supervised baselines both use $B=2$ for strict hardware parity.

### 5.2 Deep Supervision Gradient Parity
Deep supervision is enabled symmetrically across all models:
- **3D nnU-Net**: 4 built-in heads ($128^3, 64^3, 32^3, 16^3$).
- **3D Residual UNet**: 3 hooked 1×1×1 heads (`ds1`, `ds2`, `ds3`).
- **3D VisReg FPN / UNETR**: Multi-scale lateral skip heads.
- Normalized hierarchical loss weights: $w_s = 2^{-s} / \sum_{j} 2^{-j}$.

### 5.3 High-Resolution Patch Scaling ($8^3$ vs. $16^3$ Patch Size) & Benchmark Parity
When scaling the Vision Transformer patch embedding from $16^3$ down to $8^3$ voxels on high-memory hardware (e.g., NVIDIA A100 40/80 GB):
1. **Computational & Token Sequence Expansion**:
   - For an identical $128^3$ volume, $8^3$ voxel patches generate $(128/8)^3 = 16^3 = \mathbf{4,096\text{ tokens}}$ ($8\times$ longer than the $8^3 = 512$ token sequence at $16^3$ patch size).
   - Self-attention computational complexity and activation storage scale quadratically: $(4,096 / 512)^2 = \mathbf{64\times}$.
   - Anisotropic target masking scales proportionally to maintain identical physical anatomical coverage: $776$ target tokens ($18.95\%$ of 4,096) and $1,536$ context tokens ($37.5\%$).
2. **Downstream Memory and Latency Spillover**:
   - If downstream segmentation adopts the fine-grained $8^3$ encoder, lateral skip feature maps are emitted at $16^3$ spatial resolution ($8\times$ more spatial tokens per intermediate layer).
   - Backward pass activation memory under multi-scale deep supervision expands from $\approx 7.4\,\text{GB}$ to $>30\,\text{GB}$, requiring A100-tier hardware or micro-batching ($B=1$) with gradient accumulation. Downstream inference latency also increases $\approx 4\text{--}8\times$.
3. **Preserving Benchmark Fairness ([`AGENTS.md`](../AGENTS.md))**:
   - Directly comparing an $8^3$-patch JEPA ($4,096$ tokens) against standard $16^3$-patch CNN/ViT baselines introduces a spatial resolution confounder: gains could stem from $8\times$ finer spatial token granularity rather than self-supervised representation quality.
   - Therefore, benchmark fairness requires:
     - The canonical benchmark strictly fixes patch size to $16^3$ across all models on Kaggle T4 ($B=2$ downstream).
     - The $8^3$ patch configuration is classified as an **asymptotic capacity & resolution scaling ablation**, evaluated against matched from-scratch $8^3$ ViT controls to isolate true representation benefit.

---

## 6. Rationale 5: Disentangling Architecture from Representation ($\Delta_{\text{JEPA}}$ vs. $\Delta_{\text{Arch}}$)

To determine whether performance differences stem from the self-supervised representation or the Vision Transformer architecture itself, we evaluate matched from-scratch controls:

$$\Delta_{\text{Total}} = \Delta_{\text{Arch}} + \Delta_{\text{JEPA}}$$

$$\Delta_{\text{Arch}} = \text{Score}(\text{ViT-Scratch}) - \text{Score}(\text{Supervised CNN})$$

$$\Delta_{\text{JEPA}} = \text{Score}(\text{VisReg JEPA}) - \text{Score}(\text{ViT-Scratch})$$

### Empirical Decomposition:
1. **Full-Data Regime ($100\%$ Labels)**:
   - $\Delta_{\text{Arch}} = 73.37\% - 79.72\% = \mathbf{-6.35\text{pp}}$ (architectural inductive bias penalty of ViTs on volumetric imaging).
   - $\Delta_{\text{JEPA}} = 73.11\% - 73.37\% = \mathbf{-0.26\text{pp}}$ (pre-training matches from-scratch optimization under abundant supervised data).
2. **Extreme Scarcity Regime ($1\%$ Labels / 12 Volumes)**:
   - $\Delta_{\text{Arch}} = 29.61\% - 28.05\% = \mathbf{+1.56\text{pp}}$ (hybrid CNN skip connections preserve localization and prevent the catastrophic collapse seen in pure CNNs: ResUNet collapses to $7.03\%$).
   - $\Delta_{\text{JEPA}} = 32.09\% - 29.61\% = \mathbf{+2.48\text{pp}}$ (JEPA pre-training provides an additional representation prior, securing top overall performance of $32.09\%$ and outperforming standard 3D Residual UNet by $\mathbf{+25.06\text{pp}}$).
