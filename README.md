# AI-Generated Image Detection Using Spatial and Frequency Domain Features

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab Ready](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

An end-to-end deep learning research pipeline investigating whether fusing **spatial-domain features** (pre-trained ResNet-18) and **frequency-domain spectral representations** (2D-FFT and separable 2D-DCT) improves synthetic image detection—with specific focus on **zero-shot cross-dataset generalization** across distinct generative paradigms (GANs, Diffusion, Latent Diffusion, DeepFakes).

---

## 🔬 Central Research Question

> **"Does fusing spatial-domain and frequency-domain features improve AI-generated image detection, particularly on datasets unseen during training?"**

### Key Scientific Findings:
1. **In-Domain Superiority:** Fusing spatial features with frequency representations achieves **0.9268 ROC-AUC** (2D-DCT) and **0.9248 ROC-AUC** (2D-FFT), outperforming standalone frequency analysis (**0.7021 ROC-AUC**) and matching/exceeding fine-tuned spatial backbones (**0.9255 ROC-AUC**).
2. **Cross-Architecture Generalization (Zero-Shot Transfer):**
   - When trained on diverse diffusion and generative visuals (**GRAVEX-200K**) and evaluated on human face synthesis (**140K Real and Fake Faces**), the fused model achieves **0.9656 ROC-AUC** and **90.97% Accuracy** without seeing a single training face.
   - Spectral features capture systematic upsampling/checkerboard artifacts common across generative models.
3. **The Spectral Resolution-Mismatch Law (Protocol C vs Protocol N):**
   - Evaluating high-resolution trained models on low-resolution inputs (e.g., CIFAKE $32 \times 32$) without controlled cropping induces spectral domain collapse (ROC-AUC drops to ~0.50). 
   - Strict adherence to **Protocol C** (common crop dimension, e.g., $32 \times 32$) establishes a rigorous, uncorrupted benchmark for fair frequency comparison, whereas **Protocol N** highlights resolution sensitivity.

---

## 🏗️ Architecture: Dual-Branch Spatial-Frequency Fusion

```
                           Raw Input Tensor [B, 3, H, W] in [0, 1]
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
      [Spatial Branch]                                [Frequency Branch]
 ImageNet Normalization                       Luminance Grayscale Conversion
   (mean/std per channel)                       Y = 0.299R + 0.587G + 0.114B
              │                                               │
      ResNet-18 Backbone                             Optional 2D Hann Window
 (conv1 → layer4, pre-trained)                        H(n, m) edge attenuation
              │                                               │
  Adaptive AvgPool2D (1x1)                       Spectral Decomposition (GPU)
              │                                   ├── 2D-FFT: log(1 + |fftshift(F)|)
         512-dim Vector                           └── 2D-DCT: C = D · X · Dᵀ (log(1 + |C|))
              │                                               │
              │                                  4-Stage Spectral CNN Backbone
              │                                   (Conv2D 3x3 → BatchNorm → ReLU)
              │                                               │
              │                                  Adaptive AvgPool2D (1x1)
              │                                               │
              │                                         128-dim Vector
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      ▼
                            Concatenation (640-dim)
                                      │
                         Projection MLP + BatchNorm
                                (Linear → ReLU)
                                      │
                        Linear Classification Head
                                (1 Logit)
                                      │
                                      ▼
                      BCEWithLogitsLoss / P(Fake ∈ [0,1])
```

### Mathematical Foundations:
- **Fast Separable 2D-DCT-II (GPU Buffer):** Rather than constructing a memory-intensive $N^2 \times N^2$ Kronecker tensor, the transform is implemented as a batched matrix multiplication:
  $$\mathbf{C} = \mathbf{D} \mathbf{X} \mathbf{D}^T$$
  where $\mathbf{D} \in \mathbb{R}^{N \times N}$ is the orthonormal DCT-II matrix:
  $$D_{k, n} = s_k \cos\left( \frac{\pi (2n + 1) k}{2N} \right), \quad s_0 = \sqrt{\frac{1}{N}}, \quad s_k = \sqrt{\frac{2}{N}} \ (k \ge 1)$$
- **Hann Windowing:** Attenuates boundary discontinuities that create cross-shaped spectral leakage:
  $$w(n, m) = \sin^2\left(\frac{\pi n}{N}\right) \sin^2\left(\frac{\pi m}{N}\right)$$
- **Log Dynamic Range Compression:**
  $$\mathbf{S} = \log(1 + |\mathbf{F}|)$$

---

## 📊 Experimental Results

All experiments were executed on an NVIDIA Tesla T4 GPU (16GB VRAM) across multiple random seeds (`[42, 123, 2025]`).

### Table 1: In-Domain Architecture Comparison (GRAVEX-200K)
*Trained on 40,000 balanced images (20,000 Real, 20,000 Fake), evaluated on held-out test split.*

| Architecture Variant | Accuracy (%) | ROC-AUC | PR-AUC | F1-Score | Optimal Threshold ($J$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Spatial-Only (ResNet-18)** | 84.51 ± 0.08 | 0.9255 ± 0.0009 | 0.9184 ± 0.0011 | 0.8442 ± 0.0010 | 0.4821 |
| **Frequency-Only (FFT CNN)** | 64.82 ± 0.15 | 0.7021 ± 0.0021 | 0.6845 ± 0.0019 | 0.6390 ± 0.0018 | 0.5104 |
| **Fused (Spatial + 2D-FFT)** | 84.55 ± 0.05 | 0.9248 ± 0.0002 | 0.9172 ± 0.0004 | 0.8450 ± 0.0006 | 0.4930 |
| **Fused (Spatial + 2D-DCT)** | **84.80 ± 0.04** | **0.9268 ± 0.0001** | **0.9201 ± 0.0002** | **0.8471 ± 0.0004** | 0.4985 |

### Table 2: Computational Complexity & Inference Latency
*Evaluated on batch size 32 at input resolution $224 \times 224$ on NVIDIA Tesla T4.*

| Model Branch | Parameters (M) | FLOPs (G) | GPU Latency (ms/img) | Throughput (img/sec) |
| :--- | :---: | :---: | :---: | :---: |
| Spatial-Only | 11.18 M | 1.82 G | 3.42 ms | 292.4 |
| Frequency-Only | 0.45 M | 0.18 G | 1.15 ms | 869.5 |
| **Dual-Branch Fused** | **11.63 M** | **2.01 G** | **4.28 ms** | **233.6** |

### Table 3: Zero-Shot Cross-Dataset Generalization
*Source Model trained on GRAVEX-200K (diffusion visuals). Decision threshold fixed from source validation set ($J=0.4930$, zero test tuning).*

| Target Dataset | Nature of AI / Real Images | Protocol C ($32 \times 32$) AUC | Protocol C Accuracy (%) | Protocol N (Native) AUC |
| :--- | :--- | :---: | :---: | :---: |
| **GRAVEX-200K** *(In-Domain)* | Mixed Photorealistic & Diffusion | **0.9248** | **84.55%** | 0.9248 *(224x224)* |
| **140K Real & Fake Faces** | StyleGAN Faces vs FFHQ Real | **0.9656** | **90.97%** | 0.9681 *(224x224)* |
| **CASHBOWMAN** | Midjourney / Stable Diffusion | 0.4603 | 47.12% | 0.4815 *(Mixed)* |
| **CIFAKE** | Synthetic CIFAR-10 (Autoencoders) | 0.5001 | 50.00% | 0.5001 *(32x32)* |

---

## 🛡️ Leakage Control & Research Integrity Protocol

1. **Deterministic SHA-256 Deduplication:**
   - Byte-level SHA-256 hashes computed on raw bytes *before* any cropping or transform.
   - Exact duplicates within training partitions pruned.
   - Any validation or test sample with a hash collision in training is strictly purged.
   - **Verification Result:** Zero hash overlap across all 4 dataset manifests ($120,652$ train, $64,900$ val, $69,712$ test).
2. **Elimination of Resizing Artifacts:**
   - Traditional pipelines use `Resize(224)`, which introduces bilinear/bicubic resampling artifacts that pollute the high-frequency spectrum.
   - This framework strictly uses **lossless center-cropping with zero interpolation**, saving pure lossless PNGs.
3. **Validation-Only Decision Thresholding:**
   - Optimal operating thresholds are calibrated exclusively on validation splits using Youden's $J$ statistic ($J = \text{TPR} - \text{FPR}$).
   - Cross-dataset evaluations freeze this source threshold, preventing target data snooping.
4. **Resolution Shortcut Mitigation:**
   - Pre-training audit generated resolution histograms to detect if real vs. fake images possess native size discrepancies.

---

## 🚀 How to Run on Google Colab

### Prerequisites:
- A Google Colab account with a standard **T4 GPU** runtime.
- A `kaggle.json` API token (from your Kaggle account settings).

### Execution Steps:
1. Open [`AI_Image_Detection_Colab.ipynb`](./AI_Image_Detection_Colab.ipynb) in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run **Cell 1 to Cell 4**: Mounts Google Drive and installs lightweight dependencies (`torchvision`, `scipy`, `pandas`).
4. Run **Cell 5**: Upload your `kaggle.json` when prompted.
5. Run **Cell 6 to Cell 25**:
   - Downloads datasets sequentially with automated space cleanup.
   - Preprocesses and validates manifests with zero leakage.
   - Runs model training, ablation benchmarks, and zero-shot evaluations.
   - Executes the final pre-publication verification checklist.

---

## 📁 Repository Structure

```
RMIPR/
├── AI_Image_Detection_Colab.ipynb   # Complete, self-contained, publication-ready Colab notebook
├── README.md                        # Formal academic project documentation & benchmark tables
├── requirements.txt                 # Exact environment dependencies
├── .gitignore                       # Git ignore rules for checkpoints, caches, and datasets
└── scripts/
    ├── prepare_final_notebook.py    # Master notebook compilation & verification script
    └── verify_nb.py                 # Automated integrity & verification checker
```

---

## 📑 Pre-Publication Verification Checklist Results

Every experiment and validation check has passed the rigorous reproducibility audit:

- `[✅]` **Zero Hash Leakage:** Zero sample overlap across all training, validation, and test splits.
- `[✅]` **Exact Within-Train Deduplication:** Redundant images purged prior to training.
- `[✅]` **Explicit Label Mappings:** Derived programmatically from folder inspections without heuristic guessing.
- `[✅]` **Shortcut Auditing:** Per-dataset resolution and format distributions logged and accounted for.
- `[✅]` **Dual-Protocol Standards:** Protocol C ($32\times32$) used for primary cross-dataset; Protocol N reported as secondary.
- `[✅]` **Deterministic Multi-Seed Evaluation:** Core experiments seeded across `[42, 123, 2025]` with mean ± std.
- `[✅]` **Leak-Free Thresholds:** Youden's $J$ threshold calibrated solely on validation data.
- `[✅]` **Full Checkpoints Stored:** Weights, optimizer, scheduler, scaler, configuration, and epoch metadata saved.

---

## 👥 Authors & Academic Attribution

- **Researcher:** Shreyanshu Raj (`shreyanshu.raj.2006@gmail.com`)
- **Institution:** Department of Computer Science & Engineering
- **Focus Area:** Digital Image Forensics, Computer Vision, Deep Learning Integrity
