<div align="center">

# Q-SlotSelect: Query-Guided Slot Selection with Top-K Semantic Filtering

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

</div>

<div align="justify">

**Q-SlotSelect** is a novel module for Composed Video Retrieval (CoVR) that addresses the semantic granularity mismatch between specific queries and broad video representations. Unlike traditional approaches that uniformly aggregate all semantic slots via mean-pooling, Q-SlotSelect enables **query-conditioned, dynamic slot selection** through cross-attention and Top-K filtering mechanisms.

**Key Innovation**: Q-SlotSelect leverages the insight that slot aggregation should not be static but actively guided by the query. It employs cross-attention where the composed query attends to all slot tokens, followed by a Top-K filtering mechanism to suppress irrelevant slots, resulting in a refined, query-specific video representation.

</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

## 🔍 Overview

Composed Video Retrieval demands fine-grained alignment between textual modification instructions and specific visual concepts within a video. State-of-the-art methods based on Q-Former architectures (e.g., BLIP-2) encode videos into semantic slots. However, traditional mean-pooling aggregation overlooks a critical issue: while each slot captures distinct visual semantics, a user's composed query typically attends to only a subset of them.

**Q-SlotSelect** bridges this gap by introducing:
- **Dynamic Aggregation**: Query-conditioned selection instead of static pooling
- **Top-K Filtering**: Hard masking of low-relevance slots based on attention scores
- **Lightweight Design**: Only three learnable projection matrices (WQ, WK, WV)

## 🏗️ Architecture

The framework consists of two core branches:

1. **Query Encoding Branch**: The reference image/video and textual modification are encoded by BLIP-2's vision encoder and Q-Former, fused into a compact query representation.

2. **Video Slot Specialization Branch**: 
   - Extract semantic slots using frozen BLIP-2
   - Compute cross-attention scores between query and slots
   - Apply Top-K masking to filter low-relevance slots
   - Generate query-specialized video representation via weighted aggregation

**Core Module** (`src/model/blip2/xpool_cross_att.py`):
- Multi-headed cross-attention between query and video slots
- Top-K selection and hard masking
- Renormalization for valid probability distribution

## 🔧 Installation

### Environment Setup

```bash
conda create --name q-slotselect python=3.10
conda activate q-slotselect
python -m pip install -r requirements.txt
```

**Requirements**: Python 3.10, PyTorch 2.4, CUDA-capable GPU recommended

## 📊 Datasets

We evaluate Q-SlotSelect on three compositional retrieval benchmarks:

### WebVid-CoVR
The primary dataset for training and evaluation. Contains ~1.6M video-text-video triplets.

```bash
# Download annotations
bash tools/scripts/download_annotation.sh covr

# Download videos (requires mpi4py)
conda install -c conda-forge mpi4py
ln -s /path/to/your/datasets/folder datasets
python tools/scripts/download_covr.py --split=train  # or val, test
```

### CIRR (Composed Image Retrieval on Real-life images)
Contains 27k real-world images with complex attribute interactions.

```bash
bash tools/scripts/download_annotation.sh cirr
# Follow instructions at https://github.com/lil-lab/nlvr/tree/master/nlvr2#direct-image-download for images
```

### FashionIQ (Fashion Image Retrieval and Query modification)
Fashion product retrieval across three categories (Dress, Shirt, Toptee).

```bash
bash tools/scripts/download_annotation.sh fiq
# Download images from https://github.com/hongwang600/fashion-iq-metadata/tree/master/image_url
```

## 🚀 Usage

<!-- 
TODO: Add usage instructions
This section is intentionally left blank for the author to complete.
Please add training and evaluation commands after finalizing the implementation.
-->

## 📈 Results

### Main Results

**Implementation Details**: We use L=32 learnable query tokens in the Q-Former. The Top-K parameter k is set to 16 for WebVid-CoVR and FashionIQ, and k=24 for CIRR based on sensitivity analysis. All models are built upon BLIP-2 with frozen ViT-L visual encoder pretrained on COCO.

### WebVid-CoVR Test Set (In-domain)

| Method | R@1 | R@5 | R@10 | R@50 |
|--------|-----|-----|------|------|
| CoVR-BLIP | 53.13 | 79.93 | 86.85 | 97.69 |
| CoVR-ECDE | 60.12 | 84.32 | 91.27 | 98.72 |
| CoVR-BLIP2 | 59.82 | 83.84 | 91.28 | 98.24 |
| FDCA | 54.80 | 82.27 | 89.84 | 97.70 |
| HUD | 63.38 | 86.93 | 92.29 | 98.76 |
| **Q-SlotSelect (Ours)** | **63.72** | **86.76** | **92.68** | **98.92** |

**Improvement**: +3.90 points R@1 over CoVR-BLIP2 baseline (59.82 → 63.72)

### FashionIQ (Fine-tuned)

Results after training on FashionIQ training set (k=16):

| Category | R@10 | R@50 |
|----------|------|------|
| Dress | 50.62 | 72.63 |
| Shirt | 55.00 | 73.31 |
| Toptee | 57.06 | 77.77 |
| **Average** | **54.23** | **74.57** |

**Improvement**: +5.37 points Average R@10 over CoVR-BLIP2 baseline without additional pretraining (48.86 → 54.23)

### CIRR (Fine-tuned)

Results after training on CIRR training set (k=24):

| Recall@K | R@1 | R@5 | R@10 | R@50 |
|----------|-----|-----|------|------|
| Full Set | 52.10 | 81.78 | 89.61 | 97.93 |

| R(subset)@K | R@1 | R@2 | R@3 |
|-------------|-----|-----|-----|
| Subset | 78.79 | 91.16 | 96.48 |

**Improvement**: +1.23 points R@1 and +2.09 points R(subset)@1 over CoVR-BLIP2 baseline without additional pretraining (50.87 → 52.10, 76.70 → 78.79)

## 🔬 Key Findings

1. **In-domain Performance**: Query-guided slot selection (k=16) achieves 63.72% R@1 on WebVid-CoVR, outperforming static mean-pooling baseline (59.82%) by 3.90 points.

2. **Dataset-Specific Training**: Models trained separately on FashionIQ and CIRR (not zero-shot transferred from WebVid-CoVR) achieve strong performance with consistent improvements over baselines.

3. **Top-K Sensitivity**: Optimal k varies by dataset—k=16 for WebVid-CoVR/FashionIQ, k=24 for CIRR—reflecting differences in semantic distribution complexity.

4. **Ablation Validation**: Dynamic query-guided pooling (QueryPool) provides substantial gains over static mean-pooling; adding Top-K hard masking (QSS) further improves performance by suppressing irrelevant slots.

5. **Interpretability**: Attention visualization confirms that Q-SlotSelect effectively highlights query-relevant semantic slots while suppressing noise from irrelevant visual content.

## 📁 Repository Structure

```
CoVR/
├── configs/              # Hydra configuration files
│   ├── data/            # Dataset configurations
│   ├── model/           # Model configurations
│   └── experiment/      # Experiment configurations
├── src/                 # Source code
│   ├── model/           # Model implementations
│   │   └── blip2/
│   │       ├── xpool_cross_att.py    # Q-SlotSelect module
│   │       └── blip2_cir.py          # Main model
│   └── datamodules/     # Data loading
├── tools/               # Utility scripts
│   ├── embs/           # Embedding computation
│   └── scripts/        # Download scripts
├── train.py            # Training script
└── test.py             # Evaluation script
```

## 📝 Citation

<!-- 
TODO: Add citation information
This section is intentionally left blank for the author to complete.
Please add proper BibTeX citation after the paper is officially published.
-->

## 🙏 Acknowledgements

This work builds upon the CoVR framework and BLIP-2 architecture. We thank the original authors for their contributions to the field of composed video retrieval.

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**Q-SlotSelect** | Chongqing University of Posts and Telecommunications

For questions or issues, please open an issue on GitHub.

</div>
