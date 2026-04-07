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

### WebVid-CoVR
The primary dataset for training and evaluation.

```bash
# Download annotations
bash tools/scripts/download_annotation.sh covr

# Download videos (requires mpi4py)
conda install -c conda-forge mpi4py
ln -s /path/to/your/datasets/folder datasets
python tools/scripts/download_covr.py --split=train  # or val, test
```

### Zero-shot Evaluation Datasets

**CIRR** (Composed Image Retrieval on Real-life images):
```bash
bash tools/scripts/download_annotation.sh cirr
# Follow instructions at https://github.com/lil-lab/nlvr/tree/master/nlvr2#direct-image-download for images
```

**FashionIQ** (Fashion Image Retrieval and Query modification):
```bash
bash tools/scripts/download_annotation.sh fiq
# Download images from https://github.com/hongwang600/fashion-iq-metadata/tree/master/image_url
```

**CIRCO** (Composed Image Retrieval for Complex Objects):
Follow the structure in [CIRCO repository](https://github.com/miccunifi/CIRCO.git).

## 🚀 Usage

<!-- 
TODO: Add usage instructions
This section is intentionally left blank for the author to complete.
Please add training and evaluation commands after finalizing the implementation.
-->

## 📈 Results

### WebVid-CoVR Test Set

| Method | R@1 | R@5 | R@10 | R@50 |
|--------|-----|-----|------|------|
| CoVR-BLIP | 53.13 | 79.93 | 86.85 | 97.69 |
| CoVR-ECDE | 60.12 | 84.32 | 91.27 | 98.72 |
| CoVR-BLIP2 | 59.82 | 83.84 | 91.28 | 98.24 |
| FDCA | 54.80 | 82.27 | 89.84 | 97.70 |
| HUD | 63.38 | 86.93 | 92.29 | 98.76 |
| **Q-SlotSelect (Ours)** | **63.72** | **86.76** | **92.68** | **98.92** |

**Improvement**: +3.9% R@1 over CoVR-BLIP2 baseline

### FashionIQ (Zero-shot)

| Category | R@10 | R@50 |
|----------|------|------|
| Dress | 50.62 | 72.63 |
| Shirt | 55.00 | 73.31 |
| Toptee | 57.06 | 77.77 |
| **Average** | **54.23** | **74.57** |

**Improvement**: +5.37% Average R@10 over CoVR-BLIP2 baseline (48.86 → 54.23)

### CIRR (Zero-shot)

| Recall@K | R@1 | R@5 | R@10 | R@50 |
|----------|-----|-----|------|------|
| Full Set | 52.10 | 81.78 | 89.61 | 97.93 |

| R(subset)@K | R@1 | R@2 | R@3 |
|-------------|-----|-----|-----|
| Subset | 78.79 | 91.16 | 96.48 |

**Improvement**: +1.23% R@1 and +2.09% R(subset)@1 over CoVR-BLIP2 baseline

## 🔬 Key Findings

1. **In-domain Performance**: Query-guided specialization significantly improves retrieval accuracy on training domain (WebVid-CoVR).

2. **Cross-domain Trade-off**: Enhanced in-domain accuracy may incur trade-offs in cross-domain zero-shot transferability.

3. **Interpretability**: Attention visualization reveals that Q-SlotSelect effectively highlights query-relevant semantic slots while suppressing noise.

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
