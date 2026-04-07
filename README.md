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

### Step 1: Compute BLIP-2 Embeddings

Before training, pre-compute BLIP-2 embeddings for videos and images:

```bash
# WebVid-CoVR training videos
python tools/embs/save_blip2_embs_vids.py \
    --video_dir datasets/WebVid/2M/train \
    --todo_ids annotation/webvid-covr/webvid2m-covr_train.csv

# WebVid-CoVR test videos
python tools/embs/save_blip2_embs_vids.py \
    --video_dir datasets/WebVid/8M/train \
    --todo_ids annotation/webvid-covr/webvid8m-covr_test.csv

# CIRR images
python tools/embs/save_blip2_embs_imgs.py \
    --image_dir datasets/CIRR/images/test1 \
    --save_dir datasets/CIRR/blip2-embs-large/test1
python tools/embs/save_blip2_embs_imgs.py \
    --image_dir datasets/CIRR/images/dev \
    --save_dir datasets/CIRR/blip2-embs-large/dev
python tools/embs/save_blip2_embs_imgs.py \
    --image_dir datasets/CIRR/images/train \
    --save_dir datasets/CIRR/blip2-embs-large/train

# FashionIQ images
python tools/embs/save_blip2_embs_imgs.py \
    --image_dir datasets/fashion-iq/images/
```

**Note**: Use `--num_shards` and `--shard_id` for multi-GPU processing.

### Step 2: Training

Launch training with Hydra configuration:

```bash
python train.py [OPTIONS]
```

**Example - Train on WebVid-CoVR**:
```bash
python train.py \
    data=webvid-covr \
    model=blip2-coco \
    model/ckpt=blip2-l-coco \
    trainer=gpu \
    trainer.devices=1
```

**Key Parameters**:
- `data=webvid-covr`: WebVid-CoVR dataset
- `model=blip2-coco`: BLIP-2 model with Q-SlotSelect
- `trainer=gpu`: Single GPU training
- `trainer=ddp`: Distributed Data Parallel (multi-GPU)

### Step 3: Evaluation

```bash
python test.py test=<test> [OPTIONS]
```

**Example - Evaluate on all benchmarks**:
```bash
python test.py test=all
```

**Available test sets**:
- `test=webvid-covr`: WebVid-CoVR test set
- `test=cirr`: CIRR test set
- `test=fashioniq`: FashionIQ (dress, shirt, toptee)
- `test=circo`: CIRCO test set
- `test=all`: All benchmarks

### Configuration Options

**Models**:
- `model=blip2-coco`: BLIP-2 with Q-SlotSelect module

**Checkpoints**:
- `model/ckpt=blip2-l-coco`: BLIP-2 Large pretrained on COCO
- `model/ckpt=blip2-l-coco_webvid-covr`: Finetuned on WebVid-CoVR with Q-SlotSelect

**Loss Functions**:
- HN-NCE (Hard Negative Noise Contrastive Estimation) with temperature τ=0.07
- Attention entropy regularization (λ=0.2)

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
| Dress | TBD | TBD |
| Shirt | TBD | TBD |
| Toptee | TBD | TBD |
| **Average** | **TBD** | **TBD** |

### CIRR (Zero-shot)

| Metric | Value |
|--------|-------|
| R@1 | TBD |
| R@5 | TBD |
| R@10 | TBD |
| R@50 | TBD |

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

If you use this code in your research, please cite:

```bibtex
@article{qslotselect2026,
  title={Q-SlotSelect: Query-Guided Slot Selection with Top-K Semantic Filtering},
  author={Zeng, XianHua and Yang, Guyu and Tan, RuiYao},
  journal={Nuclear Physics B},
  year={2026},
  publisher={Elsevier}
}
```

## 🙏 Acknowledgements

This work builds upon the CoVR framework and BLIP-2 architecture. We thank the original authors for their contributions to the field of composed video retrieval.

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**Q-SlotSelect** | Chongqing University of Posts and Telecommunications

For questions or issues, please open an issue on GitHub.

</div>
