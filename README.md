# In-context Ranking Preference Optimization (IRPO)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of **In-context Ranking Preference Optimization (IRPO)** for optimizing LLMs on ranking tasks.

> **In-context Ranking Preference Optimization**  
> Junda Wu*, Rohan Surana*, Zhouhang Xie, Yiran Shen, Yu Xia, Tong Yu, Ryan A. Rossi, Prithviraj Ammanabrolu, Julian McAuley  
> [arXiv:2504.15477](https://arxiv.org/abs/2504.15477)

---

## Overview

IRPO is a framework for optimizing large language models (LLMs) on ranking tasks using listwise preference feedback. Unlike pairwise preference optimization methods (e.g., DPO), IRPO directly optimizes on ranked lists with graded relevance labels.

**Key Features:**
- **Listwise optimization**: Optimizes over entire ranked lists rather than pairwise comparisons
- **Position-aware weighting**: Incorporates both item relevance and ranking position using NDCG-inspired weights
- **Gradient-based**: Enables end-to-end differentiable optimization of ranking objectives
- **Flexible relevance**: Supports graded relevance labels (not just binary preferences)

**Supported Methods:**
| Method | Description |
|--------|-------------|
| `rdpo` | IRPO (our method) - listwise ranking preference optimization |
| `lipo` | LiPO baseline - listwise preference optimization |
| `sdpo` | SDPO baseline - sequential DPO with multiple negatives (in `sdpo/`) |
| `dpo` | DPO baseline - pairwise direct preference optimization |
| `sft` | SFT baseline - supervised fine-tuning |

---


## Quick Start

### Training

```bash
# Train with IRPO (rdpo) loss
python train.py \
    model=llama3b \
    loss=rdpo \
    loss.beta=1.0 \
    embed_dirs=datasets/redial_train.csv \
    datasets=[reddit] \
    exp_name=llama3b_redial_irpo \
    n_epochs=5
```
### Evaluation

```bash
python evaluate.py \
    model=llama3b \
    embed_dirs=datasets/redial_eval.csv \
    datasets=[reddit] \
    eval_batch_size=32 \
    model.archive=path/to/checkpoint/policy.pt
```

### Using Shell Scripts

```bash
# Training
./scripts/train.sh redial llama3b rdpo

# Evaluation
./scripts/evaluate.sh redial llama3b path/to/checkpoint/policy.pt
```

---

## Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Model config (`llama3b`, `gemma2-2b`, `phi3-mini`, etc.) | `gemma2-2b` |
| `loss` | Loss function (`rdpo`, `lipo`, `sft`, `dpo`) | `rdpo` |
| `loss.beta` | Temperature parameter for preference optimization | `1.0` |
| `datasets` | Dataset type (`[reddit]` or `[nlp]`) | `[reddit]` |
| `embed_dirs` | Path to training/eval CSV file | - |
| `n_epochs` | Number of training epochs | `3` |
| `batch_size` | Training batch size | `32` |
| `eval_batch_size` | Evaluation batch size | `4` |
| `gradient_accumulation_steps` | Gradient accumulation steps | `8` |
| `lr` | Learning rate | `1e-5` |
| `model.archive` | Path to checkpoint for evaluation/resumption | `null` |

---

## Project Structure

```
irpo/
├── train.py              # Main training script
├── evaluate.py           # Main evaluation script
├── trainers.py           # Trainer classes and loss functions (IRPO, LiPO, SFT)
├── evaluaters.py         # Evaluation logic
├── train_datasets.py     # Training data loading
├── evaluate_datasets.py  # Evaluation data loading
├── prompts.py            # Model-specific prompt templates
├── utils.py              # Utility functions
├── config/
│   ├── config.yaml       # Main configuration
│   ├── model/            # Model configs (llama3b, gemma2-2b, phi3-mini, etc.)
│   └── loss/             # Loss configs (rdpo, lipo, sft, dpo)
├── scripts/
│   ├── train.sh          # Training shell script
│   └── evaluate.sh       # Evaluation shell script
├── sdpo/                 # SDPO baseline implementation
├── iterative/            # Iterative/online training experiments
└── sample_data/          # Example data format
```

---

## Dataset Format

See `sample_data/` for example CSV files with the expected format.

---

## Metrics

Evaluation computes standard ranking metrics:
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank  
- **Recall@k**: Recall at position k
- **MR@k**: Mean Relevance at position k

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{wu2025context,
  title={In-context Ranking Preference Optimization},
  author={Wu, Junda and Surana, Rohan and Xie, Zhouhang and Shen, Yiran and Xia, Yu and Yu, Tong and Rossi, Ryan A and Ammanabrolu, Prithviraj and McAuley, Julian},
  journal={arXiv preprint arXiv:2504.15477},
  year={2025}
}
```

---


## Acknowledgments

This codebase builds upon the [Direct Preference Optimization](https://github.com/eric-mitchell/direct-preference-optimization) implementation.
