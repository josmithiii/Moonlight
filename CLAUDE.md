# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Moonlight is a research implementation demonstrating the Muon optimizer (MomentUm Orthogonalized by Newton-schulz) for training language models. The project includes a scalable distributed implementation of Muon and compares it against AdamW for training small-scale language models. The main contribution is showing that Muon can achieve ~2x computational efficiency compared to AdamW.

## Core Architecture

The codebase is organized as a minimal training framework with the following key components:

- **Muon Optimizer** (`examples/toy_train.py`): Complete implementation of the Muon optimizer with Newton-Schulz orthogonalization, including fallback to AdamW for embedding/head layers and non-2D parameters
- **Training Pipeline** (`examples/toy_train.py`): End-to-end training script supporting both Muon and AdamW optimizers on Qwen2-based models
- **Dataset Handling** (`MoonDataset` class): Tokenizes and caches datasets with configurable sequence length

## Key Technical Details

### Muon Optimizer Implementation
- Uses Newton-Schulz iteration (`zeropower_via_newtonschulz5`) for matrix orthogonalization
- Automatically adjusts learning rate based on parameter matrix dimensions: `lr * 0.2 * sqrt(max(height, width))`
- Separates parameters into Muon-optimized (2D weight matrices) and AdamW-optimized (embeddings, biases, layer norms)
- Compiled with `@torch.compile` for performance

### Training Configuration
- Supports Qwen2-based models with configurable hidden sizes
- Default batch size: 16, sequence length: 512
- Uses cosine learning rate schedule with warmup
- Logs training progress with loguru to `logs/train_{model}_{optimizer}_lr{lr}.log`

## Development Commands

### Training Models
```bash
# Train with Muon optimizer
python3 examples/toy_train.py --model qwen --optimizer muon --dataset openwebtext-100k --hidden_size 896 --lr 1e-3

# Train with AdamW baseline
python3 examples/toy_train.py --model qwen --optimizer adamw --dataset openwebtext-100k --hidden_size 896 --lr 1e-3
```

### Dependencies Installation
```bash
pip install -r requirements.txt
```

Required dependencies: `datasets==3.3.2`, `loguru==0.7.3`, `numpy==2.2.3`, `torch==2.6.0`, `tqdm==4.67.1`, `transformers==4.49.0`

## Dataset Support

Currently supports `openwebtext-100k` dataset via HuggingFace. The `MoonDataset` class automatically:
- Tokenizes texts using the model's tokenizer
- Caches tokenized data as `.bin` files to avoid re-tokenization
- Creates fixed-length sequences for training

## Research Context

This is research software focused on optimizer comparison and scaling analysis. The codebase demonstrates:
- Muon's ~2x training efficiency compared to AdamW
- Scaling law validation showing Muon requires ~52% of AdamW's training FLOPs for comparable performance
- Integration with standard transformer architectures (Qwen2)

The implementation prioritizes research flexibility over production robustness - expect to fail fast on errors rather than graceful fallbacks.