#!/usr/bin/env python3
"""Enhanced model diagram generation for Moonlight project."""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import os
import argparse
import math

# Import Moonlight components
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from examples.toy_train import get_model_and_dataloader, Muon, zeropower_via_newtonschulz5
from transformers import Qwen2Config, Qwen2ForCausalLM

def create_text_summary(model, optimizer=None, model_name="Qwen2"):
    """Create a text summary of the model architecture."""
    print("=" * 80)
    print(f"Moonlight Project - {model_name} Architecture Summary")
    print("=" * 80)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Configuration:")
    if hasattr(model, 'config'):
        config = model.config
        print(f"Hidden size: {config.hidden_size}")
        print(f"Number of layers: {config.num_hidden_layers}")
        print(f"Number of attention heads: {config.num_attention_heads}")
        print(f"Intermediate size: {config.intermediate_size}")
        print(f"Vocabulary size: {config.vocab_size}")
        print(f"Max position embeddings: {config.max_position_embeddings}")

    print(f"\nParameter Count:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Show parameter breakdown by type
    print(f"\nParameter Breakdown:")
    param_groups = {}
    for name, param in model.named_parameters():
        param_type = "other"
        if "embed" in name:
            param_type = "embeddings"
        elif "attention" in name:
            param_type = "attention"
        elif "mlp" in name or "feed_forward" in name:
            param_type = "mlp"
        elif "norm" in name:
            param_type = "normalization"
        elif "lm_head" in name:
            param_type = "output_head"
        
        if param_type not in param_groups:
            param_groups[param_type] = 0
        param_groups[param_type] += param.numel()

    for param_type, count in param_groups.items():
        print(f"  {param_type}: {count:,} ({count/total_params*100:.1f}%)")

    # Show optimizer information if provided
    if optimizer:
        print(f"\nOptimizer Configuration:")
        if isinstance(optimizer, Muon):
            muon_params = [p for p in optimizer.param_groups[0]['params'] if optimizer.state[p]["use_muon"]]
            adamw_params = [p for p in optimizer.param_groups[0]['params'] if not optimizer.state[p]["use_muon"]]
            
            muon_param_count = sum(p.numel() for p in muon_params)
            adamw_param_count = sum(p.numel() for p in adamw_params)
            
            print(f"  Optimizer: Muon (hybrid)")
            print(f"  Muon parameters: {muon_param_count:,} ({muon_param_count/total_params*100:.1f}%)")
            print(f"  AdamW parameters: {adamw_param_count:,} ({adamw_param_count/total_params*100:.1f}%)")
            print(f"  Newton-Schulz steps: {optimizer.param_groups[0]['ns_steps']}")
            print(f"  Momentum: {optimizer.param_groups[0]['momentum']}")
        else:
            print(f"  Optimizer: {type(optimizer).__name__}")

def create_muon_optimizer_diagram():
    """Create ASCII diagram showing Muon optimizer architecture."""
    print("\n" + "="*80)
    print("MUON OPTIMIZER ARCHITECTURE")
    print("="*80)

    print("""
    Parameter Classification:
    ┌─────────────────────┐    ┌─────────────────────┐
    │   Model Parameters  │    │   Model Parameters  │
    │                     │    │                     │
    │  • Weight Matrices  │    │  • Embeddings       │
    │    (2D, ≥2D)        │    │  • Biases           │
    │  • Exclude embed_   │    │  • Layer norms      │
    │    tokens, lm_head  │    │  • 1D parameters    │
    └─────────────────────┘    └─────────────────────┘
              │                           │
              ▼                           ▼
    ┌─────────────────────┐    ┌─────────────────────┐
    │   MUON Optimizer    │    │   AdamW Optimizer   │
    │                     │    │                     │
    │ 1. SGD + Momentum   │    │ 1. Gradient moments │
    │    • buf ← βbuf+g   │    │    • m₁ ← β₁m₁+(1-β₁)g │
    │    • g ← g + βbuf   │    │    • m₂ ← β₂m₂+(1-β₂)g² │
    │                     │    │                     │
    │ 2. Newton-Schulz    │    │ 2. Bias correction  │
    │    Orthogonalization│    │    • m̂₁ ← m₁/(1-β₁ᵗ) │
    │    • u ← NS₅(g)     │    │    • m̂₂ ← m₂/(1-β₂ᵗ) │
    │                     │    │                     │
    │ 3. Adaptive LR      │    │ 3. Update rule      │
    │    • lr′← lr×0.2×   │    │    • θ ← θ - lr×m̂₁   │
    │      √max(h,w)      │    │           ────────   │
    │                     │    │           √m̂₂ + ε    │
    │ 4. Update           │    │                     │
    │    • θ ← θ - lr′×u  │    │                     │
    └─────────────────────┘    └─────────────────────┘

    Newton-Schulz Orthogonalization (NS₅):
    ┌──────────────────────────────────────────────────────┐
    │  G ← Gradient matrix                                 │
    │  X ← G / (||G|| + ε)     # Normalize spectral norm   │
    │  for i = 1 to 5:         # 5 NS iterations          │
    │    A ← X @ Xᵀ                                        │
    │    B ← b×A + c×A@A       # Quintic coefficients     │
    │    X ← a×X + B@X         # a=3.4445, b=-4.7750,     │
    │  return X                #          c=2.0315        │
    └──────────────────────────────────────────────────────┘

    Key Benefits:
    • ~2x computational efficiency vs AdamW
    • Orthogonal updates preserve gradient directions
    • Automatic learning rate scaling for matrix dimensions
    • Stable bfloat16 computation on GPU
    """)

def create_qwen2_architecture_diagram():
    """Create ASCII diagram showing Qwen2 model architecture."""
    print("\n" + "="*80)
    print("QWEN2 MODEL ARCHITECTURE")
    print("="*80)

    print("""
    Input Tokens [batch_size, seq_len]
                    │
                    ▼
    ┌──────────────────────────────────────────┐
    │          Token Embedding                 │
    │     embed_tokens: vocab_size → hidden    │
    └──────────────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────────────┐  ← 12 layers
    │          Qwen2 Decoder Layer            │  │
    │                                          │  │
    │  ┌────────────────────────────────────┐  │  │
    │  │     RMSNorm (input_layernorm)      │  │  │
    │  └────────────────────────────────────┘  │  │
    │                    │                     │  │
    │                    ▼                     │  │
    │  ┌────────────────────────────────────┐  │  │
    │  │    Multi-Head Self Attention       │  │  │
    │  │  • 16 heads, head_dim = hidden/16  │  │  │
    │  │  • RoPE positional encoding       │  │  │
    │  │  • Q,K,V projections + output      │  │  ├─┐
    │  └────────────────────────────────────┘  │  │ │
    │                    │                     │  │ │
    │                    ▼                     │  │ │
    │  ┌────────────────────────────────────┐  │  │ │
    │  │     RMSNorm (post_attention)       │  │  │ │
    │  └────────────────────────────────────┘  │  │ │
    │                    │                     │  │ │
    │                    ▼                     │  │ │
    │  ┌────────────────────────────────────┐  │  │ │
    │  │         MLP (Feed Forward)         │  │  │ │
    │  │  • gate_proj: hidden → 4864       │  │  │ │
    │  │  • up_proj: hidden → 4864         │  │  │ │
    │  │  • SiLU activation                 │  │  │ │
    │  │  • down_proj: 4864 → hidden       │  │  │ │
    │  └────────────────────────────────────┘  │  │ │
    │                                          │ ◄┘ │
    └──────────────────────────────────────────┘    │
                    │                              │
                    ▼                              │
                 Residual ←────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────────────┐
    │              RMSNorm                     │
    │            (final norm)                  │
    └──────────────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────────────┐
    │            LM Head                       │
    │      hidden → vocab_size (151,936)      │
    └──────────────────────────────────────────┘
                    │
                    ▼
             Output Logits [batch_size, seq_len, vocab_size]
    """)

def create_qwen2_model(hidden_size=896):
    """Create just the Qwen2 model without dataset loading."""
    config = Qwen2Config(
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151643,
        hidden_act="silu",
        hidden_size=hidden_size,
        initializer_range=0.02,
        intermediate_size=4864,
        max_position_embeddings=513,
        max_window_layers=12,
        model_type="qwen2",
        num_attention_heads=16,
        num_hidden_layers=12,
        num_key_value_heads=16,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        sliding_window=1024,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        use_cache=True,
        use_mrope=False,
        use_sliding_window=False,
        vocab_size=151936,
    )
    return Qwen2ForCausalLM(config)

def create_lightweight_optimizer(optimizer_name, model, lr=1e-3, wd=0.1):
    """Create optimizer without full dataset dependencies."""
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]
        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def generate_model_diagrams(optimizer_name=None, hidden_size=896):
    """Generate diagrams for Moonlight models and optimizers."""
    print("Generating Moonlight model architecture diagrams...")
    
    # Create model without dataset loading
    model = create_qwen2_model(hidden_size)
    
    optimizer = None
    if optimizer_name == "muon":
        optimizer = create_lightweight_optimizer("muon", model, lr=1e-3, wd=0.1)
    elif optimizer_name == "adamw":
        optimizer = create_lightweight_optimizer("adamw", model, lr=1e-3, wd=0.1)
    
    # Show model summary
    create_text_summary(model, optimizer, model_name=f"Qwen2 (h={hidden_size})")
    
    # Show architecture diagrams
    create_qwen2_architecture_diagram()
    
    # Show optimizer-specific diagrams
    if optimizer_name == "muon":
        create_muon_optimizer_diagram()
    elif optimizer_name == "adamw":
        create_adamw_optimizer_diagram()
    elif optimizer_name is None:
        # Show both
        create_muon_optimizer_diagram()
        create_adamw_optimizer_diagram()

def create_adamw_optimizer_diagram():
    """Create ASCII diagram showing AdamW optimizer architecture."""
    print("\n" + "="*80)
    print("ADAMW OPTIMIZER ARCHITECTURE")
    print("="*80)

    print("""
    AdamW Update Rule:
    ┌──────────────────────────────────────────────────────┐
    │  Input: parameters θ, gradients g, learning rate lr │
    │                                                      │
    │  State variables:                                    │
    │  • m₁ ← first moment (momentum)                      │
    │  • m₂ ← second moment (variance)                     │
    │  • t  ← time step                                    │
    │                                                      │
    │  Update rule:                                        │
    │  1. t ← t + 1                                        │
    │  2. m₁ ← β₁ × m₁ + (1 - β₁) × g                      │
    │  3. m₂ ← β₂ × m₂ + (1 - β₂) × g²                     │
    │  4. m̂₁ ← m₁ / (1 - β₁ᵗ)    # Bias correction        │
    │  5. m̂₂ ← m₂ / (1 - β₂ᵗ)    # Bias correction        │
    │  6. θ ← θ - lr × (m̂₁ / (√m̂₂ + ε) + λ × θ)           │
    │                                                      │
    │  Hyperparameters:                                    │
    │  • β₁ = 0.9     (momentum decay)                     │
    │  • β₂ = 0.95    (variance decay)                     │
    │  • ε = 1e-8     (numerical stability)               │
    │  • λ = 0.1      (weight decay)                       │
    └──────────────────────────────────────────────────────┘

    Key Features:
    • Adaptive learning rates per parameter
    • Momentum for smooth convergence
    • Bias correction for early training steps
    • Weight decay for regularization
    """)

def main():
    parser = argparse.ArgumentParser(description="Generate Moonlight architecture diagrams")
    parser.add_argument("--optimizer", choices=["muon", "adamw"], default=None,
                       help="Optimizer to generate diagrams for (default: both)")
    parser.add_argument("--hidden-size", type=int, default=896,
                       help="Hidden size for model (default: 896)")
    parser.add_argument("--compare-optimizers", action="store_true",
                       help="Generate comparison diagrams for both optimizers")

    args = parser.parse_args()

    if args.compare_optimizers:
        print("Generating comparison diagrams for Muon vs AdamW optimizers...")
        generate_model_diagrams(optimizer_name=None, hidden_size=args.hidden_size)
    else:
        generate_model_diagrams(optimizer_name=args.optimizer, hidden_size=args.hidden_size)

    print(f"\n{'='*80}")
    print("Moonlight architecture diagram generation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
