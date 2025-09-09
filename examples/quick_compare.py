#!/usr/bin/env python3
"""Quick comparison of Muon vs AdamW efficiency on a small model."""

import torch
import torch.nn as nn
import numpy as np
import random
import time
from transformers import Qwen2Config, Qwen2ForCausalLM
import argparse
import sys
import os

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not installed. Plotting disabled.")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.toy_train import Muon, MORGN, get_optimizer

# Set seeds for reproducibility
def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_tiny_model(hidden_size=128, num_layers=2):
    """Create a tiny Qwen2 model for quick testing."""
    config = Qwen2Config(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,  # Must match num_attention_heads for now
        intermediate_size=hidden_size * 4,
        vocab_size=1000,  # Small vocabulary
        max_position_embeddings=128,
        model_type="qwen2",
        torch_dtype="float32",  # Use float32 for stability in small models
        use_sliding_window=False,  # Disable sliding window
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
    )
    return Qwen2ForCausalLM(config)

def create_synthetic_data(batch_size=8, seq_len=32, vocab_size=1000):
    """Create random data for testing."""
    return torch.randint(0, vocab_size, (batch_size, seq_len))

def train_step(model, optimizer, data, device):
    """Single training step."""
    data = data.to(device)
    outputs = model(input_ids=data, labels=data)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def compare_optimizers(hidden_size=128, num_steps=100, lr=1e-3, device='cpu', seed=42):
    """Compare AdamW, Muon, and MORGN on the same model architecture."""
    print(f"\n{'='*70}")
    print(f"Comparing AdamW vs Muon vs MORGN")
    print(f"Model: Qwen2 (hidden_size={hidden_size}, layers=2)")
    print(f"Training steps: {num_steps}, LR: {lr}")
    print(f"Random seed: {seed} (fixed for fair comparison)")
    print(f"{'='*70}\n")
    
    # Set initial seed for model creation
    set_seed(seed)
    
    # Create base model that both optimizers will start from
    base_model = create_tiny_model(hidden_size=hidden_size).to(device)
    base_state_dict = base_model.state_dict()
    
    results = {}
    
    for optimizer_name in ['adamw', 'muon', 'morgn']:
        print(f"\nTraining with {optimizer_name.upper()}...")
        
        # Reset seed for each optimizer to see same data
        set_seed(seed)
        
        # Create fresh model with same initialization
        model = create_tiny_model(hidden_size=hidden_size).to(device)
        model.load_state_dict(base_state_dict)  # Ensure same initialization
        
        # Create optimizer
        optimizer = get_optimizer(optimizer_name, model, lr=lr, wd=0.0)
        
        # Training loop
        losses = []
        start_time = time.time()
        
        for step in range(num_steps):
            data = create_synthetic_data()
            loss = train_step(model, optimizer, data, device)
            losses.append(loss)
            
            if step % 20 == 0:
                print(f"  Step {step:3d}: Loss = {loss:.4f}")
        
        train_time = time.time() - start_time
        
        # Calculate convergence metrics
        final_loss = np.mean(losses[-10:])  # Average of last 10 steps
        convergence_speed = losses[0] / losses[-1] if losses[-1] > 0 else float('inf')
        
        results[optimizer_name] = {
            'losses': losses,
            'final_loss': final_loss,
            'convergence_speed': convergence_speed,
            'train_time': train_time
        }
        
        print(f"\n  {optimizer_name.upper()} Results:")
        print(f"    Final loss: {final_loss:.4f}")
        print(f"    Convergence ratio: {convergence_speed:.2f}x")
        print(f"    Training time: {train_time:.2f}s")
    
    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    # Get final losses for all optimizers
    final_losses = {name: results[name]['final_loss'] for name in results}
    convergence_speeds = {name: results[name]['convergence_speed'] for name in results}
    
    # Find best performer by final loss
    best_optimizer = min(final_losses, key=final_losses.get)
    print(f"🏆 Best final loss: {best_optimizer.upper()} ({final_losses[best_optimizer]:.6f})")
    
    # Compare each optimizer to the best
    for optimizer_name in ['adamw', 'muon', 'morgn']:
        if optimizer_name == best_optimizer:
            continue
        improvement = ((final_losses[optimizer_name] - final_losses[best_optimizer]) / final_losses[optimizer_name]) * 100
        if improvement > 0:
            print(f"✓ {best_optimizer.upper()} achieved {improvement:.1f}% lower loss than {optimizer_name.upper()}")
    
    # Compare convergence speeds
    best_conv = max(convergence_speeds, key=convergence_speeds.get)
    print(f"🚀 Fastest convergence: {best_conv.upper()} ({convergence_speeds[best_conv]:.2f}x)")
    
    # Find crossover points
    for opt1 in ['muon', 'morgn']:
        if opt1 in results:
            adamw_losses = results['adamw']['losses']
            opt_losses = results[opt1]['losses']
            
            crossover_step = None
            for i in range(min(len(opt_losses), len(adamw_losses))):
                if opt_losses[i] < adamw_losses[i]:
                    crossover_step = i
                    break
            
            if crossover_step is not None:
                print(f"✓ {opt1.upper()} became better than AdamW at step {crossover_step}")
    
    return results

def plot_comparison(results, save_path='optimizer_comparison.png'):
    """Plot loss curves for both optimizers."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping plot generation (matplotlib not installed)")
        print("Install with: pip install matplotlib")
        return
    
    plt.figure(figsize=(10, 6))
    
    for optimizer_name, data in results.items():
        losses = data['losses']
        label = f"{optimizer_name.upper()} (final: {data['final_loss']:.4f})"
        plt.plot(losses, label=label, linewidth=2)
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('AdamW vs Muon vs MORGN Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.close()  # Close the figure to free memory

def main():
    parser = argparse.ArgumentParser(description='Quick Muon vs AdamW comparison')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='Hidden size of the model (default: 128)')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of training steps (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plot')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Device selection logic
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using Apple MPS (Metal Performance Shaders) - auto-detected")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA GPU - auto-detected")
        else:
            device = torch.device('cpu')
            print("Using CPU - auto-detected")
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print(f"Using CPU ({'explicit' if args.device == 'cpu' else 'fallback'})")
    
    # Run comparison
    results = compare_optimizers(
        hidden_size=args.hidden_size,
        num_steps=args.steps,
        lr=args.lr,
        device=device,
        seed=args.seed
    )
    
    # Generate plot if requested
    if args.plot:
        plot_comparison(results)
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()