#!/usr/bin/env python3
"""Simplified demonstration of Muon vs AdamW on a toy problem."""

import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.toy_train import Muon, MORGN

# Set seeds for reproducibility
def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleModel(nn.Module):
    """Simple 2-layer neural network for demonstration."""
    def __init__(self, input_size=64, hidden_size=128, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def create_toy_problem(batch_size=32, input_size=64, output_size=10):
    """Create a simple regression problem."""
    # Generate random input and target
    X = torch.randn(batch_size, input_size)
    y = torch.randn(batch_size, output_size)
    return X, y

def train_model(model, optimizer_name, num_steps=50, lr=1e-2, seed=42):
    """Train model with specified optimizer."""
    # Set seed for this training run
    set_seed(seed)
    
    model_copy = SimpleModel()  # Fresh model
    model_copy.load_state_dict(model.state_dict())  # Same initialization
    
    # Create optimizer
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model_copy.parameters(), lr=lr)
    elif optimizer_name == 'muon':
        # Separate parameters for Muon
        muon_params = [p for p in model_copy.parameters() if p.ndim >= 2]
        adamw_params = [p for p in model_copy.parameters() if p.ndim < 2]
        
        optimizer = Muon(
            lr=lr,
            wd=0.0,
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=0.95,
            ns_steps=5
        )
    else:  # morgn
        # Separate parameters for MORGN
        morgn_params = [p for p in model_copy.parameters() if p.ndim >= 2]
        adamw_params = [p for p in model_copy.parameters() if p.ndim < 2]
        
        optimizer = MORGN(lr=lr, wd=0.0, morgn_params=morgn_params, adamw_params=adamw_params)
    
    losses = []
    criterion = nn.MSELoss()
    
    for step in range(num_steps):
        X, y = create_toy_problem()
        
        # Forward pass
        output = model_copy(X)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return losses

def main():
    print("\n" + "="*60)
    print("SIMPLE THREE-WAY OPTIMIZER DEMONSTRATION")
    print("="*60)
    print("\nProblem: Simple 2-layer network regression")
    print("Network: 64 → 128 → 10")
    print("Training steps: 50")
    print("Learning rate: 0.01")
    print("Random seed: 42 (fixed for fair comparison)")
    print("-"*60)
    
    # Set global seed for model initialization
    set_seed(42)
    
    # Create base model for consistent initialization
    base_model = SimpleModel()
    
    # Count parameters
    total_params = sum(p.numel() for p in base_model.parameters())
    matrix_params = sum(p.numel() for p in base_model.parameters() if p.ndim >= 2)
    bias_params = sum(p.numel() for p in base_model.parameters() if p.ndim < 2)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Matrices (Muon): {matrix_params:,} ({matrix_params/total_params*100:.1f}%)")
    print(f"  Biases (AdamW): {bias_params:,} ({bias_params/total_params*100:.1f}%)")
    
    # Train with all three optimizers
    print("\n" + "-"*60)
    print("Training with AdamW...")
    adamw_losses = train_model(base_model, 'adamw', seed=42)
    
    print("Training with Muon...")
    muon_losses = train_model(base_model, 'muon', seed=42)
    
    print("Training with MORGN...")
    morgn_losses = train_model(base_model, 'morgn', seed=42)
    
    # Compare results
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    # Store all results
    results = {
        'adamw': adamw_losses,
        'muon': muon_losses,
        'morgn': morgn_losses
    }
    
    # Initial and final losses
    print(f"\nInitial Loss:")
    for name, losses in results.items():
        print(f"  {name.upper():6s}: {losses[0]:.4f}")
    
    print(f"\nFinal Loss (step 50):")
    for name, losses in results.items():
        print(f"  {name.upper():6s}: {losses[-1]:.4f}")
    
    # Average loss over last 10 steps
    final_avgs = {name: np.mean(losses[-10:]) for name, losses in results.items()}
    
    print(f"\nAverage Loss (last 10 steps):")
    for name, avg in final_avgs.items():
        print(f"  {name.upper():6s}: {avg:.4f}")
    
    # Find best performer
    best_optimizer = min(final_avgs, key=final_avgs.get)
    print(f"\n🏆 Best performer: {best_optimizer.upper()} ({final_avgs[best_optimizer]:.4f})")
    
    # Compare to best
    for name, avg in final_avgs.items():
        if name != best_optimizer:
            improvement = ((avg - final_avgs[best_optimizer]) / avg) * 100
            if improvement > 0:
                print(f"✓ {best_optimizer.upper()} achieved {improvement:.1f}% lower loss than {name.upper()}")
    
    # Show sample losses
    print("\n" + "-"*60)
    print("Sample Loss Values:")
    print("Step |  AdamW  |  Muon   | MORGN  ")
    print("-----|---------|---------|--------")
    for step in [0, 10, 20, 30, 40, 49]:
        if step < min(len(losses) for losses in results.values()):
            print(f"  {step:2d} | {results['adamw'][step]:7.4f} | {results['muon'][step]:7.4f} | {results['morgn'][step]:7.4f}")

if __name__ == "__main__":
    main()
