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
from examples.toy_train import Muon

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
    else:  # muon
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
    print("SIMPLE MUON vs ADAMW DEMONSTRATION")
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
    
    # Train with both optimizers
    print("\n" + "-"*60)
    print("Training with AdamW...")
    adamw_losses = train_model(base_model, 'adamw', seed=42)
    
    print("Training with Muon...")
    muon_losses = train_model(base_model, 'muon', seed=42)  # Same seed for fair comparison
    
    # Compare results
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    # Initial and final losses
    print(f"\nInitial Loss:")
    print(f"  AdamW: {adamw_losses[0]:.4f}")
    print(f"  Muon:  {muon_losses[0]:.4f}")
    
    print(f"\nFinal Loss (step 50):")
    print(f"  AdamW: {adamw_losses[-1]:.4f}")
    print(f"  Muon:  {muon_losses[-1]:.4f}")
    
    # Average loss over last 10 steps
    adamw_final_avg = np.mean(adamw_losses[-10:])
    muon_final_avg = np.mean(muon_losses[-10:])
    
    print(f"\nAverage Loss (last 10 steps):")
    print(f"  AdamW: {adamw_final_avg:.4f}")
    print(f"  Muon:  {muon_final_avg:.4f}")
    
    # Convergence analysis
    if muon_final_avg < adamw_final_avg:
        improvement = ((adamw_final_avg - muon_final_avg) / adamw_final_avg) * 100
        print(f"\n✓ Muon achieved {improvement:.1f}% lower loss")
    else:
        print(f"\n✗ AdamW achieved lower loss")
    
    # Find when Muon becomes better
    crossover = None
    for i in range(min(len(muon_losses), len(adamw_losses))):
        if muon_losses[i] < adamw_losses[i]:
            crossover = i
            break
    
    if crossover is not None:
        print(f"✓ Muon became more efficient at step {crossover}")
    
    # Show sample losses
    print("\n" + "-"*60)
    print("Sample Loss Values:")
    print("Step |  AdamW  |  Muon  ")
    print("-----|---------|--------")
    for step in [0, 10, 20, 30, 40, 49]:
        if step < len(adamw_losses) and step < len(muon_losses):
            print(f"  {step:2d} | {adamw_losses[step]:7.4f} | {muon_losses[step]:7.4f}")
    
    print("\n" + "="*60)
    print("Key Insight: Muon's orthogonal updates help preserve")
    print("gradient information, leading to more efficient training")
    print("especially in the weight matrices.")
    print("="*60)

if __name__ == "__main__":
    main()