#!/usr/bin/env python3
"""Comparison of Muon vs AdamW on matrix factorization of positive definite matrices."""

import torch
import torch.nn as nn
import numpy as np
import random
import time
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

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_positive_definite_matrix(size: int, condition_number: float = 100.0, device: torch.device = torch.device('cpu')) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a random positive definite matrix with specified condition number."""
    # For MPS compatibility, generate matrix on CPU first if QR not available
    if device.type == 'mps':
        # Create on CPU then move to MPS
        A = torch.randn(size, size, device='cpu')
        Q, _ = torch.linalg.qr(A)
        
        # Create diagonal matrix with specified condition number
        eigenvals = torch.logspace(0, np.log10(condition_number), size, device='cpu')
        eigenvals = eigenvals / eigenvals.sum() * size  # Normalize to reasonable scale
        D = torch.diag(eigenvals)
        
        # Construct positive definite matrix: Q @ D @ Q.T
        target_matrix = Q @ D @ Q.T
        return target_matrix.to(device), eigenvals.to(device)
    else:
        # Generate directly on target device
        A = torch.randn(size, size, device=device)
        Q, _ = torch.linalg.qr(A)
        
        # Create diagonal matrix with specified condition number
        eigenvals = torch.logspace(0, np.log10(condition_number), size, device=device)
        eigenvals = eigenvals / eigenvals.sum() * size  # Normalize to reasonable scale
        D = torch.diag(eigenvals)
        
        # Construct positive definite matrix: Q @ D @ Q.T
        target_matrix = Q @ D @ Q.T
        return target_matrix, eigenvals

class MatrixFactorizationModel(nn.Module):
    """Simple model that learns to factorize a positive definite matrix."""
    
    def __init__(self, matrix_size: int, rank: int = None):
        super().__init__()
        if rank is None:
            rank = matrix_size  # Full rank factorization
        
        self.matrix_size = matrix_size
        self.rank = rank
        
        # Learn factors A and B such that A @ B.T ≈ target_matrix
        self.factor_A = nn.Parameter(torch.randn(matrix_size, rank) * 0.1)
        self.factor_B = nn.Parameter(torch.randn(matrix_size, rank) * 0.1)
    
    def forward(self) -> torch.Tensor:
        """Compute the reconstructed matrix A @ B.T."""
        return self.factor_A @ self.factor_B.T
    
    def get_muon_adamw_params(self):
        """Separate parameters for Muon (2D) and AdamW (others)."""
        muon_params = [p for p in self.parameters() if p.ndim >= 2]
        adamw_params = [p for p in self.parameters() if p.ndim < 2]
        return muon_params, adamw_params

def train_factorization(target_matrix: torch.Tensor, optimizer_name: str, 
                       num_steps: int = 200, lr: float = 1e-2, 
                       rank: int = None, device: torch.device = torch.device('cpu'),
                       seed: int = 42) -> tuple[list[float], MatrixFactorizationModel]:
    """Train matrix factorization with specified optimizer."""
    set_seed(seed)
    
    matrix_size = target_matrix.size(0)
    model = MatrixFactorizationModel(matrix_size, rank).to(device)
    
    # Create optimizer
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    elif optimizer_name == 'muon':
        muon_params, adamw_params = model.get_muon_adamw_params()
        
        optimizer = Muon(
            lr=lr,
            wd=0.0,
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=0.95,
            ns_steps=5
        )
    else:  # morgn
        morgn_params, adamw_params = model.get_muon_adamw_params()
        
        optimizer = MORGN(
            lr=lr,
            wd=0.0,
            morgn_params=morgn_params,
            adamw_params=adamw_params,
            momentum=0.95,
            ns_steps=5
        )
    
    losses = []
    criterion = nn.MSELoss()
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Compute reconstruction
        reconstructed = model()
        
        # Loss: MSE between target and reconstruction
        loss = criterion(reconstructed, target_matrix)
        
        # Additional loss to encourage positive definiteness (optional)
        # We could add regularization here, but for now keep it simple
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 50 == 0:
            print(f"  Step {step:3d}: Loss = {loss.item():.6f}")
    
    return losses, model

def analyze_reconstruction_quality(model: MatrixFactorizationModel, target_matrix: torch.Tensor) -> dict:
    """Analyze the quality of matrix reconstruction."""
    with torch.no_grad():
        reconstructed = model()
        
        # Compute various metrics
        mse_error = torch.mean((reconstructed - target_matrix) ** 2).item()
        frobenius_error = torch.norm(reconstructed - target_matrix, 'fro').item()
        relative_error = frobenius_error / torch.norm(target_matrix, 'fro').item()
        
        # Check if reconstructed matrix is positive definite
        # Compute eigenvalues on CPU for MPS compatibility
        if target_matrix.device.type == 'mps':
            eigenvals_target = torch.linalg.eigvals(target_matrix.cpu()).real
            eigenvals_recon = torch.linalg.eigvals(reconstructed.cpu()).real
        else:
            eigenvals_target = torch.linalg.eigvals(target_matrix).real
            eigenvals_recon = torch.linalg.eigvals(reconstructed).real
        
        min_eigval_target = eigenvals_target.min().item()
        min_eigval_recon = eigenvals_recon.min().item()
        
        is_pos_def_target = min_eigval_target > 0
        is_pos_def_recon = min_eigval_recon > 0
        
        return {
            'mse_error': mse_error,
            'frobenius_error': frobenius_error,
            'relative_error': relative_error,
            'min_eigval_target': min_eigval_target,
            'min_eigval_recon': min_eigval_recon,
            'is_pos_def_target': is_pos_def_target,
            'is_pos_def_recon': is_pos_def_recon,
        }

def compare_matrix_factorization(matrix_size: int = 64, rank: int = None, 
                                num_steps: int = 200, lr: float = 1e-2,
                                condition_number: float = 100.0,
                                device: torch.device = torch.device('cpu'), 
                                seed: int = 42) -> dict:
    """Compare AdamW, Muon, and MORGN on matrix factorization task."""
    print(f"\n{'='*70}")
    print(f"Matrix Factorization: AdamW vs Muon vs MORGN")
    print(f"{'='*70}")
    print(f"Problem: Factorize positive definite matrix")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    if rank is None:
        rank = matrix_size
    print(f"Factorization rank: {rank}")
    print(f"Condition number: {condition_number}")
    print(f"Training steps: {num_steps}, Learning rate: {lr}")
    print(f"Random seed: {seed} (fixed for fair comparison)")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Create target positive definite matrix
    set_seed(seed)
    target_matrix, eigenvals = create_positive_definite_matrix(matrix_size, condition_number, device)
    
    print(f"Target matrix properties:")
    print(f"  Eigenvalue range: [{eigenvals.min().item():.4f}, {eigenvals.max().item():.4f}]")
    print(f"  Condition number: {(eigenvals.max() / eigenvals.min()).item():.2f}")
    print(f"  Frobenius norm: {torch.norm(target_matrix, 'fro').item():.4f}")
    
    results = {}
    
    for optimizer_name in ['adamw', 'muon', 'morgn']:
        print(f"\nTraining with {optimizer_name.upper()}...")
        
        start_time = time.time()
        losses, model = train_factorization(
            target_matrix, optimizer_name, num_steps, lr, rank, device, seed
        )
        train_time = time.time() - start_time
        
        # Analyze reconstruction quality
        quality_metrics = analyze_reconstruction_quality(model, target_matrix)
        
        results[optimizer_name] = {
            'losses': losses,
            'final_loss': losses[-1],
            'train_time': train_time,
            'model': model,
            **quality_metrics
        }
        
        print(f"\n  {optimizer_name.upper()} Results:")
        print(f"    Final MSE loss: {losses[-1]:.6f}")
        print(f"    Relative error: {quality_metrics['relative_error']:.4f}")
        print(f"    Min eigenvalue (recon): {quality_metrics['min_eigval_recon']:.4f}")
        print(f"    Positive definite: {quality_metrics['is_pos_def_recon']}")
        print(f"    Training time: {train_time:.2f}s")
    
    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    # Get final losses and relative errors for all optimizers
    final_losses = {name: results[name]['final_loss'] for name in results}
    relative_errors = {name: results[name]['relative_error'] for name in results}
    
    # Find best performer by final loss
    best_loss_optimizer = min(final_losses, key=final_losses.get)
    print(f"🏆 Best final loss: {best_loss_optimizer.upper()} ({final_losses[best_loss_optimizer]:.6f})")
    
    # Compare each optimizer to the best
    for optimizer_name in results:
        if optimizer_name == best_loss_optimizer:
            continue
        improvement = ((final_losses[optimizer_name] - final_losses[best_loss_optimizer]) / final_losses[optimizer_name]) * 100
        if improvement > 0:
            print(f"✓ {best_loss_optimizer.upper()} achieved {improvement:.1f}% lower loss than {optimizer_name.upper()}")
    
    # Find best by relative error
    best_error_optimizer = min(relative_errors, key=relative_errors.get)
    print(f"🎯 Best relative error: {best_error_optimizer.upper()} ({relative_errors[best_error_optimizer]:.4f})")
    
    # Check convergence speed
    def find_convergence_step(losses, threshold=0.9):
        if not losses:
            return len(losses)
        final_loss = losses[-1]
        initial_loss = losses[0]
        target_loss = final_loss + threshold * (initial_loss - final_loss)
        
        for i, loss in enumerate(losses):
            if loss <= target_loss:
                return i
        return len(losses)
    
    convergence_steps = {name: find_convergence_step(results[name]['losses']) for name in results}
    fastest_convergence = min(convergence_steps, key=convergence_steps.get)
    print(f"🚀 Fastest convergence: {fastest_convergence.upper()} (step {convergence_steps[fastest_convergence]})")
    
    # Show convergence comparison
    for name, step in convergence_steps.items():
        if name != fastest_convergence:
            if step < len(results[name]['losses']):
                print(f"   {name.upper()} converged at step {step}")
            else:
                print(f"   {name.upper()} did not fully converge")
    
    return results

def plot_matrix_factorization_comparison(results: dict, save_path: str = 'matrix_factorization_comparison.png') -> None:
    """Plot loss curves and reconstruction quality for both optimizers."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping plot generation (matplotlib not installed)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Loss curves
    for optimizer_name, data in results.items():
        losses = data['losses']
        label = f"{optimizer_name.upper()} (final: {data['final_loss']:.6f})"
        axes[0].plot(losses, label=label, linewidth=2)
    
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Matrix Factorization: AdamW vs Muon vs MORGN')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: Reconstruction quality comparison
    optimizer_names = list(results.keys())
    metrics = ['final_loss', 'relative_error', 'frobenius_error']
    metric_labels = ['Final Loss', 'Relative Error', 'Frobenius Error']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, optimizer_name in enumerate(optimizer_names):
        values = [results[optimizer_name][metric] for metric in metrics]
        axes[1].bar(x + i*width, values, width, label=optimizer_name.upper())
    
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Reconstruction Quality Comparison')
    axes[1].set_xticks(x + width/2)
    axes[1].set_xticklabels(metric_labels)
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare Muon vs AdamW on matrix factorization')
    parser.add_argument('--matrix-size', type=int, default=64,
                       help='Size of the matrix to factorize (default: 64)')
    parser.add_argument('--rank', type=int, default=None,
                       help='Rank of factorization (default: full rank)')
    parser.add_argument('--steps', type=int, default=200,
                       help='Number of training steps (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-2,
                       help='Learning rate (default: 1e-2)')
    parser.add_argument('--condition-number', type=float, default=100.0,
                       help='Condition number of target matrix (default: 100.0)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
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
    results = compare_matrix_factorization(
        matrix_size=args.matrix_size,
        rank=args.rank,
        num_steps=args.steps,
        lr=args.lr,
        condition_number=args.condition_number,
        device=device,
        seed=args.seed
    )
    
    # Generate plots if requested
    if args.plot:
        plot_matrix_factorization_comparison(results)
    
    print("\nMatrix factorization experiment complete!")
    print("\nKey insight: This three-way comparison shows how different")
    print("optimizers perform on matrix factorization problems.")
    print("MORGN is currently a stub (SGD+momentum) but will be enhanced")
    print("to demonstrate novel optimization techniques.")

if __name__ == "__main__":
    main()