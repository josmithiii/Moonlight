#!/usr/bin/env python3
"""Comparison of optimizers on rotated paraboloid problem with correlated gradients."""

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
from examples.toy_train import Muon, MORGN

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_rotation_matrix(dim: int, angle_degrees: float = 45.0, mode: str = 'random') -> torch.Tensor:
    """Create an orthogonal rotation matrix.

    - mode='random': full dense random orthogonal (maximizes coupling)
    - mode='block': repeated 2x2 rotation blocks by the given angle
    """
    if mode == 'random':
        A = torch.randn(dim, dim, dtype=torch.float32)
        Q, _ = torch.linalg.qr(A)
        # Ensure a proper rotation (det=+1)
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        return Q
    else:
        theta = angle_degrees * np.pi / 180.0
        R = torch.eye(dim, dtype=torch.float32)
        for i in range(0, dim-1, 2):
            c, s = np.cos(theta), np.sin(theta)
            R[i:i+2, i:i+2] = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
        return R

def create_rotated_quadratic_problem(dim: int = 896,
                                    condition_number: float = 1000.0,
                                    rotation_angle: float = 45.0,
                                    rotation_mode: str = 'random',
                                    device: torch.device = torch.device('cpu')) -> tuple:
    """
    Create a rotated quadratic optimization problem.
    
    Returns:
        H: Hessian matrix
        optimal_point: The optimal solution (zeros)
        loss_fn: Loss function
        grad_fn: Gradient function
    """
    # Create diagonal Hessian with specified condition number
    eigenvals = torch.logspace(0, np.log10(condition_number), dim, dtype=torch.float32, device=device)
    eigenvals = eigenvals / eigenvals.mean() * 10.0  # Scale for reasonable loss values
    H_diag = torch.diag(eigenvals)
    
    # Create rotation matrix
    R = create_rotation_matrix(dim, rotation_angle, rotation_mode).to(device)
    
    # Rotated Hessian
    H = R @ H_diag @ R.T
    H = (H + H.T) / 2  # Ensure symmetry
    
    # Optimal point (we'll optimize toward zero)
    optimal_point = torch.zeros(dim, device=device)
    
    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        """Quadratic loss: 0.5 * x^T H x"""
        return 0.5 * torch.dot(x, H @ x)
    
    def grad_fn(x: torch.Tensor) -> torch.Tensor:
        """Gradient: H x"""
        return H @ x
    
    return H, optimal_point, loss_fn, grad_fn

class QuadraticModel(nn.Module):
    """Parameterized quadratic with a 2D parameter for Muon compatibility."""

    def __init__(self, dim: int, init_scale: float = 1.0):
        super().__init__()
        # Choose a near-square matrix shape so rows * cols == dim
        rows = int(np.floor(np.sqrt(dim)))
        while rows > 1 and dim % rows != 0:
            rows -= 1
        cols = dim // rows
        self.shape = (rows, cols)
        # Initialize 2D parameter
        self.W = nn.Parameter(torch.randn(self.shape) * init_scale)

    def forward(self) -> torch.Tensor:
        # Flatten into vector space that H expects
        return self.W.view(-1)

    def get_muon_adamw_params(self):
        # Muon works on 2D params directly
        return [self.W], []

def train_quadratic(H: torch.Tensor,
                   optimizer_name: str,
                   num_steps: int = 200,
                   lr: float = 1e-2,
                   dim: int = 896,
                   device: torch.device = torch.device('cpu'),
                   seed: int = 42) -> tuple[list[float], QuadraticModel, list[float]]:
    """Train on rotated quadratic problem."""
    set_seed(seed)
    
    model = QuadraticModel(dim, init_scale=1.0).to(device)
    
    # Create optimizer
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    elif optimizer_name == 'sgd_momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'adamw':
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
    elif optimizer_name == 'morgn':
        morgn_params, adamw_params = model.get_muon_adamw_params()
        optimizer = MORGN(lr=lr, wd=0.0, morgn_params=morgn_params, adamw_params=adamw_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    losses = []
    distances = []  # Distance from optimum
    
    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        # Get current parameters (flattened)
        x = model()

        # Compute loss (quadratic form)
        loss = 0.5 * torch.dot(x, H @ x)

        loss.backward()
        optimizer.step()
        
        # Record metrics
        losses.append(loss.item())
        distances.append(torch.norm(x).item())
        
        if step % 50 == 0:
            print(f"  Step {step:3d}: Loss = {loss.item():.6f}, Distance = {distances[-1]:.6f}")
    
    return losses, model, distances

def compare_rotated_paraboloid(dim: int = 64,
                              condition_number: float = 100.0,
                              rotation_angle: float = 45.0,
                              rotation_mode: str = 'random',
                              num_steps: int = 200,
                              lr: float = 1e-2,
                              device: torch.device = torch.device('cpu'),
                              seed: int = 42) -> dict:
    """Compare optimizers on rotated paraboloid problem."""
    print(f"\n{'='*70}")
    print(f"Rotated Paraboloid: Optimizer Comparison")
    print(f"{'='*70}")
    print(f"Problem: Minimize rotated quadratic function")
    print(f"Dimension: {dim}")
    print(f"Condition number: {condition_number}")
    print(f"Rotation angle: {rotation_angle}°")
    print(f"Training steps: {num_steps}, Learning rate: {lr}")
    print(f"Random seed: {seed}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Create problem
    set_seed(seed)
    H, optimal_point, loss_fn, grad_fn = create_rotated_quadratic_problem(
        dim, condition_number, rotation_angle, rotation_mode, device
    )
    
    # Analyze problem structure
    eigenvals = torch.linalg.eigvalsh(H)
    print(f"Problem properties:")
    print(f"  Eigenvalue range: [{eigenvals.min().item():.4f}, {eigenvals.max().item():.4f}]")
    print(f"  Actual condition number: {(eigenvals.max() / eigenvals.min()).item():.2f}")
    
    # Compute gradient correlation (off-diagonal dominance)
    H_normalized = H / H.abs().max()
    off_diagonal_sum = (H_normalized.abs() - torch.diag(torch.diag(H_normalized).abs())).sum()
    total_sum = H_normalized.abs().sum()
    correlation_measure = off_diagonal_sum / total_sum
    print(f"  Gradient correlation measure: {correlation_measure.item():.3f}")
    print(f"    (0 = uncorrelated/diagonal, 1 = fully correlated)")
    
    results = {}
    
    # Test each optimizer
    for optimizer_name in ['sgd', 'sgd_momentum', 'adamw', 'muon', 'morgn']:
        print(f"\nTraining with {optimizer_name.upper()}...")
        
        start_time = time.time()
        losses, model, distances = train_quadratic(
            H, optimizer_name, num_steps, lr, dim, device, seed
        )
        train_time = time.time() - start_time
        
        # Compute convergence metrics
        final_loss = losses[-1]
        final_distance = distances[-1]
        
        # Find steps to convergence (within 1% of initial loss)
        initial_loss = max(losses[0], 1e-12)
        threshold_loss = initial_loss * 0.01
        steps_to_converge = num_steps
        for i, loss in enumerate(losses):
            if loss < threshold_loss:
                steps_to_converge = i
                break
        
        results[optimizer_name] = {
            'losses': losses,
            'distances': distances,
            'final_loss': final_loss,
            'final_distance': final_distance,
            'steps_to_converge': steps_to_converge,
            'train_time': train_time,
            'model': model
        }
        
        print(f"  {optimizer_name.upper()} Results:")
        print(f"    Final loss: {final_loss:.6f}")
        print(f"    Final distance from optimum: {final_distance:.6f}")
        print(f"    Steps to 1% of initial: {steps_to_converge}")
        print(f"    Training time: {train_time:.2f}s")
    
    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    # Find best performer by convergence speed
    convergence_times = {name: results[name]['steps_to_converge'] for name in results}
    fastest = min(convergence_times, key=convergence_times.get)
    print(f"🚀 Fastest convergence: {fastest.upper()} ({convergence_times[fastest]} steps)")
    
    # Compare relative performance
    for name in results:
        if name != fastest:
            speedup = convergence_times[name] / convergence_times[fastest]
            print(f"   {fastest.upper()} is {speedup:.1f}x faster than {name.upper()}")
    
    # Best final loss
    final_losses = {name: results[name]['final_loss'] for name in results}
    best_loss = min(final_losses, key=final_losses.get)
    print(f"\n🏆 Best final loss: {best_loss.upper()} ({final_losses[best_loss]:.6f})")
    
    return results

def plot_rotated_paraboloid_comparison(results: dict, save_path: str = 'rotated_paraboloid_comparison.png') -> None:
    """Plot convergence curves for all optimizers."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping plot generation (matplotlib not installed)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Loss curves
    for name, data in results.items():
        losses = data['losses']
        axes[0].plot(losses, label=f"{name.upper()}", linewidth=2)
    
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Rotated Paraboloid: Loss Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: Distance from optimum
    for name, data in results.items():
        distances = data['distances']
        axes[1].plot(distances, label=f"{name.upper()}", linewidth=2)
    
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Distance from Optimum')
    axes[1].set_title('Distance to Solution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare optimizers on rotated paraboloid')
    parser.add_argument('--dim', type=int, default=64,
                       help='Problem dimension (default: 64)')
    parser.add_argument('--condition-number', type=float, default=100.0,
                       help='Condition number of Hessian (default: 100.0)')
    parser.add_argument('--rotation-angle', type=float, default=45.0,
                       help='Rotation angle in degrees (default: 45.0)')
    parser.add_argument('--steps', type=int, default=200,
                       help='Number of training steps (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-2,
                       help='Learning rate (default: 1e-2)')
    parser.add_argument('--rotation-mode', type=str, default='random',
                       choices=['random', 'block'],
                       help='Rotation type: random dense or block 2x2')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use (default: cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Run comparison
    results = compare_rotated_paraboloid(
        dim=args.dim,
        condition_number=args.condition_number,
        rotation_angle=args.rotation_angle,
        rotation_mode=args.rotation_mode,
        num_steps=args.steps,
        lr=args.lr,
        device=device,
        seed=args.seed
    )
    
    # Generate plots if requested
    if args.plot:
        plot_rotated_paraboloid_comparison(results)
    
    print("\nRotated paraboloid experiment complete!")
    print("\nKey insight: This problem tests how well optimizers handle")
    print("correlated gradients and misaligned curvature directions.")
    print("Muon should excel here due to its Newton-like behavior.")

if __name__ == "__main__":
    main()
