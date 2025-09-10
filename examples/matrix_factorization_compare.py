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

def create_positive_definite_matrix(size: int,
                                    condition_number: float = 100.0,
                                    device: torch.device = torch.device('cpu'),
                                    evec_mode: str = 'random') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a positive definite matrix with specified condition number.

    evec_mode:
      - 'random': random orthogonal eigenvectors (Q from QR)
      - 'identity': coordinate-aligned eigenvectors (Q = I)
    Returns (target_matrix, eigenvalues, Q)
    """
    # Choose device for QR (CPU is the safest across backends)
    qr_device = 'cpu' if device.type == 'mps' else device

    if evec_mode == 'identity':
        Q = torch.eye(size, device=qr_device)
    else:
        A = torch.randn(size, size, device=qr_device)
        Q, _ = torch.linalg.qr(A)
        # Ensure det(Q)=+1 for a proper rotation
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0] # Negate first column to ensure det(Q)=+1

    # Eigenvalues with desired condition number
    eigenvals = torch.logspace(0, np.log10(condition_number), size, device=qr_device)
    eigenvals = eigenvals / eigenvals.sum() * size  # Normalize scale
    D = torch.diag(eigenvals)

    # Construct SPD matrix
    target_matrix = Q @ D @ Q.T
    return target_matrix.to(device), eigenvals.to(device), Q.to(device)

def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def check_gradient_anomalies(model: nn.Module, step: int, threshold: float = 1000.0) -> dict:
    """Check for gradient anomalies and return summary."""
    anomalies = {
        'has_nan': False,
        'has_inf': False,
        'large_grads': [],
        'zero_grads': [],
        'grad_norms': {}
    }
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_norm = grad.norm().item()
            anomalies['grad_norms'][name] = grad_norm
            
            if torch.isnan(grad).any():
                anomalies['has_nan'] = True
                print(f"WARNING: NaN gradients in {name} at step {step}")
            
            if torch.isinf(grad).any():
                anomalies['has_inf'] = True
                print(f"WARNING: Inf gradients in {name} at step {step}")
            
            if grad_norm > threshold:
                anomalies['large_grads'].append((name, grad_norm))
                print(f"WARNING: Large gradients in {name} at step {step}: {grad_norm:.6f}")
            
            if grad_norm < 1e-8:
                anomalies['zero_grads'].append((name, grad_norm))
        else:
            print(f"WARNING: No gradients for {name} at step {step}")
    
    return anomalies

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


class SymmetricFactorizationModel(nn.Module):
    """Learn a single factor W such that W @ W.T ≈ target_matrix (SPD factorization)."""

    def __init__(self, matrix_size: int):
        super().__init__()
        self.matrix_size = matrix_size
        self.rank = matrix_size
        # Initialize close to scaled identity to avoid singularities
        self.W = nn.Parameter(torch.eye(matrix_size) + 0.01 * torch.randn(matrix_size, matrix_size))

    def forward(self) -> torch.Tensor:
        return self.W @ self.W.T

    def get_muon_adamw_params(self):
        return [self.W], []

def train_factorization(target_matrix: torch.Tensor, optimizer_name: str,
                       num_steps: int = 200, lr: float = 1e-2,
                       rank: int = None, device: torch.device = torch.device('cpu'),
                       seed: int = 42,
                       orthogonal_reg: float = 0.0,
                       muon_lr: float | None = None,
                       muon_ns_steps: int = 6,
                       muon_lr_warmdown_at: float = 0.7,
                       muon_lr_decay_factor: float = 0.1,
                       morgn_lr: float | None = None,
                       morgn_lambda: float = 0.99,
                       morgn_eps: float = 1e-3,
                       morgn_directions: int = 8,
                       morgn_two_sided: bool = False,
                       morgn_right_lambda: float | None = None,
                       morgn_right_directions: int | None = None,
                       morgn_precond_warmup_steps: int = 0,
                       morgn_precond_warmup_exp: float = 1.0,
                       morgn_momentum: float = 0.0,
                       morgn_nesterov: bool = False,
                       morgn_clamp_warmdown_at: float = 0.0,
                       morgn_clamp_decay_factor: float = 1.0,
                       morgn_lr_warmdown_at: float = 0.0,
                       morgn_lr_decay_factor: float = 1.0,
                       morgn_lr_warmdown2_at: float = 0.0,
                       morgn_lr_decay2_factor: float = 1.0,
                       morgn_step_clamp: float = 0.0,
                       morgn_analytic_gn: bool = False,
                       morgn_analytic_gamma: float = 1e-3,
                       clip_grad_norm: float = 0.0) -> tuple[list[float], MatrixFactorizationModel]:
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
            lr=muon_lr if muon_lr is not None else lr,
            wd=0.0,
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=0.95,
            ns_steps=muon_ns_steps
        )
    else:  # morgn
        morgn_params, adamw_params = model.get_muon_adamw_params()
        optimizer = MORGN(
            lr=morgn_lr if morgn_lr is not None else lr,
            wd=0.0,
            morgn_params=morgn_params,
            lambda_=morgn_lambda,
            eps=morgn_eps,
            directions=morgn_directions,
            two_sided=morgn_two_sided,
            right_lambda=morgn_right_lambda,
            right_directions=morgn_right_directions,
            precond_warmup_steps=morgn_precond_warmup_steps,
            precond_warmup_exp=morgn_precond_warmup_exp,
            momentum=morgn_momentum,
            nesterov=morgn_nesterov,
            analytic_gn=morgn_analytic_gn,
            analytic_gamma=morgn_analytic_gamma,
            step_clamp=morgn_step_clamp,
            adamw_params=adamw_params,
        )
    
    losses = []
    grad_norms = []
    criterion = nn.MSELoss()
    
    for step in range(num_steps):
        # One-time Muon warmdown
        if optimizer_name == 'muon':
            warm_step = int(muon_lr_warmdown_at * num_steps)
            if step == warm_step:
                for pg in optimizer.param_groups:
                    old = pg['lr']
                    pg['lr'] = old * muon_lr_decay_factor
                print(f"[MUON] Warmdown at step {step}: LR scaled by {muon_lr_decay_factor}")
        # One-time MORGN clamp warmdown
        if optimizer_name == 'morgn' and morgn_clamp_warmdown_at and morgn_clamp_warmdown_at > 0.0:
            cstep = int(morgn_clamp_warmdown_at * num_steps)
            if step == cstep:
                for pg in optimizer.param_groups:
                    if 'step_clamp' in pg:
                        oldc = pg['step_clamp']
                        pg['step_clamp'] = oldc * morgn_clamp_decay_factor
                print(f"[MORGN] Clamp warmdown at step {step}: clamp scaled by {morgn_clamp_decay_factor}")
        # One-time MORGN LR warmdown
        if optimizer_name == 'morgn' and morgn_lr_warmdown_at and morgn_lr_warmdown_at > 0.0:
            lstep = int(morgn_lr_warmdown_at * num_steps)
            if step == lstep:
                for pg in optimizer.param_groups:
                    oldlr = pg['lr']
                    pg['lr'] = oldlr * morgn_lr_decay_factor
                print(f"[MORGN] Warmdown at step {step}: LR scaled by {morgn_lr_decay_factor}")
        if optimizer_name == 'morgn' and morgn_lr_warmdown2_at and morgn_lr_warmdown2_at > 0.0:
            lstep2 = int(morgn_lr_warmdown2_at * num_steps)
            if step == lstep2:
                for pg in optimizer.param_groups:
                    oldlr = pg['lr']
                    pg['lr'] = oldlr * morgn_lr_decay2_factor
                print(f"[MORGN] Warmdown2 at step {step}: LR scaled by {morgn_lr_decay2_factor}")

        optimizer.zero_grad()
        
        # Compute reconstruction
        reconstructed = model()
        
        # Loss: MSE between target and reconstruction
        loss = criterion(reconstructed, target_matrix)

        # Optional orthogonality regularization to favor orthonormal factors
        if orthogonal_reg > 0.0:
            I_A = torch.eye(model.rank, device=device)
            I_B = torch.eye(model.rank, device=device)
            reg_A = torch.norm(model.factor_A.T @ model.factor_A - I_A, p='fro')**2
            reg_B = torch.norm(model.factor_B.T @ model.factor_B - I_B, p='fro')**2
            loss = loss + orthogonal_reg * (reg_A + reg_B)
        
        # Additional loss to encourage positive definiteness (optional)
        # We could add regularization here, but for now keep it simple
        
        loss.backward()
        
        # Gradient monitoring
        grad_norm = compute_grad_norm(model)
        anomalies = check_gradient_anomalies(model, step, threshold=1000.0)
        
        # Gradient clipping with monitoring
        if clip_grad_norm and clip_grad_norm > 0.0:
            actual_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            if actual_norm > clip_grad_norm:
                print(f"  Step {step}: Clipped gradient norm from {actual_norm:.6f} to {clip_grad_norm}")
        
        optimizer.step()
        
        losses.append(loss.item())
        grad_norms.append(grad_norm)
        
        if step % 50 == 0:
            print(f"  Step {step:3d}: Loss = {loss.item():.6f}, Grad Norm = {grad_norm:.6f}")
            
            # Report any gradient anomalies every 50 steps
            if anomalies['large_grads']:
                print(f"    Large gradients detected: {len(anomalies['large_grads'])} parameters")
            if anomalies['zero_grads']:
                print(f"    Very small gradients detected: {len(anomalies['zero_grads'])} parameters")
    
    return losses, model


def train_symmetric_factorization(target_matrix: torch.Tensor, optimizer_name: str,
                                  num_steps: int = 200, lr: float = 1e-2,
                                  device: torch.device = torch.device('cpu'), seed: int = 42,
                                  muon_lr: float | None = None, muon_ns_steps: int = 6,
                                  muon_lr_warmdown_at: float = 0.7,
                                  muon_lr_decay_factor: float = 0.1,
                                  morgn_lr: float | None = None, morgn_lambda: float = 0.99,
                                  morgn_eps: float = 1e-3,
                                  morgn_directions: int = 8,
                                  morgn_two_sided: bool = False,
                                  morgn_right_lambda: float | None = None,
                                  morgn_right_directions: int | None = None,
                                  morgn_precond_warmup_steps: int = 0,
                                  morgn_precond_warmup_exp: float = 1.0,
                                  morgn_momentum: float = 0.0,
                                  morgn_nesterov: bool = False,
                                  morgn_clamp_warmdown_at: float = 0.0,
                                  morgn_clamp_decay_factor: float = 1.0,
                                  morgn_lr_warmdown_at: float = 0.0,
                                  morgn_lr_decay_factor: float = 1.0,
                                  morgn_lr_warmdown2_at: float = 0.0,
                                  morgn_lr_decay2_factor: float = 1.0,
                                  morgn_step_clamp: float = 0.0,
                                  morgn_analytic_gn: bool = False,
                                  morgn_analytic_gamma: float = 1e-3,
                                  clip_grad_norm: float = 0.0) -> tuple[list[float], SymmetricFactorizationModel]:
    """Train SPD factorization W such that W W^T ≈ target_matrix."""
    set_seed(seed)

    matrix_size = target_matrix.size(0)
    model = SymmetricFactorizationModel(matrix_size).to(device)

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    elif optimizer_name == 'muon':
        muon_params, adamw_params = model.get_muon_adamw_params()
        optimizer = Muon(
            lr=muon_lr if muon_lr is not None else lr,
            wd=0.0,
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=0.95,
            ns_steps=muon_ns_steps,
        )
    else:  # morgn
        morgn_params, adamw_params = model.get_muon_adamw_params()
        optimizer = MORGN(
            lr=morgn_lr if morgn_lr is not None else lr,
            wd=0.0,
            morgn_params=morgn_params,
            lambda_=morgn_lambda,
            eps=morgn_eps,
            directions=morgn_directions,
            two_sided=morgn_two_sided,
            right_lambda=morgn_right_lambda,
            right_directions=morgn_right_directions,
            precond_warmup_steps=morgn_precond_warmup_steps,
            precond_warmup_exp=morgn_precond_warmup_exp,
            momentum=morgn_momentum,
            nesterov=morgn_nesterov,
            analytic_gn=morgn_analytic_gn,
            analytic_gamma=morgn_analytic_gamma,
            step_clamp=morgn_step_clamp,
            adamw_params=adamw_params,
        )

    criterion = nn.MSELoss()
    losses = []
    grad_norms = []
    
    for step in range(num_steps):
        if optimizer_name == 'muon':
            warm_step = int(muon_lr_warmdown_at * num_steps)
            if step == warm_step:
                for pg in optimizer.param_groups:
                    old = pg['lr']
                    pg['lr'] = old * muon_lr_decay_factor
                print(f"[MUON] Warmdown at step {step}: LR scaled by {muon_lr_decay_factor}")
        if optimizer_name == 'morgn' and morgn_clamp_warmdown_at and morgn_clamp_warmdown_at > 0.0:
            cstep = int(morgn_clamp_warmdown_at * num_steps)
            if step == cstep:
                for pg in optimizer.param_groups:
                    if 'step_clamp' in pg:
                        oldc = pg['step_clamp']
                        pg['step_clamp'] = oldc * morgn_clamp_decay_factor
                print(f"[MORGN] Clamp warmdown at step {step}: clamp scaled by {morgn_clamp_decay_factor}")
        if optimizer_name == 'morgn' and morgn_lr_warmdown_at and morgn_lr_warmdown_at > 0.0:
            lstep = int(morgn_lr_warmdown_at * num_steps)
            if step == lstep:
                for pg in optimizer.param_groups:
                    oldlr = pg['lr']
                    pg['lr'] = oldlr * morgn_lr_decay_factor
                print(f"[MORGN] Warmdown at step {step}: LR scaled by {morgn_lr_decay_factor}")
        if optimizer_name == 'morgn' and morgn_lr_warmdown2_at and morgn_lr_warmdown2_at > 0.0:
            lstep2 = int(morgn_lr_warmdown2_at * num_steps)
            if step == lstep2:
                for pg in optimizer.param_groups:
                    oldlr = pg['lr']
                    pg['lr'] = oldlr * morgn_lr_decay2_factor
                print(f"[MORGN] Warmdown2 at step {step}: LR scaled by {morgn_lr_decay2_factor}")
        
        optimizer.zero_grad(set_to_none=True)
        reconstructed = model()  # Call 
        loss = criterion(reconstructed, target_matrix)
        loss.backward()
        
        # Gradient monitoring
        grad_norm = compute_grad_norm(model)
        anomalies = check_gradient_anomalies(model, step, threshold=1000.0)
        
        # Gradient clipping with monitoring
        if clip_grad_norm and clip_grad_norm > 0.0:
            actual_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            if actual_norm > clip_grad_norm:
                print(f"  Step {step}: Clipped gradient norm from {actual_norm:.6f} to {clip_grad_norm}")
        
        optimizer.step()
        losses.append(loss.item())
        grad_norms.append(grad_norm)
        
        if step % 50 == 0:
            print(f"  Step {step:3d}: Loss = {loss.item():.6f}, Grad Norm = {grad_norm:.6f}")
            
            # Report any gradient anomalies every 50 steps
            if anomalies['large_grads']:
                print(f"    Large gradients detected: {len(anomalies['large_grads'])} parameters")
            if anomalies['zero_grads']:
                print(f"    Very small gradients detected: {len(anomalies['zero_grads'])} parameters")

    return losses, model

def analyze_reconstruction_quality(model: nn.Module, target_matrix: torch.Tensor, symmetric: bool = False) -> dict:
    """Analyze reconstruction quality; robust to NaNs and uses symmetric eigendecomp when applicable."""
    with torch.no_grad():
        reconstructed = model()

        # Basic errors
        diff = reconstructed - target_matrix
        mse_error = torch.mean(diff.pow(2)).item()
        frobenius_error = torch.norm(diff, 'fro').item()
        denom = torch.norm(target_matrix, 'fro').item()
        relative_error = (frobenius_error / denom) if denom > 0 else float('inf')

        # NaN/Inf guard
        if not torch.isfinite(reconstructed).all():
            return {
                'mse_error': mse_error,
                'frobenius_error': frobenius_error,
                'relative_error': relative_error,
                'min_eigval_target': float('nan'),
                'min_eigval_recon': float('nan'),
                'is_pos_def_target': False,
                'is_pos_def_recon': False,
            }

        # Eigenvalues on CPU for maximum backend stability
        tgt = target_matrix.detach().cpu()
        rec = reconstructed.detach().cpu()

        try:
            if symmetric:
                eigenvals_target = torch.linalg.eigvalsh(tgt)
                eigenvals_recon = torch.linalg.eigvalsh(rec)
            else:
                eigenvals_target = torch.linalg.eigvals(tgt).real
                eigenvals_recon = torch.linalg.eigvals(rec).real
            min_eigval_target = float(eigenvals_target.min().item())
            min_eigval_recon = float(eigenvals_recon.min().item())
        except Exception:
            # Fallback if LAPACK complains
            min_eigval_target = float('nan')
            min_eigval_recon = float('nan')

        tol = 1e-8
        is_pos_def_target = (min_eigval_target > tol) if np.isfinite(min_eigval_target) else False
        is_pos_def_recon = (min_eigval_recon > tol) if np.isfinite(min_eigval_recon) else False

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
                                seed: int = 42,
                                orthogonal_reg: float = 0.0,
                                muon_lr: float | None = None,
                                muon_ns_steps: int = 6,
                                muon_lr_warmdown_at: float = 0.7,
                                muon_lr_decay_factor: float = 0.1,
                                symmetric: bool = False,
                                evec_mode: str = 'random',
                                morgn_lr: float | None = None,
                                morgn_lambda: float = 0.99,
                                morgn_eps: float = 1e-3,
                                morgn_directions: int = 8,
                                morgn_two_sided: bool = False,
                                morgn_right_lambda: float | None = None,
                                morgn_right_directions: int | None = None,
                                morgn_precond_warmup_steps: int = 0,
                                morgn_precond_warmup_exp: float = 1.0,
                                morgn_momentum: float = 0.0,
                                morgn_nesterov: bool = False,
                                morgn_clamp_warmdown_at: float = 0.0,
                                morgn_clamp_decay_factor: float = 1.0,
                                morgn_lr_warmdown_at: float = 0.0,
                                morgn_lr_decay_factor: float = 1.0,
                                morgn_lr_warmdown2_at: float = 0.0,
                                morgn_lr_decay2_factor: float = 1.0,
                                morgn_analytic_gn: bool = False,
                                morgn_analytic_gamma: float = 1e-3,
                                morgn_step_clamp: float = 0.0,
                                clip_grad_norm: float = 0.0) -> dict:
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
    target_matrix, eigenvals, Q = create_positive_definite_matrix(
        matrix_size, condition_number, device, evec_mode=evec_mode
    )
    
    print(f"Target matrix properties:")
    print(f"  Eigenvalue range: [{eigenvals.min().item():.4f}, {eigenvals.max().item():.4f}]")
    print(f"  Condition number: {(eigenvals.max() / eigenvals.min()).item():.2f}")
    print(f"  Frobenius norm: {torch.norm(target_matrix, 'fro').item():.4f}")
    
    print(f"Eigenvector mode: {evec_mode} ({'aligned' if evec_mode=='identity' else 'rotated'})")
    results = {}
    
    for optimizer_name in ['adamw', 'muon', 'morgn']:
        print(f"\nTraining with {optimizer_name.upper()}...")
        
        start_time = time.time()
        # Helpful summary of MORGN settings
        if optimizer_name == 'morgn':
            rl = morgn_right_lambda if morgn_right_lambda is not None else morgn_lambda
            rd = morgn_right_directions if morgn_right_directions is not None else morgn_directions
            print("  MORGN settings:")
            print(f"    lr={morgn_lr if morgn_lr is not None else lr}, lambda={morgn_lambda}, eps={morgn_eps}")
            print(f"    directions={morgn_directions}, step_clamp={morgn_step_clamp}, two_sided={morgn_two_sided}")
            if morgn_two_sided:
                print(f"    right_lambda={rl}, right_directions={rd}")
            if morgn_precond_warmup_steps > 0:
                print(f"    precond_warmup_steps={morgn_precond_warmup_steps}, exp={morgn_precond_warmup_exp}")
            if (morgn_momentum or 0.0) > 0:
                print(f"    momentum={morgn_momentum}, nesterov={morgn_nesterov}")
            if morgn_clamp_warmdown_at and morgn_clamp_warmdown_at > 0.0:
                print(f"    clamp_warmdown_at={morgn_clamp_warmdown_at}, clamp_decay={morgn_clamp_decay_factor}")
            if morgn_lr_warmdown_at and morgn_lr_warmdown_at > 0.0:
                print(f"    lr_warmdown_at={morgn_lr_warmdown_at}, lr_decay={morgn_lr_decay_factor}")
            if morgn_lr_warmdown2_at and morgn_lr_warmdown2_at > 0.0:
                print(f"    lr_warmdown2_at={morgn_lr_warmdown2_at}, lr_decay2={morgn_lr_decay2_factor}")
            if morgn_analytic_gn:
                print(f"    analytic_gn=True, gamma={morgn_analytic_gamma}")
        if symmetric:
            losses, model = train_symmetric_factorization(
                target_matrix, optimizer_name, num_steps, lr, device, seed,
                muon_lr=muon_lr, muon_ns_steps=muon_ns_steps,
                muon_lr_warmdown_at=muon_lr_warmdown_at,
                muon_lr_decay_factor=muon_lr_decay_factor,
                morgn_lr=morgn_lr, morgn_lambda=morgn_lambda, morgn_eps=morgn_eps,
                morgn_directions=morgn_directions, morgn_two_sided=morgn_two_sided,
                morgn_right_lambda=morgn_right_lambda, morgn_right_directions=morgn_right_directions,
                morgn_precond_warmup_steps=morgn_precond_warmup_steps,
                morgn_precond_warmup_exp=morgn_precond_warmup_exp,
                morgn_momentum=morgn_momentum,
                morgn_nesterov=morgn_nesterov,
                morgn_clamp_warmdown_at=morgn_clamp_warmdown_at,
                morgn_clamp_decay_factor=morgn_clamp_decay_factor,
                morgn_step_clamp=morgn_step_clamp,
                clip_grad_norm=clip_grad_norm
            )
        else:
            losses, model = train_factorization(
                target_matrix, optimizer_name, num_steps, lr, rank, device, seed,
                orthogonal_reg=orthogonal_reg,
                muon_lr=muon_lr,
                muon_ns_steps=muon_ns_steps,
                muon_lr_warmdown_at=muon_lr_warmdown_at,
                muon_lr_decay_factor=muon_lr_decay_factor,
                morgn_lr=morgn_lr, morgn_lambda=morgn_lambda, morgn_eps=morgn_eps,
                morgn_directions=morgn_directions, morgn_two_sided=morgn_two_sided,
                morgn_right_lambda=morgn_right_lambda, morgn_right_directions=morgn_right_directions,
                morgn_precond_warmup_steps=morgn_precond_warmup_steps,
                morgn_precond_warmup_exp=morgn_precond_warmup_exp,
                morgn_momentum=morgn_momentum,
                morgn_nesterov=morgn_nesterov,
                morgn_clamp_warmdown_at=morgn_clamp_warmdown_at,
                morgn_clamp_decay_factor=morgn_clamp_decay_factor,
                morgn_step_clamp=morgn_step_clamp,
                clip_grad_norm=clip_grad_norm
            )
        train_time = time.time() - start_time
        
        # Analyze reconstruction quality
        quality_metrics = analyze_reconstruction_quality(model, target_matrix, symmetric=symmetric)
        
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
    parser.add_argument('--orthogonal-reg', type=float, default=0.0,
                       help='Orthogonality regularization weight (default: 0.0)')
    parser.add_argument('--muon-lr', type=float, default=None,
                       help='Override learning rate for Muon (default: same as --lr)')
    parser.add_argument('--muon-ns-steps', type=int, default=6,
                       help='Newton-Schulz steps for Muon (default: 6)')
    parser.add_argument('--muon-lr-warmdown-at', type=float, default=0.7,
                       help='Fraction of steps when Muon LR is decayed (default: 0.7)')
    parser.add_argument('--muon-lr-decay-factor', type=float, default=0.1,
                       help='Scale Muon LR by this factor at warmdown (default: 0.1)')
    # MORGN-specific flags
    parser.add_argument('--morgn-lr', type=float, default=None,
                       help='Override learning rate for MORGN (default: same as --lr)')
    parser.add_argument('--morgn-lambda', type=float, default=0.995,
                       help='MORGN forgetting factor lambda (default: 0.995)')
    parser.add_argument('--morgn-eps', type=float, default=1e-2,
                       help='MORGN initial inverse scale epsilon (P0=(1/eps)I)')
    parser.add_argument('--morgn-step-clamp', type=float, default=0.1,
                       help='Clamp step Frobenius norm to this fraction of ||W|| (default: 0.1)')
    parser.add_argument('--morgn-directions', type=int, default=8,
                       help='Number of gradient columns assimilated per step by MORGN (default: 8)')
    parser.add_argument('--morgn-two-sided', action='store_true',
                       help='Enable two-sided preconditioning: ΔW = P G Q with right-side RLS update')
    parser.add_argument('--morgn-right-lambda', type=float, default=None,
                       help='Right-side forgetting factor (defaults to --morgn-lambda)')
    parser.add_argument('--morgn-right-directions', type=int, default=None,
                       help='Right-side number of directions (defaults to --morgn-directions)')
    parser.add_argument('--morgn-precond-warmup-steps', type=int, default=0,
                       help='Blend from SGD to preconditioned step over this many steps')
    parser.add_argument('--morgn-precond-warmup-exp', type=float, default=1.0,
                       help='Exponent for warmup blend t^exp (1.0=linear)')
    parser.add_argument('--morgn-momentum', type=float, default=0.0,
                       help='Gradient momentum coefficient for MORGN (0 disables)')
    parser.add_argument('--morgn-nesterov', action='store_true',
                       help='Use Nesterov-style momentum in MORGN')
    parser.add_argument('--morgn-clamp-warmdown-at', type=float, default=0.0,
                       help='Fraction of steps when MORGN clamp is decayed (0 disables)')
    parser.add_argument('--morgn-clamp-decay-factor', type=float, default=1.0,
                       help='Multiply MORGN clamp by this factor at warmdown time')
    parser.add_argument('--morgn-lr-warmdown-at', type=float, default=0.0,
                       help='Fraction of steps when MORGN LR is decayed (0 disables)')
    parser.add_argument('--morgn-lr-decay-factor', type=float, default=1.0,
                       help='Multiply MORGN LR by this factor at warmdown time')
    # Optional second-stage LR warmdown (to mimic Muon plateau break)
    parser.add_argument('--morgn-lr-warmdown2-at', type=float, default=0.0,
                       help='Second-stage MORGN LR decay time (fraction of steps, 0 disables)')
    parser.add_argument('--morgn-lr-decay2-factor', type=float, default=1.0,
                       help='Multiply LR by this factor at second warmdown time')
    parser.add_argument('--morgn-analytic-gn', action='store_true',
                       help='Use analytic two-sided Gauss-Newton preconditioner for SPD tasks')
    parser.add_argument('--morgn-analytic-gamma', type=float, default=1e-3,
                       help='Damping gamma for analytic GN preconditioner')
    # Global gradient clipping
    parser.add_argument('--clip-grad-norm', type=float, default=0.0,
                       help='Clip gradient norm to this value (0 disables)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
    parser.add_argument('--out', type=str, default=None,
                       help='Output plot filename (auto if not specified)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--symmetric', action='store_true',
                       help='Use symmetric factorization W W^T ≈ T (SPD)')
    parser.add_argument('--evec-mode', type=str, default='random',
                       choices=['random', 'identity'],
                       help='Eigenvector mode: random rotation or coordinate-aligned')
    
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
        seed=args.seed,
        orthogonal_reg=args.orthogonal_reg,
        muon_lr=args.muon_lr,
        muon_ns_steps=args.muon_ns_steps,
        muon_lr_warmdown_at=args.muon_lr_warmdown_at,
        muon_lr_decay_factor=args.muon_lr_decay_factor,
        symmetric=args.symmetric,
        evec_mode=args.evec_mode,
        morgn_lr=args.morgn_lr,
        morgn_lambda=args.morgn_lambda,
        morgn_eps=args.morgn_eps,
        morgn_directions=args.morgn_directions,
        morgn_two_sided=args.morgn_two_sided,
        morgn_right_lambda=args.morgn_right_lambda,
        morgn_right_directions=args.morgn_right_directions,
        morgn_precond_warmup_steps=args.morgn_precond_warmup_steps,
        morgn_precond_warmup_exp=args.morgn_precond_warmup_exp,
        morgn_momentum=args.morgn_momentum,
        morgn_nesterov=args.morgn_nesterov,
        morgn_clamp_warmdown_at=args.morgn_clamp_warmdown_at,
        morgn_clamp_decay_factor=args.morgn_clamp_decay_factor,
        morgn_lr_warmdown_at=args.morgn_lr_warmdown_at,
        morgn_lr_decay_factor=args.morgn_lr_decay_factor,
        morgn_lr_warmdown2_at=args.morgn_lr_warmdown2_at,
        morgn_lr_decay2_factor=args.morgn_lr_decay2_factor,
        morgn_step_clamp=args.morgn_step_clamp,
        morgn_analytic_gn=args.morgn_analytic_gn,
        morgn_analytic_gamma=args.morgn_analytic_gamma,
        clip_grad_norm=args.clip_grad_norm
    )
    
    # Generate plots if requested
    if args.plot:
        if args.out:
            out_path = args.out
        else:
            mode = 'sym' if args.symmetric else 'rect'
            evec = args.evec_mode
            out_path = f"{mode}_{evec}_matrix_factorization_n{args.matrix_size}_k{int(args.condition_number)}_s{args.steps}.png"
        plot_matrix_factorization_comparison(results, save_path=out_path)
    
    print("\nMatrix factorization experiment complete!")
    print("\nKey insight: This three-way comparison shows how different")
    print("optimizers perform on positive-definite matrix factorization.")

if __name__ == "__main__":
    main()
