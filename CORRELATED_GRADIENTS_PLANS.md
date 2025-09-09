# Correlated Gradients Test Problems for Optimizer Comparison

## Motivation

The matrix factorization problem revealed that AdamW significantly outperforms Muon on certain tasks. This suggests we need test problems that better showcase when orthogonal gradient updates (Muon) or novel optimization techniques (MORGN) provide advantages.

The key insight: **Muon approximates Newton's method** by normalizing the curvature via orthogonalization. It should excel when:
1. Gradients are highly correlated
2. The loss surface has misaligned principal curvature directions
3. Standard gradient descent would zigzag due to poor conditioning

## Test Problem Suite

### 1. Rotated Paraboloid (Primary Test) ⭐

**Concept**: Take an axis-aligned quadratic and rotate it to misalign gradients with curvature.

```python
def create_rotated_quadratic_problem(dim=896, condition_number=1000.0, rotation_angle=45):
    # Diagonal Hessian with specified condition number
    eigenvals = torch.logspace(0, np.log10(condition_number), dim)
    H_diag = torch.diag(eigenvals)
    
    # Rotate in each 2D plane
    R = create_rotation_matrix(dim, rotation_angle)
    H = R @ H_diag @ R.T
    
    def loss_fn(x, target=torch.zeros(dim)):
        return 0.5 * (x - target).T @ H @ (x - target)
    
    return H, loss_fn
```

**Why it's good**:
- Direct test of Newton approximation
- Controllable difficulty via rotation angle and condition number
- Clean interpretation: measures ability to follow curvature

**Expected results**:
- AdamW: Slow zigzagging convergence
- Muon: Should approximate Newton and converge quickly
- MORGN: Depends on implementation

### 2. Rosenbrock Valley

**Concept**: Classic optimization test with narrow curved valley.

```python
def rosenbrock_loss(params, a=1, b=100):
    # Can extend to N-dimensional version
    result = 0
    for i in range(len(params)-1):
        result += (a - params[i])**2 + b*(params[i+1] - params[i]**2)**2
    return result
```

**Why it's good**:
- Highly non-convex with strong correlation structure
- Tests ability to navigate curved valleys
- Standard benchmark in optimization literature

### 3. Ill-Conditioned Least Squares

**Concept**: Regression with highly correlated features.

```python
def correlated_regression_problem(dim=896, correlation=0.95, n_samples=10000):
    # Correlation matrix: exponentially decaying correlations
    Sigma = correlation ** torch.abs(torch.arange(dim).unsqueeze(0) - 
                                     torch.arange(dim).unsqueeze(1))
    L = torch.linalg.cholesky(Sigma)
    
    # Correlated design matrix
    X = torch.randn(n_samples, dim) @ L.T
    w_true = torch.randn(dim)
    y = X @ w_true + 0.1 * torch.randn(n_samples)
    
    # Hessian is X^T X (highly ill-conditioned)
    def loss_fn(w):
        return torch.mean((X @ w - y)**2)
    
    return X, y, w_true, loss_fn
```

**Why it's good**:
- Realistic ML problem structure
- Gradient correlation comes from data correlation
- Tests practical benefit on regression tasks

### 4. Neural Network with Correlated Inputs

**Concept**: Train small MLP on data with correlated features.

```python
def create_correlated_mlp_problem(input_dim=64, hidden_dim=128, output_dim=10, 
                                  correlation=0.9):
    # Generate training data with correlated inputs
    Sigma = correlation ** torch.abs(torch.arange(input_dim).unsqueeze(0) - 
                                     torch.arange(input_dim).unsqueeze(1))
    L = torch.linalg.cholesky(Sigma)
    
    X_train = torch.randn(1000, input_dim) @ L.T
    y_train = torch.randint(0, output_dim, (1000,))
    
    # MLP model
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    
    return model, X_train, y_train
```

**Why it's good**:
- Tests optimizer on actual neural network
- Correlation propagates through layers
- More realistic than pure quadratic problems

### 5. Eigenvalue Decomposition as Optimization

**Concept**: Frame eigenvalue problem as optimization (Oja's rule style).

```python
def create_eigenvalue_problem(matrix_size=896, condition_number=1000):
    # Create symmetric matrix with known eigenstructure
    eigenvals = torch.logspace(0, np.log10(condition_number), matrix_size)
    Q, _ = torch.linalg.qr(torch.randn(matrix_size, matrix_size))
    A = Q @ torch.diag(eigenvals) @ Q.T
    
    # Find top eigenvector via maximization
    def loss_fn(v):
        v_norm = v / torch.norm(v)
        return -v_norm.T @ A @ v_norm  # Negative for maximization
    
    return A, loss_fn
```

**Why it's good**:
- Tests ability to find dominant direction
- Gradient points away from optimum initially
- Benefits from curvature-aware updates

### 6. Coupled Oscillator System

**Concept**: Optimize parameters of coupled harmonic oscillators.

```python
def create_coupled_oscillator_problem(n_oscillators=32, coupling_strength=0.5):
    # Mass matrix (diagonal)
    M = torch.eye(n_oscillators)
    
    # Stiffness matrix (tridiagonal with coupling)
    K = 2 * torch.eye(n_oscillators)
    K += -coupling_strength * (torch.diag(torch.ones(n_oscillators-1), 1) + 
                               torch.diag(torch.ones(n_oscillators-1), -1))
    
    # Find equilibrium position given forces
    def loss_fn(x, forces):
        return 0.5 * x.T @ K @ x - forces.T @ x
    
    return M, K, loss_fn
```

**Why it's good**:
- Physical interpretation
- Naturally coupled system
- Tests handling of banded structure

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Create `examples/correlated_gradients_compare.py` with all test problems
2. Implement configurable test harness for easy problem switching
3. Add visualization of gradient correlation and convergence paths

### Phase 2: Systematic Testing
1. Run each optimizer on each problem
2. Vary key parameters:
   - Dimension (64, 256, 896)
   - Condition number (10, 100, 1000, 10000)
   - Correlation strength (0.1, 0.5, 0.9, 0.99)
   - Rotation angle (0°, 22.5°, 45°, 67.5°)

### Phase 3: Analysis
1. Create heatmaps showing optimizer performance vs problem parameters
2. Identify regimes where each optimizer excels
3. Use insights to guide MORGN development

## Success Metrics

1. **Convergence Speed**: Steps to reach target loss
2. **Wall Clock Time**: Actual computation time
3. **Gradient Alignment**: Cosine similarity between update and negative gradient
4. **Eigenvalue Spectrum**: How well optimizer handles different curvatures
5. **Path Efficiency**: Ratio of path length to optimal straight-line distance

## Expected Outcomes

- **AdamW**: Good on axis-aligned problems, struggles with rotation/correlation
- **Muon**: Should excel on rotated/correlated problems due to Newton approximation
- **MORGN (stub)**: Currently just SGD+momentum, will underperform
- **MORGN (future)**: Design space for novel approach combining benefits

## Next Steps

1. ✅ Write this planning document
2. ⏳ Implement rotated paraboloid comparison
3. ⏳ Add Makefile targets for new tests
4. ⏳ Run systematic comparisons
5. ⏳ Create visualization suite
6. ⏳ Use results to inform MORGN algorithm design

## Key Insight

The fundamental challenge is that **gradient descent assumes the loss surface is isotropic** (same curvature in all directions), but real problems have:
- Different curvatures along different directions (condition number)
- Misalignment between gradient and optimal update direction (rotation)
- Correlation between parameters (off-diagonal Hessian terms)

Muon addresses this via orthogonalization (approximate Newton). MORGN could explore other approaches like:
- Adaptive preconditioning
- Learned coordinate transformations
- Momentum in transformed spaces
- Hybrid methods

This test suite will reveal which approach works best in which regime!