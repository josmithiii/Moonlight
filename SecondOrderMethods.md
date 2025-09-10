# Second-Order Optimization Methods

## Overview

Second-order optimization methods utilize both the gradient (first derivatives) and the Hessian matrix (second derivatives) of the objective function to guide the optimization process. This approach allows for more informed and potentially faster convergence compared to first-order methods, which rely solely on gradient information.

The key advantage is that second-order methods can account for the **curvature** of the loss surface, enabling them to:
- Navigate ill-conditioned problems more effectively
- Avoid zigzagging in narrow valleys
- Achieve faster convergence near optima
- Handle correlated gradients better

## Classical Second-Order Methods

### 1. Newton's Method

**Core Idea**: Use the full Hessian matrix to determine both step direction and size.

**Update Rule**:
```
x_{k+1} = x_k - H^{-1} * ∇f(x_k)
```

Where `H` is the Hessian matrix of second derivatives.

**Advantages**:
- Quadratic convergence near the optimum
- Optimal step size and direction in quadratic problems
- Accounts for all curvature information

**Disadvantages**:
- O(n³) computational cost to compute and invert Hessian
- O(n²) memory requirement
- Requires Hessian to be positive definite
- Can be unstable far from optimum

### 2. Quasi-Newton Methods

These methods approximate the Hessian to reduce computational cost while maintaining superlinear convergence.

#### BFGS (Broyden-Fletcher-Goldfarb-Shanno)

**Core Idea**: Build up Hessian approximation iteratively using gradient differences.

**Update Formula**:
```
B_{k+1} = B_k + (y_k y_k^T)/(y_k^T s_k) - (B_k s_k s_k^T B_k)/(s_k^T B_k s_k)
```

Where:
- `s_k = x_{k+1} - x_k` (step taken)
- `y_k = ∇f(x_{k+1}) - ∇f(x_k)` (gradient difference)

**Advantages**:
- Superlinear convergence
- No need to compute actual Hessian
- Maintains positive definiteness

**Disadvantages**:
- O(n²) memory for storing approximation
- O(n²) computation per step

#### L-BFGS (Limited-memory BFGS)

**Core Idea**: Store only the most recent gradient and step differences instead of full matrix.

**Key Features**:
- Stores only `m` recent vector pairs (typically m = 3-20)
- Implicitly represents Hessian approximation
- Uses two-loop recursion to compute search direction

**Advantages**:
- O(n) memory requirement
- Suitable for large-scale problems
- Available in PyTorch as `torch.optim.LBFGS`

**Disadvantages**:
- May require line search (needs closure function)
- Can be sensitive to numerical precision

### 3. Trust Region Methods

**Core Idea**: Define a "trust region" where the quadratic model is reliable, then solve a constrained optimization problem within this region.

**Subproblem**:
```
min_p  ∇f^T p + (1/2) p^T H p
subject to  ||p|| ≤ Δ
```

**Examples**:
- Levenberg-Marquardt: `(H + λI)^{-1} ∇f`
- Dogleg methods: Combine Cauchy point and Newton step

**Advantages**:
- Robust convergence guarantees
- Handles indefinite Hessians
- Adaptive step size control

## Modern Adaptive Second-Order Methods

### 4. Natural Gradient Methods

**Core Idea**: Use the Fisher Information Matrix (FIM) as a preconditioner instead of the Hessian.

**Update Rule**:
```
θ_{k+1} = θ_k - α F^{-1} ∇L
```

Where `F` is the Fisher Information Matrix.

**Applications**:
- Reinforcement learning (TRPO, PPO)
- Variational inference
- Neural network training

**Examples**:
- **K-FAC**: Kronecker-factored approximation of Fisher matrix
- **TRPO**: Trust Region Policy Optimization

### 5. AdaHessian

**Core Idea**: Adaptively estimate diagonal Hessian elements using Hutchinson's trace estimator.

**Key Features**:
- Uses random vectors to estimate Hessian diagonal
- Maintains exponential moving averages like Adam
- Practical for deep learning

**Update Rule**:
```
v_t = β₂ v_{t-1} + (1-β₂) * diag(H_t)
θ_{t+1} = θ_t - α * g_t / (√v_t + ε)
```

### 6. Shampoo

**Core Idea**: Maintain separate preconditioners for each parameter dimension using matrix square roots.

**Key Features**:
- Full-matrix preconditioning for each layer
- Uses matrix square root as preconditioner
- More sophisticated than diagonal methods

## Methods in the Moonlight Codebase

### MORGN (Momentum Orthogonalization via Recursive Gauss-Newton)

**Core Concept**: Implements a left-preconditioned update using Recursive Least Squares (RLS) approach.

**Key Features**:
- Maintains inverse Hessian approximation `P` (m×m) for the smaller dimension
- Uses Sherman-Morrison formula for efficient rank-1 updates
- Processes multiple gradient directions per step

**Update Process**:
1. **Left Newton Step**: `U = P @ G` where `G` is the gradient matrix
2. **Parameter Update**: `W ← W - lr * U`
3. **Preconditioner Update**: Use Sherman-Morrison to update `P`

**Sherman-Morrison Formula**:
```
P_{k+1} = P_k - (P_k g g^T P_k) / (1 + g^T P_k g)
```

**Computational Complexity**:
- Memory: O(m²) where m ≤ n is the smaller dimension
- Computation: O(m² * k) where k is number of gradient directions per step

**Advantages**:
- Captures up to k gradient directions per iteration
- More efficient than full Newton (works on smaller dimension)
- Maintains positive definiteness through initialization

### Muon's Newton-like Properties

While Muon is primarily an orthogonalization-based method, it has second-order characteristics:

**Newton Approximation**: The orthogonalization process can be viewed as normalizing curvature, similar to how Newton's method uses the Hessian to account for curvature.

**When Muon Excels** (from `CORRELATED_GRADIENTS_PLANS.md`):
1. **Highly correlated gradients** - Standard SGD would zigzag
2. **Misaligned curvature** - Principal directions don't align with coordinates
3. **Ill-conditioned problems** - Large condition numbers

## When Second-Order Methods Excel

### Problem Characteristics

1. **Correlated Gradients**: When gradient components are highly correlated, first-order methods waste steps zigzagging.

2. **Ill-Conditioned Problems**: High condition numbers (ratio of largest to smallest eigenvalue) make first-order methods slow.

3. **Narrow Valleys**: Like the Rosenbrock function, where the optimum lies in a narrow curved valley.

4. **Rotated Quadratics**: When principal curvature directions are misaligned with coordinate axes.

### Test Problems for Second-Order Methods

From the codebase's `CORRELATED_GRADIENTS_PLANS.md`:

#### Rotated Paraboloid
```python
def create_rotated_quadratic_problem(dim=896, condition_number=1000.0, rotation_angle=45):
    # Diagonal Hessian with specified condition number
    eigenvals = torch.logspace(0, np.log10(condition_number), dim)
    H_diag = torch.diag(eigenvals)
    
    # Rotate to misalign gradients with curvature
    R = create_rotation_matrix(dim, rotation_angle)
    H = R @ H_diag @ R.T
    
    def loss_fn(x, target=torch.zeros(dim)):
        return 0.5 * (x - target).T @ H @ (x - target)
    
    return H, loss_fn
```

This problem is ideal because:
- **Direct test** of Newton approximation capability
- **Controllable difficulty** via rotation angle and condition number
- **Clean interpretation**: measures ability to follow curvature

## Computational Tradeoffs

| Method | Memory | Computation/Step | Convergence Rate | Notes |
|--------|---------|------------------|------------------|-------|
| SGD | O(n) | O(n) | Linear | Baseline first-order |
| Adam | O(n) | O(n) | Linear | Adaptive first-order |
| BFGS | O(n²) | O(n²) | Superlinear | Full quasi-Newton |
| L-BFGS | O(mn) | O(mn) | Superlinear | m ≈ 3-20 |
| Newton | O(n²) | O(n³) | Quadratic | Full second-order |
| MORGN | O(m²) | O(m²k) | ? | Left-preconditioned, k directions |
| AdaHessian | O(n) | O(n) | ? | Diagonal Hessian approximation |

## Closure Functions in PyTorch

Many second-order methods require **multiple evaluations** of the objective function per step, which is why PyTorch optimizers accept a `closure` parameter:

```python
def step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()
    # ... perform optimization step
    return loss
```

**When closures are used**:
- **L-BFGS**: For line search to find optimal step size
- **Some trust region methods**: To evaluate trial steps
- **Research optimizers**: That need multiple forward passes

**Current usage in Moonlight**: The implemented optimizers (Muon, MORGN) include closure support for API compliance, but the training loops don't currently use closures.

## Practical Considerations

### Choosing a Second-Order Method

1. **Problem Size**:
   - Small problems (n < 1000): Full Newton or BFGS feasible
   - Large problems: L-BFGS, AdaHessian, or MORGN-style approaches

2. **Memory Constraints**:
   - Limited memory: L-BFGS, diagonal methods
   - Abundant memory: BFGS, full Newton

3. **Function Evaluations**:
   - Expensive function evaluations: Methods with good per-iteration progress
   - Cheap evaluations: Can afford more iterations with simpler methods

4. **Gradient Availability**:
   - Exact gradients: Classical methods work well
   - Noisy gradients: May need adaptive or robust variants

### Implementation Tips

1. **Initialization**: Second-order methods often need good initialization of the Hessian approximation (e.g., `P₀ = εI` in MORGN).

2. **Numerical Stability**: Watch for singular matrices, use regularization when needed.

3. **Scaling**: Consider the scale of different parameters - may need preconditioning.

4. **Convergence Criteria**: Second-order methods can converge very quickly near the optimum.

## Future Directions

### Hybrid Approaches
- Combine first-order and second-order information (like FOSI)
- Switch between methods based on problem characteristics
- Use second-order methods for preconditioning first-order updates

### Distributed Second-Order Methods
- Federated learning with second-order updates
- Communication-efficient Hessian approximations
- Parallel matrix operations for large-scale problems

### Neural Network Specific
- Block-diagonal Hessian approximations
- Layer-wise second-order updates
- Integration with modern architectures (Transformers, etc.)

## References and Further Reading

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*. Springer.
- Martens, J. (2020). *New Insights and Perspectives on the Natural Gradient Method*. JMLR.
- Yao, Z. et al. (2021). *ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning*. AAAI.
- The MORGN implementation in this codebase draws from Recursive Least Squares and Gauss-Newton methods.
