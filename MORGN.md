# MORGN Optimizer [by Gemini 2.5 Pro in gemini CLI]

MORGN (Momentum Orthogonalization via Recursive Gauss-Newton) is a
second-order optimization algorithm designed for training deep neural
networks. It is a quasi-Newton method that approximates the inverse
Hessian matrix using a recursive Gauss-Newton (RLS-style) update. This
allows the optimizer to take more effective, Newton-like steps, which
can lead to faster convergence and better performance on difficult
optimization problems.

## How it Works

The core of the MORGN optimizer is the left-preconditioned update for
2D parameters. For each matrix parameter `W` (with dimensions `m x
n`), the optimizer maintains an inverse Hessian approximation `P`
(with dimensions `m x m`) on the smaller/left dimension (transposing
if `m > n`). The update is then computed as follows:

1.  **Compute the preconditioned step:** `U = P @ G`, where `G` is the gradient of the loss with respect to `W`.
2.  **Update the parameter:** `W <- W - lr * U`, where `lr` is the learning rate.

The inverse Hessian approximation `P` is updated at each step using a rank-1 RLS/Sherman-Morrison formula. This update incorporates new gradient information and allows the optimizer to build up a more accurate approximation of the inverse Hessian over time.

## Key Features

*   **Second-Order Optimization:** MORGN is a second-order optimization algorithm that uses curvature information to take more effective steps.

*   **Quasi-Newton Method:** MORGN is a quasi-Newton method that
    approximates the inverse Hessian matrix instead of computing it
    directly. This makes it more computationally efficient than
    traditional Newton-based methods.

*   **Recursive Gauss-Newton Update:** MORGN uses a recursive
    Gauss-Newton (RLS-style) update to approximate the inverse Hessian
    matrix. This allows the optimizer to build up a more accurate
    approximation of the inverse Hessian over time.

*   **Vectorized Woodbury Update:** The implementation uses a
    vectorized RLS/Woodbury update to efficiently update `P` using
    multiple gradient directions at once. This batches what would
    otherwise be a sequence of rank‑1 Sherman–Morrison updates.

*   **Analytic Gauss-Newton (Sylvester):** There's an option to use an
    `analytic_gn` preconditioner for SPD matrix factorization tasks. It
    computes the Gauss–Newton step by solving the Sylvester equation
    `(W W^T + γI) ΔW + ΔW (W^T W + γI) = G` via eigendecompositions,
    which more closely matches true GN than chaining two inverses.

*   **Hyperparameter Complexity:** The MORGN optimizer has a large
    number of hyperparameters, and their interactions can be
    complex. This makes it difficult to tune.

## Hyperparameters

The MORGN optimizer has a number of hyperparameters that can be used to control its behavior. These include:

*   `lr`: The learning rate.
*   `wd`: The weight decay.
*   `lambda_`: The forgetting factor for the RLS update.
*   `eps`: The initial inverse scale for the RLS update.
*   `directions`: The number of gradient columns to assimilate per step.
*   `step_clamp`: A clamp on the Frobenius norm of the update.
*   `two_sided`: Whether to use a two-sided preconditioner.
*   `right_lambda`: The forgetting factor for the right-sided RLS update.
*   `right_directions`: The number of gradient columns to assimilate per step for the right-sided RLS update.
*   `precond_warmup_steps`: The number of steps to warm up the preconditioner.
*   `precond_warmup_exp`: The exponent for the preconditioner warmup.
*   `momentum`: The momentum for the gradient.
*   `nesterov`: Whether to use Nesterov momentum.
*   `analytic_gn`: Whether to use the analytic Gauss-Newton preconditioner
    (Sylvester-based; recommended for SPD factorization tasks).
*   `analytic_gamma`: The damping `γ` for the analytic Gauss-Newton preconditioner.
    Smaller values are more Newton-like but less stable.

## Notes and Caveats

* `P` and (optionally) `Q` are kept symmetric and updated with a
  forgetting factor `λ` using the standard RLS recursion
  `P ← (1/λ)(P − P G (λI + G^T P G)^{-1} G^T P)`. For numerical
  stability, small ridge terms and Cholesky-based solves are used where
  available.
* The left-only preconditioner (`P`) approximates curvature on the
  smaller matrix dimension. When strong right-side curvature is present
  (e.g., rotated SPD factorization), enabling `two_sided` can provide a
  closer quasi-Newton step.
* Early in training, blending the raw gradient with the preconditioned
  update (`precond_warmup_steps`, `precond_warmup_exp`) and capping the
  update norm (`step_clamp`) improve robustness.
