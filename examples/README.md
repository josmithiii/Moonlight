Examples Overview

This folder contains runnable training and comparison scripts used to evaluate AdamW, Muon, and MORGN. Most examples have Makefile shortcuts and can also be run directly.

Before running examples:
- Create the environment: `make setup`
- On RTX 5090/CUDA 12.4: `make fix-rtx5090`
- Force CPU if needed: `CUDA_VISIBLE_DEVICES=""`
- Many scripts auto-detect `cuda`/`mps`/`cpu` via a `--device` flag.

Quick Compare (tiny Qwen2)
- Script: `examples/quick_compare.py`
- Goal: fast AdamW vs Muon vs MORGN comparison on a tiny Qwen2 model.
- Make targets:
  - `make quick-compare` (console)
  - `make qcp` (console + plot)
- Direct run:
  - `uv run python examples/quick_compare.py --steps 200 --hidden-size 256 --plot`
- Notes: Produces `optimizer_comparison.png` when `--plot` is used.

Matrix Factorization (SPD)
- Script: `examples/matrix_factorization_compare.py`
- Goal: factorize a symmetric positive definite matrix; stresses curvature.
- Key flags:
  - `--symmetric` to use SPD factorization (W W^T ≈ T)
  - `--evec-mode {random,identity}` to choose eigenvectors
  - MORGN: `--morgn-two-sided`, `--morgn-analytic-gn`, `--morgn-analytic-gamma`
  - Convergence metric: `--convergence-mode {percent_to_final,fraction_initial,target_loss}` with `--convergence-fraction` or `--convergence-target-loss`
- Make targets:
  - Identity evecs (AdamW favored): `make id`
  - Rotated evecs (Muon favored): `make rot`
  - Rotated + MORGN two-sided: `make rot2`
  - Rotated + MORGN analytic GN (Sylvester): `make rot-gn`
  - Rotated + MORGN two-sided + analytic GN: `make rot2-gn`
  - Fair high-condition comparison (fraction_initial metric): `make fair`
- Direct run examples:
  - Identity SPD: `uv run python examples/matrix_factorization_compare.py --symmetric --evec-mode identity --matrix-size 512 --condition-number 5000 --steps 300 --lr 1e-2 --plot`
  - Rotated SPD + two-sided + analytic GN: `uv run python examples/matrix_factorization_compare.py --symmetric --evec-mode random --matrix-size 512 --condition-number 20000 --steps 400 --lr 1e-2 --morgn-two-sided --morgn-analytic-gn --morgn-lr 0.02 --morgn-lambda 0.999 --morgn-right-lambda 0.999 --morgn-step-clamp 0.8 --convergence-mode fraction_initial --convergence-fraction 1e-3 --plot`
- Output: saves a comparison plot (filename depends on flags or `--out`).

Rotated Paraboloid (Correlated Gradients)
- Script: `examples/rotated_paraboloid_compare.py`
- Goal: quadratic testbed with controllable condition and rotation; shows handling of correlated gradients.
- Make targets:
  - Random dense rotation: `make rp`
  - Block 2x2 rotation: `make rbp`
- Direct run: `uv run python examples/rotated_paraboloid_compare.py --dim 64 --condition-number 100 --rotation-angle 45 --steps 200 --plot`
- Output: `rotated_paraboloid_comparison.png` when `--plot` is used.

Simple Demo (Tiny MLP Regression)
- Script: `examples/simple_demo.py`
- Goal: quick three-way demo on a tiny 2-layer network.
- Make target: `make sd`
- Direct run: `uv run python examples/simple_demo.py`

Toy Training (Qwen2 on OpenWebText-100k)
- Script: `examples/toy_train.py`
- Goal: minimal training loop for Qwen2 on a small dataset; supports AdamW, Muon, MORGN.
- Make targets:
  - `make train-muon`
  - `make train-adamw`
  - `make compare-optimizers` (runs all three)
- Direct run:
  - `uv run python examples/toy_train.py --model qwen --optimizer muon --hidden_size 896`
  - `uv run python examples/toy_train.py --model qwen --optimizer morgn --hidden_size 896`
- Notes:
  - Uses `datasets` to load `Elriggs/openwebtext-100k` and caches tokens to `openwebtext-100k.bin`.
  - Device auto-detected; use `CUDA_VISIBLE_DEVICES=""` to force CPU.

Experimental/Prototypes
- These scripts explore alternate MORGN variations and are not wired into the main Make targets. Run directly if needed.
- `examples/matrix_factorization_compare_improved.py` — adds an experimental `MORGNImproved` baseline and extra knobs (trust region, adaptive ridge).
- `examples/morgn_improved.py` — prototype optimizer with adaptive ridge, trust region, and curvature-aware LR.
- `examples/toy_train_improved.py` / `examples/toy_train_jos.py` — historical variants of the toy training loop.

Tips
- Plots: Most scripts accept `--plot` and save a PNG next to the repo root.
- Determinism: All scripts fix seeds (`--seed`) for reproducibility.
- Backend notes: On Apple Silicon, scripts auto-select `mps` when available. On CUDA, bfloat16 is used in some paths (e.g., Muon).
- For very high condition numbers, consider enabling MORGN’s two-sided preconditioning and analytic GN with moderate damping (`--morgn-analytic-gamma`).
