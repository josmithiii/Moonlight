# Repository Guidelines

## Project Structure & Module Organization
- `examples/` — runnable training and comparison scripts (primary code).
- `viz/` — diagram/plot scripts; outputs PNGs in repo root.
- `logs/` — training logs (auto-created). Avoid committing large logs unless requested.
- `Makefile` — common tasks and documented examples.
- `setup.sh` — creates a local `uv` virtual env and VS Code/Cursor configs.
- `pyproject.toml`, `requirements.txt` — dependencies. Python ≥ 3.10.

## Build, Test, and Development Commands
- `make setup` — initialize env using `uv` and create `.venv`.
- `make help` — list available targets.
- Quick runs: `make quick-compare`, `make qcp` (with plot), `make sd`.
- Training demos: `make train-muon`, `make train-adamw`, `make compare-optimizers`.
- Direct run: `uv run python examples/toy_train.py --model qwen --optimizer muon --hidden_size 896`.
- GPU notes: use `CUDA_VISIBLE_DEVICES=""` to force CPU; `make fix-rtx5090` for CUDA 12.4 wheels.

## Coding Style & Naming Conventions
- Python style: PEP 8, 4-space indents, one import per line.
- Naming: `snake_case` for functions/vars, `CamelCase` for classes, `UPPER_SNAKE` for constants.
- Prefer type hints and docstrings for public functions/classes.
- Logging: use `loguru` where present; avoid print in library code.
- Formatting: if you use tools, prefer `black` and `isort` locally; no enforced config yet.

## Testing Guidelines
- No formal unit tests yet. Validate with small, reproducible runs:
  - `make quick-compare` or `uv run python examples/quick_compare.py --steps 50 --hidden-size 128`.
- If adding tests, place in `tests/` as `test_*.py`, prefer `pytest` and deterministic seeds.
- Keep tests fast (tiny models, ≤100 steps) and avoid external downloads.

## Commit & Pull Request Guidelines
- Commits: imperative mood and concise (e.g., "Add MORGN optimizer stub", "Fix launch config").
- PRs: include a clear description, reproduction commands, and artifacts when relevant (plots like `optimizer_comparison.png`, key log snippets).
- Link related issues; update `README.md`/`Makefile` when changing flags or user workflows.
- Avoid committing large binaries; prefer references to reproducible commands.

## Security & Configuration Tips
- Secrets: do not store credentials in code or logs.
- Environments: prefer `make setup` with `uv`; `.venv/` is local-only.
- Large artifacts: dataset caches or `.bin` token files shouldn’t be added unless documented.
