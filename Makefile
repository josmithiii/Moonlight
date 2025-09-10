# Makefile for Moonlight project
# Run 'make help' to see all available targets

SHELL := /bin/bash
PYTHON := /usr/bin/python3
# Virtual environment activation
VENV_ACTIVATE := source .venv/bin/activate &&
TRAIN_SCRIPT := examples/toy_train.py

# Default parameters
MODEL := qwen
DATASET := openwebtext-100k
HIDDEN_SIZE := 896
LR := 1e-3
WD := 0.1

# Log directory
LOGS_DIR := logs

.PHONY: help setup setup-runpod clean train-muon train-adamw train-both compare-optimizers quick-compare quick-compare-plot quick-compare-factorization qcfp-sym-id qcfp-sym-rot simple-demo test-sizes all viz-muon viz-adamw viz-compare

h help:
	@echo "Moonlight Training Makefile"
	@echo ""
	@echo "Setup targets:"
	@echo "  setup          - Create uv environment and install dependencies"
	@echo "  clean          - Remove virtual environment and generated files"
	@echo ""
	@echo "Training targets (documented examples):"
	@echo "  train-muon     - Train with Muon optimizer (hidden_size=896, lr=1e-3)"
	@echo "  train-adamw    - Train with AdamW optimizer (hidden_size=896, lr=1e-3)"
	@echo "  train-both     - Train with both optimizers sequentially"
	@echo ""
	@echo "Comparison targets:"
	@echo "  compare-optimizers - Train all three optimizers with same config for comparison"
	@echo "  quick-compare  - Quick efficiency comparison (AdamW vs Muon vs MORGN)"
	@echo "  quick-compare-factorization - Compare on matrix factorization problem"
	@echo "  qcfp-sym-id   - SPD factorization with identity eigenvectors (AdamW favored)"
	@echo "  qcfp-sym-rot  - SPD factorization with rotated eigenvectors (Muon favored)"
	@echo "  qcfp-sym-rot2 - SPD factorization (rotated) with MORGN two-sided preconditioner"
	@echo "  simple-demo    - Simple three-way demo on toy problem"
	@echo "  test-sizes     - Test different model sizes (512, 896, 1024)"
	@echo ""
	@echo "Visualization targets:"
	@echo "  viz-muon       - Show Muon optimizer architecture diagrams"
	@echo "  viz-adamw      - Show AdamW optimizer architecture diagrams"
	@echo "  viz-compare    - Show comparison of both optimizers"
	@echo ""
	@echo "Convenience targets:"
	@echo "  all            - Run both documented examples"
	@echo ""
	@echo "Parameters (can override with make VAR=value):"
	@echo "  MODEL=$(MODEL), DATASET=$(DATASET), HIDDEN_SIZE=$(HIDDEN_SIZE)"
	@echo "  LR=$(LR), WD=$(WD)"
	@echo ""
	@echo "Other targets:"
	@grep -E '^[.a-zA-Z0-9_ -]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}' || true

#       @grep -E '^[.a-zA-Z0-9_ -]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' | less -R

setup-runpod: ## Add ~/.local/bin to path for RunPod Ubuntu systems:
	@echo "Setting up for RunPod ..."
	echo 'export PATH="$$HOME/.local/bin:$$PATH"' >> ~/.bashrc
	@echo "Configuring git..."
	git config --global user.name "Julius O. Smith III"
	git config --global user.email "julius.smith@gmail.com"
	git config --global credential.helper store
	@echo "Now say 'source ~/.bashrc' and run 'git push' once to store credentials"

setup:
	@echo "Setting up environment..."
	./setup.sh

$(LOGS_DIR):
	mkdir -p $(LOGS_DIR)

# Virtual environment setup
.venv/bin/activate:
	@echo "Setting up virtual environment..."
	./setup.sh

# Documented examples from CLAUDE.md and README.md
train-muon: $(LOGS_DIR) .venv/bin/activate
	@echo "Training with Muon optimizer (documented example)..."
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) \
		--optimizer muon \
		--dataset $(DATASET) \
		--hidden_size $(HIDDEN_SIZE) \
		--lr $(LR) \
		--wd $(WD)
	@echo "Training complete! Saving results..."
	@git add -f logs/ && git commit -m "Muon training results - $(MODEL) h$(HIDDEN_SIZE) lr$(LR) - $$(date)" && git push || echo "Failed to save results to git"

train-adamw: $(LOGS_DIR) .venv/bin/activate
	@echo "Training with AdamW optimizer (documented example)..."
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) \
		--optimizer adamw \
		--dataset $(DATASET) \
		--hidden_size $(HIDDEN_SIZE) \
		--lr $(LR) \
		--wd $(WD)
	@echo "Training complete! Saving results..."
	@git add -f logs/ && git commit -m "AdamW training results - $(MODEL) h$(HIDDEN_SIZE) lr$(LR) - $$(date)" && git push || echo "Failed to save results to git"

# Auto-install dependencies 
install-deps:
	@echo "Checking and installing dependencies..."
	@pip install --index-url https://download.pytorch.org/whl/cu124 -r requirements.txt >/dev/null 2>&1 && echo "Dependencies ready" || echo "Dependencies installed"

fix-rtx5090: ## Fix RTX 5090 compatibility by upgrading PyTorch to CUDA 12.4 version
	@echo "Fixing RTX 5090 compatibility..."
	pip uninstall torch torchvision -y
	pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124

# Run both documented examples
train-both: install-deps train-muon train-adamw

all: install-deps train-both

# Comparison experiments
co compare-optimizers: $(LOGS_DIR) .venv/bin/activate
	@echo "Comparing AdamW vs Muon vs MORGN with identical configurations..."
	@echo "Training with AdamW..."
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) --optimizer adamw --dataset $(DATASET) \
		--hidden_size $(HIDDEN_SIZE) --lr $(LR) --wd $(WD)
	@echo "Training with Muon..."
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) --optimizer muon --dataset $(DATASET) \
		--hidden_size $(HIDDEN_SIZE) --lr $(LR) --wd $(WD)
	@echo "Training with MORGN..."
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) --optimizer morgn --dataset $(DATASET) \
		--hidden_size $(HIDDEN_SIZE) --lr $(LR) --wd $(WD)
	@echo "Check $(LOGS_DIR)/ for training logs to compare performance"

qc quick-compare: .venv/bin/activate ## Run quick efficiency comparison between AdamW, Muon, and MORGN
	@echo "Running quick AdamW vs Muon vs MORGN comparison..."
	source .venv/bin/activate && python3 examples/quick_compare.py --steps 100 --hidden-size 128

qcp quick-compare-plot: .venv/bin/activate ## Run quick comparison with convergence plot
	@echo "Running quick comparison with plot generation..."
	source .venv/bin/activate && python3 examples/quick_compare.py --steps 200 --hidden-size 256 --plot

# Even smaller: python3 examples/quick_compare.py --steps 50 --hidden-size 128 --device mps --plot
# Force CPU: CUDA_VISIBLE_DEVICES="" python3 examples/quick_compare.py --steps 10 --hidden-size 64 --device cpu

qcfp quick-compare-factorization: .venv/bin/activate ## Run quick comparison on matrix factorization problem, default condition number 1000, 896x896 matrix
	@echo "Running Optimizer Comparisons on matrix factorization..."
	source .venv/bin/activate && python3 examples/matrix_factorization_compare.py --matrix-size $(HIDDEN_SIZE) --condition-number 1000.0 --steps 200 --plot; \
	open matrix_factorization_comparison.png

qcfp1000: ## Run quick comparison on matrix factorization problem with condition number 1000
	@echo "Running Optimizer Comparisons on matrix factorization with condition number 1000"
	python3 examples/matrix_factorization_compare.py 

# SPD factorization showcase targets
id qcfp-sym-id: .venv/bin/activate ## SPD factorization with identity eigenvectors (coordinate-aligned). AdamW should excel.
	@echo "Running SPD factorization with identity eigenvectors (AdamW favored)..."
	source .venv/bin/activate && python3 examples/matrix_factorization_compare.py \
		--symmetric --evec-mode identity --matrix-size 512 --condition-number 5000 \
		--steps 300 --lr 1e-2 \
		--muon-lr 3e-3 --muon-ns-steps 8 \
		--muon-lr-warmdown-at 0.7 --muon-lr-decay-factor 0.1 \
		--morgn-lr 0.01 --morgn-lambda 0.999 --morgn-eps 1e-4 --morgn-step-clamp 1.0 --morgn-directions 8 \
		--clip-grad-norm 1.0 --plot --out matrix_factorization_sym_id_n512_k5000_s300.png $(ARGS)
	@open matrix_factorization_sym_id_n512_k5000_s300.png || true

rot qcfp-sym-rot: .venv/bin/activate ## SPD factorization with random eigenvectors (rotated). Muon should excel.
	@echo "Running SPD factorization with rotated eigenvectors (Muon favored)..."
	source .venv/bin/activate && python3 examples/matrix_factorization_compare.py \
		--symmetric --evec-mode random --matrix-size 512 --condition-number 5000 \
		--steps 500 --lr 1e-2 --muon-lr 3e-3 --muon-ns-steps 8 \
		--muon-lr-warmdown-at 0.7 --muon-lr-decay-factor 0.1 \
		--morgn-lr 1 --morgn-lambda 0.997 --morgn-eps 1e-2 --morgn-step-clamp 0.2 --morgn-directions 8 \
		--clip-grad-norm 1.0 --plot --out matrix_factorization_sym_random_n512_k5000_s450.png $(ARGS)
	@open matrix_factorization_sym_random_n512_k5000_s450.png || true

rot2 qcfp-sym-rot2: .venv/bin/activate ## SPD factorization (rotated). MORGN two-sided preconditioner enabled.
	@echo "Running SPD factorization with rotated eigenvectors (MORGN two-sided)..."
	source .venv/bin/activate && python3 examples/matrix_factorization_compare.py \
		--symmetric --evec-mode random --matrix-size 512 --condition-number 5000 \
		--steps 500 --lr 1e-2 --muon-lr 3e-3 --muon-ns-steps 8 \
		--muon-lr-warmdown-at 0.7 --muon-lr-decay-factor 0.1 \
		--morgn-two-sided --morgn-lr 6e-3 \
		--morgn-lambda 0.997 --morgn-right-lambda 0.997 \
		--morgn-eps 1e-2 --morgn-precond-warmup-steps 100 --morgn-precond-warmup-exp 0.7 \
		--morgn-directions 16 --morgn-right-directions 16 \
		--morgn-step-clamp 0.5 --clip-grad-norm 1.0 --plot \
		--out matrix_factorization_sym_random_n512_k5000_s450_morgn2s.png $(ARGS)
	@open matrix_factorization_sym_random_n512_k5000_s450_morgn2s.png || true

rp rotated-paraboloid: .venv/bin/activate ## Run optimizer comparison on rotated paraboloid with correlated gradients
	@echo "Running optimizer comparison on rotated paraboloid problem..."
	source .venv/bin/activate && python3 examples/rotated_paraboloid_compare.py --dim 64 --condition-number 100 --rotation-angle 45 --steps 200 --plot
	@[ -f rotated_paraboloid_comparison.png ] && open rotated_paraboloid_comparison.png || true

rbp rotated-paraboloid-block: ## Run optimizer comparison on rotated paraboloid with correlated gradients, old block rotation method
	python examples/rotated_paraboloid_compare.py --rotation-mode block --dim 64 --condition-number 100 --rotation-angle 45 --steps 200 --plot
	@[ -f rotated_paraboloid_comparison.png ] && open rotated_paraboloid_comparison.png || true

sd simple-demo: .venv/bin/activate ## Run simple three-way optimizer demo on toy problem (30 seconds)
	@echo "Running simple AdamW vs Muon vs MORGN demonstration..."
	source .venv/bin/activate && python3 examples/simple_demo.py

# Test different model sizes
test-sizes: $(LOGS_DIR) .venv/bin/activate
	@echo "Testing different model sizes with Muon..."
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) --optimizer muon --dataset $(DATASET) \
		--hidden_size 512 --lr $(LR) --wd $(WD)
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) --optimizer muon --dataset $(DATASET) \
		--hidden_size 896 --lr $(LR) --wd $(WD)
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) --optimizer muon --dataset $(DATASET) \
		--hidden_size 1024 --lr $(LR) --wd $(WD)

# Custom training with parameters
train-custom: $(LOGS_DIR) .venv/bin/activate
	@if [ -z "$(OPTIMIZER)" ]; then echo "Error: OPTIMIZER not specified. Use: make train-custom OPTIMIZER=muon|adamw [MODEL=qwen] [HIDDEN_SIZE=896] [LR=1e-3]"; exit 1; fi
	@echo "Training with custom parameters..."
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) \
		--optimizer $(OPTIMIZER) \
		--dataset $(DATASET) \
		--hidden_size $(HIDDEN_SIZE) \
		--lr $(LR) \
		--wd $(WD)

# Quick smoke test
smoke-test: $(LOGS_DIR) .venv/bin/activate
	@echo "Running quick smoke test with small model..."
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) --optimizer muon --dataset $(DATASET) \
		--hidden_size 256 --lr $(LR) --wd $(WD)

# Development helpers
check-env: .venv/bin/activate
	@echo "Checking environment..."
	$(VENV_ACTIVATE) $(PYTHON) --version
	$(VENV_ACTIVATE) $(PYTHON) -c "import torch; import transformers; print('Environment OK')"

list-logs:
	@echo "Available training logs:"
	@ls -la $(LOGS_DIR)/ 2>/dev/null || echo "No logs directory found"

clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -rf $(LOGS_DIR)
	rm -rf __pycache__
	rm -rf examples/__pycache__
	@echo "Cleaned virtual environment, logs, cached datasets, and Python cache files"

dclean dist-clean: clean ## make clean plus deleting any downloaded and pre-processed dataset files
	rm -f *.bin

# VISUALIZATION TARGETS "viz*"


vm viz-muon: .venv/bin/activate ## Generate diagrams showing Muon optimizer architecture
	@echo "Generating Muon optimizer diagrams..."
	source .venv/bin/activate && python3 viz/enhanced_model_diagrams.py --optimizer muon

va viz-adamw: .venv/bin/activate ## Generate diagrams showing AdamW optimizer architecture  
	@echo "Generating AdamW optimizer diagrams..."
	source .venv/bin/activate && python3 viz/enhanced_model_diagrams.py --optimizer adamw

vc viz-compare: .venv/bin/activate ## Generate comparison diagrams for Muon vs AdamW
	@echo "Generating optimizer comparison diagrams..."
	source .venv/bin/activate && python3 viz/enhanced_model_diagrams.py --compare-optimizers

# Environment check before running training
.PHONY: check-env list-logs smoke-test train-custom quick-compare quick-compare-plot simple-demo viz-muon viz-adamw viz-compare
