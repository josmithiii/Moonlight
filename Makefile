# Makefile for Moonlight project
# Run 'make help' to see all available targets

SHELL := /bin/bash
PYTHON := /usr/bin/python3
# Use virtual env if it exists, otherwise use system python directly
VENV_ACTIVATE := test -f .venv/bin/activate && source .venv/bin/activate ||
TRAIN_SCRIPT := examples/toy_train.py

# Default parameters
MODEL := qwen
DATASET := openwebtext-100k
HIDDEN_SIZE := 896
LR := 1e-3
WD := 0.1

# Log directory
LOGS_DIR := logs

.PHONY: help setup setup-runpod clean train-muon train-adamw train-both compare-optimizers quick-compare quick-compare-plot quick-compare-factorization simple-demo test-sizes all viz-muon viz-adamw viz-compare

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

# Documented examples from CLAUDE.md and README.md
train-muon: $(LOGS_DIR)
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

train-adamw: $(LOGS_DIR)
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
co compare-optimizers: $(LOGS_DIR)
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

qc quick-compare: ## Run quick efficiency comparison between AdamW, Muon, and MORGN
	@echo "Running quick AdamW vs Muon vs MORGN comparison..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python3 examples/quick_compare.py --steps 100 --hidden-size 128; \
	else \
		python3 examples/quick_compare.py --steps 100 --hidden-size 128; \
	fi

qcp quick-compare-plot: ## Run quick comparison with convergence plot
	@echo "Running quick comparison with plot generation..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python3 examples/quick_compare.py --steps 200 --hidden-size 256 --plot; \
	else \
		python3 examples/quick_compare.py --steps 200 --hidden-size 256 --plot; \
	fi

# Even smaller: python3 examples/quick_compare.py --steps 50 --hidden-size 128 --device mps --plot
# Force CPU: CUDA_VISIBLE_DEVICES="" python3 examples/quick_compare.py --steps 10 --hidden-size 64 --device cpu

qcfp quick-compare-factorization: ## Run quick comparison on matrix factorization problem, default condition number 100
	@echo "Running Muon vs AdamW on matrix factorization..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python3 examples/matrix_factorization_compare.py --matrix-size 64 --steps 200 --plot; \
	else \
		python3 examples/matrix_factorization_compare.py --matrix-size 64 --steps 200 --plot; \
	fi

qcfp1000: ## Run quick comparison on matrix factorization problem with condition number 1000
	@echo "Running AdamW vs Muon vs MORGN on matrix factorization with condition number 1000"
	python3 examples/matrix_factorization_compare.py --condition-number 1000.0

sd simple-demo: ## Run simple three-way optimizer demo on toy problem (30 seconds)
	@echo "Running simple AdamW vs Muon vs MORGN demonstration..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python3 examples/simple_demo.py; \
	else \
		python3 examples/simple_demo.py; \
	fi

# Test different model sizes
test-sizes: $(LOGS_DIR)
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
train-custom: $(LOGS_DIR)
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
smoke-test: $(LOGS_DIR)
	@echo "Running quick smoke test with small model..."
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) --optimizer muon --dataset $(DATASET) \
		--hidden_size 256 --lr $(LR) --wd $(WD)

# Development helpers
check-env:
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


vm viz-muon: ## Generate diagrams showing Muon optimizer architecture
	@echo "Generating Muon optimizer diagrams..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python3 viz/enhanced_model_diagrams.py --optimizer muon; \
	else \
		python3 viz/enhanced_model_diagrams.py --optimizer muon; \
	fi

va viz-adamw: ## Generate diagrams showing AdamW optimizer architecture  
	@echo "Generating AdamW optimizer diagrams..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python3 viz/enhanced_model_diagrams.py --optimizer adamw; \
	else \
		python3 viz/enhanced_model_diagrams.py --optimizer adamw; \
	fi

vc viz-compare: ## Generate comparison diagrams for Muon vs AdamW
	@echo "Generating optimizer comparison diagrams..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python3 viz/enhanced_model_diagrams.py --compare-optimizers; \
	else \
		python3 viz/enhanced_model_diagrams.py --compare-optimizers; \
	fi

# Environment check before running training
.PHONY: check-env list-logs smoke-test train-custom quick-compare quick-compare-plot simple-demo viz-muon viz-adamw viz-compare
