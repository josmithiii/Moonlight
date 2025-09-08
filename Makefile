# Makefile for Moonlight project
# Run 'make help' to see all available targets

PYTHON := python3
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

.PHONY: help setup clean train-muon train-adamw train-both compare-optimizers test-sizes all

help:
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
	@echo "  compare-optimizers - Train both optimizers with same config for comparison"
	@echo "  test-sizes     - Test different model sizes (512, 896, 1024)"
	@echo ""
	@echo "Convenience targets:"
	@echo "  all            - Run both documented examples"
	@echo ""
	@echo "Parameters (can override with make VAR=value):"
	@echo "  MODEL=$(MODEL), DATASET=$(DATASET), HIDDEN_SIZE=$(HIDDEN_SIZE)"
	@echo "  LR=$(LR), WD=$(WD)"

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

train-adamw: $(LOGS_DIR)
	@echo "Training with AdamW optimizer (documented example)..."
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) \
		--optimizer adamw \
		--dataset $(DATASET) \
		--hidden_size $(HIDDEN_SIZE) \
		--lr $(LR) \
		--wd $(WD)

# Run both documented examples
train-both: train-muon train-adamw

all: train-both

# Comparison experiments
compare-optimizers: $(LOGS_DIR)
	@echo "Comparing Muon vs AdamW with identical configurations..."
	@echo "Training with Muon..."
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) --optimizer muon --dataset $(DATASET) \
		--hidden_size $(HIDDEN_SIZE) --lr $(LR) --wd $(WD)
	@echo "Training with AdamW..."
	$(VENV_ACTIVATE) $(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL) --optimizer adamw --dataset $(DATASET) \
		--hidden_size $(HIDDEN_SIZE) --lr $(LR) --wd $(WD)
	@echo "Check $(LOGS_DIR)/ for training logs to compare performance"

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
	rm -f *.bin
	rm -rf __pycache__
	rm -rf examples/__pycache__
	@echo "Cleaned virtual environment, logs, cached datasets, and Python cache files"

# Environment check before running training
.PHONY: check-env list-logs smoke-test train-custom