#!/bin/bash

# setup.sh - Create uv environment for Moonlight project

set -e  # Exit on any error

echo "Setting up Moonlight project with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment with uv
echo "Creating uv virtual environment..."
uv venv --python 3.10

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies from requirements.txt
echo "Installing dependencies..."
uv pip install -r requirements.txt

echo ""
echo "Setup complete! To activate the environment, run:"
echo "source .venv/bin/activate"
echo ""
echo "To run training:"
echo "python3 examples/toy_train.py --model qwen --optimizer muon --dataset openwebtext-100k --hidden_size 896 --lr 1e-3"