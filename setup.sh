#!/bin/bash

# setup.sh - Create uv environment for Moonlight project with full Cursor integration

set -e  # Exit on any error

echo "Setting up Moonlight project with uv and Cursor integration..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Initialize uv project if pyproject.toml doesn't exist
if [ ! -f "pyproject.toml" ]; then
    echo "Initializing uv project..."
    uv init --python 3.10
fi

# Add dependencies from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    echo "Adding dependencies from requirements.txt..."
    # Read requirements.txt and add each dependency
    while IFS= read -r line; do
        # Skip empty lines and comments
        if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
            echo "Adding dependency: $line"
            # Handle CUDA-specific torch versions gracefully
            if [[ "$line" == *"+cu"* ]]; then
                echo "  Note: Using CPU version of torch instead of CUDA version for compatibility"
                base_package=$(echo "$line" | sed 's/+cu[0-9]*//')
                uv add "$base_package" || echo "  Warning: Could not add $line, continuing..."
            else
                uv add "$line" || echo "  Warning: Could not add $line, continuing..."
            fi
        fi
    done < requirements.txt
else
    echo "No requirements.txt found, using dependencies from pyproject.toml..."
fi

# Create .vscode directory and launch.json for Cursor integration
echo "Setting up Cursor/VS Code debug configuration..."
mkdir -p .vscode

cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File (uv)",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "justMyCode": false,
            "python": "${workspaceFolder}/.venv/bin/python"
        },
        {
            "name": "Python: Matrix Factorization Compare (uv)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/examples/matrix_factorization_compare.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "justMyCode": false,
            "python": "${workspaceFolder}/.venv/bin/python",
            "args": []
        },
        {
            "name": "Python: Toy Train (uv)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/examples/toy_train.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "justMyCode": false,
            "python": "${workspaceFolder}/.venv/bin/python",
            "args": ["--model", "qwen", "--optimizer", "muon", "--dataset", "openwebtext-100k", "--hidden_size", "896", "--lr", "1e-3"]
        }
    ]
}
EOF

# Create .vscode/settings.json to set Python interpreter
cat > .vscode/settings.json << 'EOF'
{
    "python.pythonPath": "./.venv/bin/python",
    "python.defaultInterpreterPath": "./.venv/bin/python"
}
EOF

# Test that the environment works
echo "Testing uv environment..."
uv run python -c "import sys; print(f'Python: {sys.executable}'); import torch; print(f'PyTorch: {torch.__version__}')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "🔧 For command line usage:"
echo "  uv run python <script.py>           # Run any Python script"
echo "  uv run python                       # Interactive Python"
echo ""
echo "🐛 For Cursor debugging:"
echo "  1. Reload Cursor: Cmd+Shift+P → 'Developer: Reload Window'"
echo "  2. Select debug config: 'Python: Current File (uv)' or others"
echo "  3. Set breakpoints and press F5"
echo ""
echo "🏃 To run training:"
echo "  uv run python examples/toy_train.py --model qwen --optimizer muon --dataset openwebtext-100k --hidden_size 896 --lr 1e-3"
echo ""
echo "📝 The uv environment is at: $(pwd)/.venv/bin/python"
