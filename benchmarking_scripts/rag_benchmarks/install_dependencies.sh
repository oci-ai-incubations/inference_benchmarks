#!/bin/bash

# NVIDIA RAG Blueprint Benchmarking Suite - Dependency Installer

echo "Installing benchmarking dependencies..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: Not in a virtual environment. Consider activating one first."
    echo "You can activate the virtual environment with: source .venv/bin/activate"
fi

# Install Python dependencies using uv if available, otherwise pip
if command -v uv &> /dev/null; then
    echo "Using uv to install dependencies..."
    uv pip install -r requirements.txt
    uv pip install rouge-score sentence-transformers
else
    echo "Using pip to install dependencies..."
    pip install -r requirements.txt
    pip install rouge-score sentence-transformers
fi

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully!"
    echo ""
    echo "You can now run benchmarks using:"
    echo "  source .venv/bin/activate  # Activate virtual environment first"
    echo "  python run_benchmark.py --preset quick"
    echo "  python accuracy_evaluator.py"
    echo "  python example_usage.py"
else
    echo "✗ Error installing dependencies"
    exit 1
fi 