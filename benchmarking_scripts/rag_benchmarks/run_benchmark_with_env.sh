#!/bin/bash

# NVIDIA RAG Blueprint Benchmarking Suite - Runner with Virtual Environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "NVIDIA RAG Blueprint Benchmarking Suite"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "Error: Virtual environment not found at $PROJECT_ROOT/.venv"
    echo "Please run the setup first:"
    echo "  cd $PROJECT_ROOT"
    echo "  uv venv"
    echo "  source .venv/bin/activate"
    echo "  uv pip install -e .[all]"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

echo "Virtual environment activated: $VIRTUAL_ENV"

# Change to benchmark directory
cd "$SCRIPT_DIR"

# Check if dependencies are installed
if ! python -c "import psutil, numpy, matplotlib, rich" 2>/dev/null; then
    echo "Installing benchmarking dependencies..."
    ./install_dependencies.sh
fi

# Run the benchmark with provided arguments
echo "Running benchmark with arguments: $@"
python run_benchmark.py "$@" 