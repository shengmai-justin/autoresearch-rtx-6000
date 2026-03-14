#!/bin/bash
set -e

cd "$(dirname "$0")"

# -- Create venv + install deps via uv --
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.10
fi
echo "Syncing dependencies..."
uv sync

# -- Download model --
MODEL_DIR="./models/Qwen3.5-27B"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading Qwen3.5-27B..."
    uv run huggingface-cli download Qwen/Qwen3.5-27B --local-dir "$MODEL_DIR"
else
    echo "Model already exists at $MODEL_DIR"
fi

# -- Run --
echo "Starting TTT-Discover training..."
uv run python train.py "$@"
