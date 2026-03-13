#!/bin/bash
set -e

# ============================================================
# Cloud server setup script
# Run this ONCE on the cloud server from inside rl_pipeline/:
#   cd autoresearch-rtx-6000/rl_pipeline
#   bash setup_server.sh
#
# Options:
#   bash setup_server.sh --model Qwen/Qwen3.5-9B
#   bash setup_server.sh --skip-model   # skip model download
#   bash setup_server.sh --skip-data    # skip data preparation
# ============================================================

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
MODEL_DIR="./models/$(basename $MODEL_ID)"
REPO_PATH=".."   # parent dir is autoresearch-rtx-6000

# Parse args
SKIP_MODEL=false
SKIP_DATA=false
for arg in "$@"; do
    case $arg in
        --model) MODEL_ID="$2"; MODEL_DIR="./models/$(basename $MODEL_ID)"; shift 2 ;;
        --skip-model) SKIP_MODEL=true; shift ;;
        --skip-data) SKIP_DATA=true; shift ;;
    esac
done

echo "=== AutoResearch-RL Server Setup ==="
echo "Model: $MODEL_ID"
echo "Model dir: $MODEL_DIR"
echo "Repo path: $REPO_PATH"
echo ""

# -----------------------------------------------------------
# 1. Check GPU
# -----------------------------------------------------------
echo "--- Checking GPUs ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Make sure CUDA drivers are installed."
fi
echo ""

# -----------------------------------------------------------
# 2. Install uv (Python package manager)
# -----------------------------------------------------------
echo "--- Installing uv ---"
if command -v uv &> /dev/null; then
    echo "uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed: $(uv --version)"
fi
echo ""

# -----------------------------------------------------------
# 3. Install RL pipeline Python dependencies
# -----------------------------------------------------------
echo "--- Installing RL pipeline dependencies ---"
uv sync
echo ""

# -----------------------------------------------------------
# 4. Download the base model
# -----------------------------------------------------------
if [ "$SKIP_MODEL" = false ]; then
    echo "--- Downloading model: $MODEL_ID ---"
    if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
        echo "Model already downloaded at $MODEL_DIR"
    else
        mkdir -p "$MODEL_DIR"
        uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$MODEL_ID',
    local_dir='$MODEL_DIR',
    ignore_patterns=['*.gguf', '*.ggml'],
)
print('Download complete.')
"
    fi
    echo ""
else
    echo "--- Skipping model download ---"
    echo ""
fi

# -----------------------------------------------------------
# 5. Prepare autoresearch environment (data + tokenizer)
# -----------------------------------------------------------
if [ "$SKIP_DATA" = false ]; then
    echo "--- Preparing autoresearch data ---"
    cd "$REPO_PATH"
    uv sync
    uv run prepare.py
    cd -
    echo ""
else
    echo "--- Skipping data preparation ---"
    echo ""
fi

# -----------------------------------------------------------
# 6. Create output directories
# -----------------------------------------------------------
mkdir -p checkpoints trajectories

# -----------------------------------------------------------
# Done
# -----------------------------------------------------------
echo "=== Setup complete ==="
echo ""
echo "To run RL training:"
echo "  uv run rl_train.py"
echo ""
echo "To run ablation (no RL):"
echo "  uv run rl_train.py --algo none"
echo ""
echo "To run ablation (real autoresearch loop):"
echo "  uv run rl_train.py --algo none --keep-if-improved"
echo ""
echo "To evaluate a checkpoint:"
echo "  uv run rl_evaluate.py --checkpoint checkpoints/step_50.pt"
