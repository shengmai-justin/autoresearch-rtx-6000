#!/bin/bash
set -e

# ============================================================
# Run the original autoresearch loop with Qwen3.5-9B via Ollama + OpenCode
#
# Usage:
#   bash run_baseline.sh
#   bash run_baseline.sh --skip-ollama-pull   # if model already downloaded
# ============================================================

LLM_GPU=6
TRAIN_GPU=7
MODEL="qwen3.5:9b"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # autoresearch-rtx-6000/
BRANCH="autoresearch/qwen35-baseline-$(date +%b%d | tr '[:upper:]' '[:lower:]')"

SKIP_PULL=false
for arg in "$@"; do
    case $arg in
        --skip-ollama-pull) SKIP_PULL=true; shift ;;
    esac
done

echo "=== Autoresearch Baseline Setup ==="
echo "LLM GPU:    $LLM_GPU (Ollama)"
echo "Train GPU:  $TRAIN_GPU (train.py)"
echo "Model:      $MODEL"
echo "Repo:       $REPO_DIR"
echo "Branch:     $BRANCH"
echo ""

# -----------------------------------------------------------
# 1. Install Ollama (if not present)
# -----------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/ollama:$LD_LIBRARY_PATH"

if ! command -v ollama &> /dev/null; then
    echo "--- Installing Ollama (no sudo) ---"
    curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o /tmp/ollama.tgz
    mkdir -p "$HOME/.local"
    tar -C "$HOME/.local" -xzf /tmp/ollama.tgz
    rm /tmp/ollama.tgz
    echo "Installed to $HOME/.local/bin/ollama"
else
    echo "--- Ollama already installed: $(ollama --version) ---"
fi
echo ""

# -----------------------------------------------------------
# 2. Start Ollama server on LLM GPU
# -----------------------------------------------------------
echo "--- Starting Ollama server on GPU $LLM_GPU ---"
# Kill existing ollama serve if running
pkill -f "ollama serve" 2>/dev/null || true
sleep 1

CUDA_VISIBLE_DEVICES=$LLM_GPU nohup ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"

# Wait for server to be ready
echo "Waiting for Ollama server..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama server ready."
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Ollama server failed to start. Check /tmp/ollama.log"
        exit 1
    fi
    sleep 1
done
echo ""

# -----------------------------------------------------------
# 3. Pull model
# -----------------------------------------------------------
if [ "$SKIP_PULL" = false ]; then
    echo "--- Pulling $MODEL ---"
    ollama pull $MODEL
    echo ""
else
    echo "--- Skipping model pull ---"
    echo ""
fi

# -----------------------------------------------------------
# 4. Install OpenCode (if not present)
# -----------------------------------------------------------
if ! command -v opencode &> /dev/null; then
    echo "--- Installing OpenCode ---"
    npm install -g opencode
else
    echo "--- OpenCode already installed ---"
fi
echo ""

# -----------------------------------------------------------
# 5. Create OpenCode config
# -----------------------------------------------------------
echo "--- Writing opencode.json ---"
cat > "$REPO_DIR/opencode.json" << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local Ollama",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "qwen3.5:9b": {
          "name": "Qwen3.5 9B",
          "limit": {
            "context": 131072,
            "output": 32768
          }
        }
      }
    }
  }
}
EOF
echo "Written to $REPO_DIR/opencode.json"
echo ""

# -----------------------------------------------------------
# 6. Create branch and launch OpenCode
# -----------------------------------------------------------
cd "$REPO_DIR"

echo "--- Setting up git branch ---"
if git show-ref --verify --quiet "refs/heads/$BRANCH" 2>/dev/null; then
    echo "Branch $BRANCH already exists, checking out..."
    git checkout "$BRANCH"
else
    echo "Creating branch $BRANCH..."
    git checkout -b "$BRANCH"
fi
echo ""

# -----------------------------------------------------------
# 7. Launch OpenCode
# -----------------------------------------------------------
echo "=== Setup complete ==="
echo ""
echo "Launching OpenCode..."
echo "Tell it: Read program.md and follow the instructions. Start the experiment loop."
echo ""
echo "Train.py will use GPU $TRAIN_GPU via CUDA_VISIBLE_DEVICES."
echo "To stop: Ctrl+C, then 'pkill -f \"ollama serve\"' to stop Ollama."
echo ""

export CUDA_VISIBLE_DEVICES=$TRAIN_GPU
opencode
