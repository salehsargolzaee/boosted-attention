#!/bin/bash
# Setup script for RunPod A100 instances.
# Run this once after SSH-ing in.
#
# Usage:  bash setup_pod.sh

set -e

echo "============================================"
echo "  Setting up RunPod for boosted-attention"
echo "============================================"

# --- System packages ---
apt-get update -qq && apt-get install -y -qq tmux htop > /dev/null 2>&1
echo "[1/5] System packages installed"

# --- Clone repo ---
cd /workspace
if [ ! -d "boosted-attention" ]; then
    git clone https://github.com/salehsargolzaee/boosted-attention.git
    echo "[2/5] Repository cloned"
else
    cd boosted-attention && git pull && cd ..
    echo "[2/5] Repository updated"
fi

# --- Python dependencies ---
pip install -q torch torchvision torchaudio --upgrade 2>/dev/null
pip install -q datasets transformers tokenizers wandb lm-eval 2>/dev/null
echo "[3/5] Python packages installed"

# --- Create directories ---
mkdir -p /workspace/data /workspace/checkpoints /workspace/logs /workspace/results
echo "[4/5] Directories created"

# --- Pre-tokenize OpenWebText ---
echo "[5/5] Tokenizing OpenWebText (this takes ~30-60 min)..."
cd /workspace/boosted-attention
python -u -c "
from experiments.train_openwebtext import prepare_data
prepare_data(seq_len=1024)
print('Data preparation complete.')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Run experiments with:"
echo "    bash scripts/run_125m.sh"
echo "    bash scripts/run_350m.sh"
echo "============================================"
