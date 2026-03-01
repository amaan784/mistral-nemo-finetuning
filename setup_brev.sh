#!/bin/bash
# ============================================================
# Agentic World — Brev Instance Setup
# ============================================================
# Run this first on your Brev instance to install all deps.
#
# Usage:
#   chmod +x setup_brev.sh && ./setup_brev.sh
# ============================================================

set -e

echo "============================================"
echo "AGENTIC WORLD — BREV SETUP"
echo "============================================"

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Install Unsloth (handles torch, transformers, etc.)
echo "Installing Unsloth..."
pip install --upgrade pip
pip install unsloth

# Install additional deps
echo "Installing W&B, datasets, trl..."
pip install wandb datasets trl

# Login to W&B (interactive)
echo ""
echo "Login to Weights & Biases:"
echo "   Get your API key from https://wandb.ai/authorize"
wandb login

# Login to HuggingFace (optional, for pushing model)
echo ""
echo "Login to HuggingFace (optional, press Enter to skip):"
huggingface-cli login || echo "Skipped HF login"

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import unsloth
print(f'Unsloth: installed')

import wandb
print(f'W&B: {wandb.__version__}')

import datasets
print(f'Datasets: {datasets.__version__}')

import trl
print(f'TRL: {trl.__version__}')
"

echo ""
echo "============================================"
echo "SETUP COMPLETE"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Upload train.jsonl and eval.jsonl to this instance"
echo "  2. Run: python finetune.py"
echo "  3. After training: python inference.py --interactive"
echo ""
