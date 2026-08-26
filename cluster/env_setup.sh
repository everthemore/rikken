#!/usr/bin/env bash
# =============================================================================
# cluster/env_setup.sh — Leiden ALICE HPC Environment Setup
# =============================================================================
#
# Run once after logging into ALICE (login node):
#   bash cluster/env_setup.sh
#
# =============================================================================

set -e

echo "=== Setting up Rikken AI Environment on ALICE HPC Cluster ==="

# 1. Check / Install Miniforge if conda is not in PATH
if ! command -v conda &>/dev/null && [ ! -f "$HOME/miniforge3/bin/conda" ] && [ ! -f "$HOME/miniconda3/bin/conda" ]; then
    echo "[1/4] Conda not detected. Installing Miniforge into ~/miniforge3 (ALICE standard)..."
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/Miniforge3.sh
    bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
    rm -f /tmp/Miniforge3.sh
    "$HOME/miniforge3/bin/conda" init bash
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
else
    echo "[1/4] Conda found."
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
    elif [ -f "$HOME/miniforge3/bin/conda" ]; then
        eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
    elif [ -f "$HOME/miniconda3/bin/conda" ]; then
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    fi
fi

# Ensure conda is active
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

# 2. Create Conda Environment if not already present
if conda info --envs | grep -q "rikken_ai"; then
    echo "[2/4] Conda environment 'rikken_ai' already exists. Activating..."
    conda activate rikken_ai
else
    echo "[2/4] Creating conda environment 'rikken_ai' (Python 3.10)..."
    conda create -y -n rikken_ai python=3.10
    conda activate rikken_ai
fi

# 3. Install PyTorch with CUDA support (ALICE NVIDIA GPUs)
echo "[3/4] Installing PyTorch with CUDA support..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install requirements
echo "[4/4] Installing NumPy and project dependencies..."
pip install numpy tqdm matplotlib scikit-learn pytest

# 5. Create required project directories
mkdir -p cluster/slurm_logs checkpoints data/self_play

echo ""
echo "=== Verification ==="
python -c "
import torch, numpy, sklearn
print(f'Python Interpreter: {torch.__file__}')
print(f'PyTorch Version: {torch.__version__}')
print(f'PyTorch CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
"

echo "=== ALICE Environment Setup Complete! ==="
