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

# 1. Load Conda / Python Module on ALICE
module load miniconda3 2>/dev/null || module load Anaconda3 2>/dev/null || true

# Initialize conda in current subshell
if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

# 2. Create Conda Environment if not already present
if conda info --envs | grep -q "rikken_ai"; then
    echo "[1/4] Conda environment 'rikken_ai' already exists. Activating..."
    conda activate rikken_ai
else
    echo "[1/4] Creating conda environment 'rikken_ai' (Python 3.10)..."
    conda create -y -n rikken_ai python=3.10
    conda activate rikken_ai
fi

# 3. Install PyTorch with CUDA support (ALICE NVIDIA GPUs)
echo "[2/4] Installing PyTorch with CUDA support..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install requirements
echo "[3/4] Installing NumPy and project dependencies..."
pip install numpy tqdm matplotlib

# 5. Create required project directories
echo "[4/4] Creating cluster log and data directories..."
mkdir -p cluster/slurm_logs checkpoints data/self_play

echo "=== Verification ==="
python -c "
import torch, numpy
print(f'Python: {torch.__version__}')
print(f'PyTorch CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
"

echo "=== ALICE Environment Setup Complete! ==="
