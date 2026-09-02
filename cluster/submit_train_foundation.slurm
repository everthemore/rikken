#!/usr/bin/env bash
# =============================================================================
# cluster/submit_train_foundation.slurm — ALICE HPC GPU Job for Foundation Pre-training
# =============================================================================
#
# Trains BVN and Belief Network on the stratified counterfactual dataset.
#
# =============================================================================

#SBATCH --job-name=rikken_foundation_train
#SBATCH --output=cluster/slurm_logs/train_foundation_%j.out
#SBATCH --error=cluster/slurm_logs/train_foundation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # 8 CPU cores
#SBATCH --mem=16G                     # 16 GB RAM
#SBATCH --gres=gpu:1                  # Explicit GPU allocation
#SBATCH --time=01:30:00               # 1.5 hours max
#SBATCH --partition=gpu-short         # ALICE GPU partition

set -e

# Robust Conda Activation for ALICE compute nodes
module load miniconda3 2>/dev/null || module load Anaconda3 2>/dev/null || true
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate rikken_ai 2>/dev/null || conda activate rikken 2>/dev/null || source activate rikken_ai 2>/dev/null || true

PYTHON="python"
for candidate in "$CONDA_PREFIX/bin/python" "$HOME/miniforge3/envs/rikken_ai/bin/python" "$HOME/miniconda3/envs/rikken_ai/bin/python" "$HOME/.conda/envs/rikken_ai/bin/python" "$HOME/anaconda3/envs/rikken_ai/bin/python"; do
    if [ -x "$candidate" ]; then
        PYTHON="$candidate"
        break
    fi
done

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

DATA_PATH=${DATA_PATH:-"data/stratified"}
MODEL_PATH=${MODEL_PATH:-"checkpoints"}
EPOCHS=${EPOCHS:-15}
BATCH_SIZE=${BATCH_SIZE:-512}

echo "=== Training Foundation Models on GPU $(hostname) ==="
echo "Data: $DATA_PATH | Epochs: $EPOCHS | Batch size: $BATCH_SIZE"

$PYTHON -m training.train_foundation \
    --data-path "$DATA_PATH" \
    --model-path "$MODEL_PATH" \
    --bvn-epochs $EPOCHS \
    --bn-epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --device cuda

echo "=== Foundation Pre-training Complete ==="
