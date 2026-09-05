#!/usr/bin/env bash
# =============================================================================
# cluster/submit_imitation.slurm — ALICE HPC GPU Job for Heuristic Imitation Warm-Start
# =============================================================================
#
# Generates natural deals from Heuristic expert and warm-starts BVN and Belief Network.
#
# =============================================================================

#SBATCH --job-name=rikken_imitation
#SBATCH --output=cluster/slurm_logs/imitation_%j.out
#SBATCH --error=cluster/slurm_logs/imitation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # 8 CPU cores
#SBATCH --mem=16G                     # 16 GB RAM
#SBATCH --gres=gpu:1                  # 1 GPU
#SBATCH --time=00:45:00               # 45 minutes max (usually finishes in ~10 mins)
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

DEALS=${DEALS:-25000}
PLAY_GAMES=${PLAY_GAMES:-2500}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-512}
MODEL_PATH=${MODEL_PATH:-"checkpoints"}

echo "=== Training Heuristic Imitation Warm-Start on GPU $(hostname) ==="
echo "Deals: $DEALS | Play games: $PLAY_GAMES | Epochs: $EPOCHS | Batch size: $BATCH_SIZE"

$PYTHON -m training.train_imitation \
    --deals $DEALS \
    --play-games $PLAY_GAMES \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --model-path "$MODEL_PATH" \
    --device cuda

echo "=== Heuristic Imitation Pre-training Complete ==="
