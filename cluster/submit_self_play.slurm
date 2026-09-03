#!/usr/bin/env bash
# =============================================================================
# cluster/submit_self_play.slurm — ALICE HPC CPU Array Job for Self-Play
# =============================================================================
#
# Submit with:
#   sbatch cluster/submit_self_play.slurm
#
# =============================================================================

#SBATCH --job-name=rikken_self_play
#SBATCH --output=cluster/slurm_logs/self_play_%A_%a.out
#SBATCH --error=cluster/slurm_logs/self_play_%A_%a.err
#SBATCH --array=0-49                  # 50 parallel workers by default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # 4 CPU cores per task
#SBATCH --mem=4G                      # 4 GB RAM per task
#SBATCH --time=02:00:00               # 2 hours max per shard
#SBATCH --partition=cpu-short         # ALICE CPU partition

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

# Direct python interpreter fallback if conda activate didn't put it in PATH
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

# Parameters (supports positional args $1-$4 with fallback to env vars and safe defaults)
GAMES_PER_WORKER=${1:-${GAMES_PER_WORKER:-100}}
ROLLOUTS=${2:-${ROLLOUTS:-100}}
DETERMINIZATIONS=${3:-${DETERMINIZATIONS:-15}}
ITERATION=${4:-${ITERATION:-1}}
OUTPUT_DIR="data/self_play/iter_${ITERATION}"

mkdir -p "$OUTPUT_DIR" cluster/slurm_logs

echo "=========================================================="
echo "  Rikken AI — ALICE HPC Self-Play Task"
echo "  Job ID: $SLURM_ARRAY_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo "  Host: $(hostname) | Partition: $SLURM_JOB_PARTITION"
echo "  Python Binary: $($PYTHON -c 'import sys; print(sys.executable)')"
echo "  Torch Available: $($PYTHON -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "  Games: $GAMES_PER_WORKER | Rollouts: $ROLLOUTS | Det: $DETERMINIZATIONS"
echo "  Output: $OUTPUT_DIR"
echo "=========================================================="

$PYTHON -u -m training.self_play \
    --games "$GAMES_PER_WORKER" \
    --rollouts "$ROLLOUTS" \
    --determinizations "$DETERMINIZATIONS" \
    --worker-id "$SLURM_ARRAY_TASK_ID" \
    --output-dir "$OUTPUT_DIR" \
    --bvn checkpoints/bvn_best.pt \
    --bn checkpoints/bn_best.pt \
    --device cpu

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
