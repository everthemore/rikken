#!/usr/bin/env bash
# =============================================================================
# cluster/submit_stratified.slurm — ALICE HPC CPU Array Job for Stratified Pre-training
# =============================================================================
#
# Generates balanced counterfactual contract datasets across all 15 Rikken contracts.
#
# =============================================================================

#SBATCH --job-name=rikken_stratified
#SBATCH --output=cluster/slurm_logs/stratified_%A_%a.out
#SBATCH --error=cluster/slurm_logs/stratified_%A_%a.err
#SBATCH --array=0-49                  # 50 parallel workers by default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # 4 CPU cores per task
#SBATCH --mem=4G                      # 4 GB RAM per task
#SBATCH --time=01:30:00               # 1.5 hours max
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

# Parameters
GAMES_PER_CONTRACT=${GAMES_PER_CONTRACT:-25}
ROLLOUTS=${ROLLOUTS:-50}
DETERMINIZATIONS=${DETERMINIZATIONS:-5}
OUTPUT_DIR=${OUTPUT_DIR:-"data/stratified"}
WORKER_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "=== Rikken Stratified Generator Worker $WORKER_ID on $(hostname) ==="
echo "Games per Contract: $GAMES_PER_CONTRACT | Rollouts: $ROLLOUTS | Det: $DETERMINIZATIONS"

$PYTHON -m training.stratified_gen \
    --games-per-contract $GAMES_PER_CONTRACT \
    --rollouts $ROLLOUTS \
    --determinizations $DETERMINIZATIONS \
    --worker-id $WORKER_ID \
    --output-dir "$OUTPUT_DIR"

echo "=== Worker $WORKER_ID finished ==="
