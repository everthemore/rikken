#!/usr/bin/env bash
# =============================================================================
# cluster/submit_self_play.slurm — ALICE HPC CPU Array Job for Self-Play
# =============================================================================
#
# Submit with:
#   sbatch cluster/submit_self_play.slurm
#
# Or customize array range and parameters on the fly:
#   sbatch --array=0-49 cluster/submit_self_play.slurm
#
# =============================================================================

#SBATCH --job-name=rikken_self_play
#SBATCH --output=cluster/slurm_logs/self_play_%A_%a.out
#SBATCH --error=cluster/slurm_logs/self_play_%A_%a.err
#SBATCH --array=0-49                  # 50 parallel workers by default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # 4 CPU cores per task
#SBATCH --mem=4G                      # 4 GB RAM per task (ALICE limit: 16G/core)
#SBATCH --time=02:00:00               # 2 hours max per shard
#SBATCH --partition=cpu-short,cpu-medium   # ALICE short CPU partitions

set -e

# Load ALICE modules & Conda
module load miniconda3 2>/dev/null || module load Anaconda3 2>/dev/null || true
if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
conda activate rikken_ai 2>/dev/null || source activate rikken_ai 2>/dev/null || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Parameters (can be passed via environment or defaults)
GAMES_PER_WORKER=${GAMES_PER_WORKER:-500}
ROLLOUTS=${ROLLOUTS:-400}
DETERMINIZATIONS=${DETERMINIZATIONS:-20}
ITERATION=${ITERATION:-1}
OUTPUT_DIR="data/self_play/iter_${ITERATION}"

mkdir -p "$OUTPUT_DIR" cluster/slurm_logs

echo "=========================================================="
echo "  Rikken AI — ALICE HPC Self-Play Task"
echo "  Job ID: $SLURM_ARRAY_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo "  Host: $(hostname) | Partition: $SLURM_JOB_PARTITION"
echo "  Games: $GAMES_PER_WORKER | Rollouts: $ROLLOUTS | Det: $DETERMINIZATIONS"
echo "  Output: $OUTPUT_DIR"
echo "=========================================================="

python -u training/self_play.py \
    --games "$GAMES_PER_WORKER" \
    --rollouts "$ROLLOUTS" \
    --determinizations "$DETERMINIZATIONS" \
    --worker-id "$SLURM_ARRAY_TASK_ID" \
    --output-dir "$OUTPUT_DIR" \
    --bvn checkpoints/bvn_best.pt \
    --bn checkpoints/bn_best.pt \
    --device cpu

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
