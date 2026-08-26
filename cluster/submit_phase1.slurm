#!/bin/bash
# =============================================================================
# cluster/submit_phase1.slurm — Parallel Phase 1 Dataset Generation on ALICE
# =============================================================================
#
# Generates 1,000,000 bootstrap games using 50 CPU tasks (20,000 games per worker).
#
# SBATCH Configuration for Leiden ALICE:
#SBATCH --job-name=rikken_phase1
#SBATCH --partition=cpu-short,short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/slurm_logs/phase1_%A_%a.out
#SBATCH --error=cluster/slurm_logs/phase1_%A_%a.err

set -e

# Load environment
module load miniconda3 2>/dev/null || module load Anaconda3 2>/dev/null || true
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || true
conda activate rikken 2>/dev/null || source activate rikken 2>/dev/null || true

WORKER_ID=${SLURM_ARRAY_TASK_ID:-0}
GAMES_PER_WORKER=${GAMES_PER_WORKER:-20000}
OUTPUT_DIR="data"

echo "=========================================================="
echo "  Rikken AI — Phase 1 Bootstrap Worker $WORKER_ID"
echo "  Node: $(hostname) | Date: $(date)"
echo "  Generating $GAMES_PER_WORKER games to $OUTPUT_DIR/shard_$(printf "%04d" $WORKER_ID).npz"
echo "=========================================================="

python -u main.py generate \
    --games "$GAMES_PER_WORKER" \
    --workers 1 \
    --output-dir "$OUTPUT_DIR"

echo "Worker $WORKER_ID completed successfully."
