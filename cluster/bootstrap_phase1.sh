#!/usr/bin/env bash
# =============================================================================
# cluster/bootstrap_phase1.sh — Fast 1M Game Bootstrap & Training on ALICE
# =============================================================================
#
# Dispatches:
#   1. 50 parallel CPU workers generating 20,000 games each (1,000,000 total games, ~3 mins)
#   2. Dependent GPU job training BVN and BN models on the generated 1M dataset (~3 mins)
#
# Usage:
#   bash cluster/bootstrap_phase1.sh
# =============================================================================

set -e
mkdir -p data cluster/slurm_logs checkpoints

echo "=========================================================="
echo "  Launching Phase 1 Bootstrap: 1,000,000 Games on ALICE"
echo "=========================================================="

# Step 1: 50 parallel CPU tasks (20,000 games each = 1,000,000 games)
GEN_JOB=$(sbatch \
    --parsable \
    --array=0-49 \
    --export=ALL,GAMES_PER_WORKER=20000 \
    cluster/submit_phase1.slurm)

echo "  -> Dispatched 50 Data Generation Workers (Job ID: $GEN_JOB)"

# Step 2: GPU Training job (starts as soon as all 50 data workers finish)
TRAIN_JOB=$(sbatch \
    --parsable \
    --dependency=afterok:${GEN_JOB} \
    --partition=gpu-short,gpu-l4-24g,gpu-2080ti-11g,gpu-a100-80g \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=01:00:00 \
    --output=cluster/slurm_logs/bootstrap_train_%j.out \
    --error=cluster/slurm_logs/bootstrap_train_%j.err \
    --wrap="python -u main.py train-bvn --epochs 10 --batch-size 512 && python -u main.py train-bn --epochs 10 --batch-size 256")

echo "  -> Dispatched Dependent GPU Training Job (Job ID: $TRAIN_JOB)"
echo "=========================================================="
echo "Phase 1 pipeline scheduled! Track with: squeue -u \$USER"
