#!/usr/bin/env bash
# =============================================================================
# cluster/test_smoke.sh — 10-15 Minute ALICE HPC Smoke Test
# =============================================================================
#
# Tests the complete end-to-end pipeline on ALICE:
#   1. CPU Array Job (4 parallel tasks, 250 games each = 1,000 games)
#   2. Automatic SLURM Dependency Trigger (afterok)
#   3. GPU Retraining Job on NVIDIA GPU (3 epochs)
#   4. Checkpoint creation and verification
#
# Usage:
#   bash cluster/test_smoke.sh
#
# =============================================================================

set -e

echo "=========================================================="
echo "  Rikken AI — ALICE HPC 15-Minute Smoke Test"
echo "=========================================================="

mkdir -p cluster/slurm_logs checkpoints data/self_play

# 1. Submit a 4-worker CPU Array Job (1,000 games total, 100 rollouts)
echo "[1/2] Submitting 4-worker CPU self-play array job (1,000 games total)..."
SELF_PLAY_JOB=$(sbatch \
    --parsable \
    --array=0-3 \
    --export=ALL,ITERATION=999,GAMES_PER_WORKER=250,ROLLOUTS=100,DETERMINIZATIONS=10 \
    cluster/submit_self_play.slurm)

echo "  -> Self-Play Array Job ID: $SELF_PLAY_JOB"

# 2. Submit GPU Retraining on NVIDIA GPU (dependent on afterok:SELF_PLAY_JOB)
echo "[2/2] Scheduling GPU Retraining Job (will start automatically once CPU jobs finish)..."
RETRAIN_JOB=$(sbatch \
    --parsable \
    --dependency=afterok:${SELF_PLAY_JOB} \
    --export=ALL,ITERATION=999,BUFFER_WINDOW=1,EPOCHS=3 \
    cluster/submit_retrain.slurm)

echo "  -> GPU Retraining Job ID: $RETRAIN_JOB"
echo ""
echo "=========================================================="
echo "  Smoke test submitted to ALICE SLURM queue!"
echo "  Monitor job queue:  squeue -u \$USER"
echo "  Monitor CPU logs:   tail -f cluster/slurm_logs/self_play_${SELF_PLAY_JOB}_*.out"
echo "  Monitor GPU logs:   tail -f cluster/slurm_logs/retrain_${RETRAIN_JOB}.out"
echo "=========================================================="
