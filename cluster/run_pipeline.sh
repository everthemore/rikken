#!/usr/bin/env bash
# =============================================================================
# cluster/run_pipeline.sh — Automated Master Loop on Leiden ALICE HPC
# =============================================================================
#
# Runs self-play reinforcement learning iterations using SLURM job dependencies.
# Automatically chains Self-Play -> Retrain -> Next Gen Self-Play in the queue.
#
# Usage:
#   bash cluster/run_pipeline.sh [options]
#
# Options:
#   --iterations N          Number of RL iterations (default: 5)
#   --workers N             Parallel SLURM tasks in array (default: 50)
#   --games-per-worker N    Games per worker per iteration (default: 500 => 25,000 games/iter)
#   --rollouts N            ISMCTS rollouts (default: 400)
#   --determinizations N    ISMCTS determinizations (default: 20)
#   --buffer-window N       Rolling replay window size (default: 5)
#   --retrain-epochs N      Training epochs on GPU (default: 10)
#
# =============================================================================

set -e

ITERATIONS=5
WORKERS=50
GAMES_PER_WORKER=500
ROLLOUTS=200
DETERMINIZATIONS=10
BUFFER_WINDOW=5
RETRAIN_EPOCHS=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)          ITERATIONS="$2"; shift 2 ;;
        --workers)             WORKERS="$2"; shift 2 ;;
        --games-per-worker)    GAMES_PER_WORKER="$2"; shift 2 ;;
        --rollouts)            ROLLOUTS="$2"; shift 2 ;;
        --determinizations)    DETERMINIZATIONS="$2"; shift 2 ;;
        --buffer-window)       BUFFER_WINDOW="$2"; shift 2 ;;
        --retrain-epochs)      RETRAIN_EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TOTAL_GAMES_PER_ITER=$(( WORKERS * GAMES_PER_WORKER ))
ARRAY_MAX=$(( WORKERS - 1 ))

mkdir -p cluster/slurm_logs checkpoints data/self_play

echo "=========================================================="
echo "  Rikken AI — Leiden ALICE Master RL Pipeline"
echo "  Iterations: $ITERATIONS | Workers: $WORKERS ($TOTAL_GAMES_PER_ITER games/iter)"
echo "  MCTS Rollouts: $ROLLOUTS | Det: $DETERMINIZATIONS | Buffer Window: $BUFFER_WINDOW"
echo "=========================================================="

# Find latest iteration number from data/self_play/
LATEST_ITER=$(find data/self_play/ -maxdepth 1 -name "iter_*" -type d 2>/dev/null | grep -o '[0-9]\+' | sort -n | tail -n 1 || echo 0)
START_ITER=$(( LATEST_ITER + 1 ))
END_ITER=$(( START_ITER + ITERATIONS - 1 ))

echo "Chaining Generations $START_ITER through $END_ITER in SLURM DAG..."

PREV_RETRAIN_JOB=""

for (( iter=START_ITER; iter<=END_ITER; iter++ )); do
    echo ""
    echo "##########################################################"
    echo "  CONFIGURING GENERATION $iter / $END_ITER"
    echo "##########################################################"

    # Step 1: Submit Parallel Self-Play Array on ALICE CPU nodes
    EXTRA_DEP=""
    if [ -n "$PREV_RETRAIN_JOB" ]; then
        EXTRA_DEP="--dependency=afterok:${PREV_RETRAIN_JOB}"
        echo "[Gen $iter] Self-play queued (will start after Retrain Job $PREV_RETRAIN_JOB completes)..."
    else
        echo "[Gen $iter] Submitting $WORKERS CPU workers ($TOTAL_GAMES_PER_ITER games)..."
    fi

    SELF_PLAY_JOB=$(sbatch         --parsable         $EXTRA_DEP         --array=0-${ARRAY_MAX}         --export=ALL,ITERATION=${iter},GAMES_PER_WORKER=${GAMES_PER_WORKER},ROLLOUTS=${ROLLOUTS},DETERMINIZATIONS=${DETERMINIZATIONS}         cluster/submit_self_play.slurm)

    echo "  -> Self-Play Array Job ID: $SELF_PLAY_JOB"

    # Step 2: Submit GPU Retraining on ALICE GPU nodes (runs after this generation's self-play completes)
    echo "[Gen $iter] Scheduling GPU Retraining on Replay Buffer (afterok:$SELF_PLAY_JOB)..."
    RETRAIN_JOB=$(sbatch         --parsable         --dependency=afterok:${SELF_PLAY_JOB}         --export=ALL,ITERATION=${iter},BUFFER_WINDOW=${BUFFER_WINDOW},EPOCHS=${RETRAIN_EPOCHS}         cluster/submit_retrain.slurm)

    echo "  -> GPU Retraining Job ID: $RETRAIN_JOB"
    PREV_RETRAIN_JOB="$RETRAIN_JOB"
done

echo ""
echo "=========================================================="
echo "  All $ITERATIONS generations successfully chained in ALICE SLURM queue!"
echo "  Check queue status: squeue -u \$USER"
echo "  Monitor logs:       tail -f cluster/slurm_logs/*.out"
echo "=========================================================="
