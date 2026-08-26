#!/usr/bin/env bash
# =============================================================================
# cluster/submit_retrain.slurm — ALICE HPC GPU Job with Rolling Replay Buffer
# =============================================================================
#
# Retrains BVN (Transformer) and Belief Network (ResNet) on an NVIDIA GPU
# using a sliding window of the latest self-play iterations.
#
# =============================================================================

#SBATCH --job-name=rikken_retrain
#SBATCH --output=cluster/slurm_logs/retrain_%j.out
#SBATCH --error=cluster/slurm_logs/retrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # 8 CPU cores
#SBATCH --mem=16G                     # 16 GB RAM
#SBATCH --gres=gpu:1                  # Explicit GPU allocation (required on ALICE)
#SBATCH --time=02:00:00               # 2 hours max
#SBATCH --partition=gpu-short  # ALICE GPU partitions

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

ITERATION=${ITERATION:-1}
BUFFER_WINDOW=${BUFFER_WINDOW:-5}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-512}

# Build comma-separated data paths for the rolling window
REPLAY_PATHS=""
WINDOW_START=$(( ITERATION - BUFFER_WINDOW + 1 ))
if [ $WINDOW_START -lt 1 ]; then WINDOW_START=1; fi

for (( k=WINDOW_START; k<=ITERATION; k++ )); do
    DIR="data/self_play/iter_${k}"
    if [ -d "$DIR" ]; then
        if [ -n "$REPLAY_PATHS" ]; then
            REPLAY_PATHS="$REPLAY_PATHS,$DIR"
        else
            REPLAY_PATHS="$DIR"
        fi
    fi
done

if [ -z "$REPLAY_PATHS" ]; then
    REPLAY_PATHS="data/self_play/iter_${ITERATION}"
fi

echo "=========================================================="
echo "  Rikken AI — ALICE HPC GPU Retraining"
echo "  Job ID: $SLURM_JOB_ID | Host: $(hostname) | Partition: $SLURM_JOB_PARTITION"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU allocated')"
echo "  Iteration: $ITERATION | Buffer Window: $BUFFER_WINDOW generations"
echo "  Replay Buffer Directories: $REPLAY_PATHS"
echo "=========================================================="

# 1. Retrain Bidding Value Network (BVN) on Replay Buffer
echo "--- 1. Retraining BVN on Rolling Replay Buffer ---"
python -u -m training.train_bvn \
    --data-path "$REPLAY_PATHS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --resume-latest

# 2. Retrain Belief Network (BN) on Replay Buffer
echo "--- 2. Retraining Belief Network on Rolling Replay Buffer ---"
python -u -m training.train_bn \
    --data-path "$REPLAY_PATHS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --resume-latest


# 3. Benchmark Tournament Evaluation vs Baseline & Log to eval_history.json
echo "--- 3. Evaluating Generation $ITERATION in Tournament ---"
python -u - << 'EVAL_EOF'
import os, json, time
import numpy as np
from training.tournament import run_tournament

iter_num = int(os.environ.get('ITERATION', '1'))
metrics = run_tournament(
    n_games=100,
    bvn_path='checkpoints/bvn_best.pt',
    bn_path='checkpoints/bn_best.pt',
    rollouts=50,
    determinizations=10,
    seed=42 + iter_num * 1000
)

history_file = 'eval_history.json'
history = []
if os.path.exists(history_file):
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except Exception:
        history = []

entry = {
    'iteration': iter_num,
    'type': f'generation_{iter_num}',
    'neural_win_rate': metrics['neural_win_rate'],
    'declarer_win_rate': metrics['declarer_win_rate'],
    'defender_win_rate': metrics['defender_win_rate'],
    'neural_avg_tricks_declared': metrics.get('neural_avg_tricks_declared'),
    'heuristic_avg_tricks_declared': metrics.get('heuristic_avg_tricks_declared'),
    'contract_breakdown': metrics.get('contract_breakdown', {}),
    'total_games': metrics.get('total_games', 100),
    'redeals': metrics.get('redeals', 0),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
}
history.append(entry)
with open(history_file, 'w') as f:
    json.dump(history, f, indent=2)

print(f"[Gen {iter_num}] Logged evaluation metrics to {history_file}")
EVAL_EOF

echo "=== ALICE Retraining Iteration $ITERATION complete! ==="
