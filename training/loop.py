"""
training/loop.py — Automated Local Iterative Self-Play RL Loop.

Performs iterative self-play reinforcement learning on the local machine:
  For each iteration:
    1. [Self-Play]  Generates N self-play games using current NeuralAgent (BVN + BN + ISMCTS).
    2. [Retrain]    Retrains/fine-tunes BVN & BN on the newly generated self-play data.
    3. [Evaluate]   Pits the updated NeuralAgent against the Heuristic baseline in a tournament.
    4. [Track]      Logs win-rates and progression to 'eval_history.json'.

Usage:
    python -m training.loop --iterations 3 --games-per-iter 100 --eval-games 50 --rollouts 50
"""

from __future__ import annotations
import os
import argparse
import time
import json
import logging
import numpy as np

from training.self_play import generate_self_play
from training.train_bvn import train as train_bvn
from training.train_bn import train as train_bn
from training.tournament import run_tournament
import config

log = logging.getLogger(__name__)


def run_self_play_loop(
    iterations: int = 3,
    games_per_iter: int = 100,
    eval_games: int = 50,
    rollouts: int = 50,
    determinizations: int = 10,
    retrain_epochs: int = 5,
    buffer_window: int = config.REPLAY_BUFFER_WINDOW,
    output_base: str = 'data/self_play',
    model_path: str = config.MODEL_PATH,
    history_file: str = 'eval_history.json',
) -> None:
    """Execute the iterative self-play RL improvement loop."""
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Load existing history if present
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []

    print(f"\n{'='*70}")
    print(f"  STARTING LOCAL SELF-PLAY REINFORCEMENT LEARNING LOOP")
    print(f"  Iterations: {iterations} | Games/Iter: {games_per_iter} | Eval Games: {eval_games}")
    print(f"  MCTS Rollouts: {rollouts} | Determinizations: {determinizations}")
    print(f"{'='*70}\n")

    bvn_path = os.path.join(model_path, 'bvn_best.pt')
    bn_path  = os.path.join(model_path, 'bn_best.pt')

    # Determine starting iteration based on existing history
    completed_iters = [h.get('iteration', 0) for h in history if h.get('type', '').startswith('generation_')]
    start_iter = (max(completed_iters) + 1) if completed_iters else 1
    end_iter = start_iter + iterations - 1

    # Run baseline evaluation only if history is completely empty
    if not history:
        print(">>> Running Baseline Evaluation (Generation 0 vs Heuristic)... <<<")
        baseline_metrics = run_tournament(
            n_games=eval_games,
            bvn_path=bvn_path,
            bn_path=bn_path,
            rollouts=rollouts,
            determinizations=determinizations,
            seed=42,
        )
        baseline_entry = {
            'iteration': 0,
            'type': 'baseline',
            'neural_win_rate': baseline_metrics['neural_win_rate'],
            'declarer_win_rate': baseline_metrics['declarer_win_rate'],
            'defender_win_rate': baseline_metrics['defender_win_rate'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        history.append(baseline_entry)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

    print(f"Continuing self-play RL: Running Generations {start_iter} through {end_iter} ({iterations} generations)...")

    for iter_num in range(start_iter, end_iter + 1):
        iter_dir = os.path.join(output_base, f"iter_{iter_num}")
        os.makedirs(iter_dir, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"  ITERATION {iter_num} (Batch: {iter_num - start_iter + 1} / {iterations})")
        print(f"{'#'*70}\n")

        # Step 1: Self-Play Generation
        t0 = time.time()
        print(f"[Iter {iter_num}] Generating {games_per_iter} Self-Play Games...")
        generate_self_play(
            n_games=games_per_iter,
            bvn_path=bvn_path,
            bn_path=bn_path,
            rollouts=rollouts,
            determinizations=determinizations,
            worker_id=0,
            output_dir=iter_dir,
            seed=42 + iter_num * 100,
        )
        gen_time = time.time() - t0

        # Step 2: Replay Buffer Aggregation (Last W iterations)
        window_start = max(1, iter_num - buffer_window + 1)
        replay_dirs = [
            os.path.join(output_base, f"iter_{k}")
            for k in range(window_start, iter_num + 1)
            if os.path.exists(os.path.join(output_base, f"iter_{k}"))
        ]
        replay_data_paths = ",".join(replay_dirs)
        print(f"\n[Iter {iter_num}] Replay Buffer: aggregating {len(replay_dirs)} iterations (iters {window_start}..{iter_num})")

        print(f"[Iter {iter_num}] Retraining BVN on Rolling Buffer ({retrain_epochs} epochs)...")
        train_bvn(
            epochs=retrain_epochs,
            data_path=replay_data_paths,
            model_path=model_path,
            resume_latest=True,
        )

        print(f"\n[Iter {iter_num}] Retraining BN on Rolling Buffer ({retrain_epochs} epochs)...")
        train_bn(
            epochs=retrain_epochs,
            data_path=replay_data_paths,
            model_path=model_path,
            resume_latest=True,
        )

        # Step 3: Tournament Evaluation vs Baseline
        print(f"\n[Iter {iter_num}] Evaluating Generation {iter_num} in Tournament...")
        eval_metrics = run_tournament(
            n_games=eval_games,
            bvn_path=bvn_path,
            bn_path=bn_path,
            rollouts=rollouts,
            determinizations=determinizations,
            seed=42 + iter_num * 1000,
        )

        iter_entry = {
            'iteration': iter_num,
            'type': f'generation_{iter_num}',
            'neural_win_rate': eval_metrics['neural_win_rate'],
            'declarer_win_rate': eval_metrics['declarer_win_rate'],
            'defender_win_rate': eval_metrics['defender_win_rate'],
            'gen_time_seconds': gen_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        history.append(iter_entry)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        # Print Evolution Summary Table
        print(f"\n{'='*70}")
        print(f"  POLICY EVOLUTION TRACKER (Updated after Iteration {iter_num})")
        print(f"{'='*70}")
        print(f"  {'Iteration':12s} | {'Overall WinRate':16s} | {'Declarer (Offense)':18s} | {'Defender (Defense)':18s}")
        print(f"  {'-'*68}")
        for h in history:
            print(f"  Gen {h['iteration']:2d} ({h['type']:10s}) | {h['neural_win_rate']:14.1%} | {h['declarer_win_rate']:16.1%} | {h['defender_win_rate']:16.1%}")
        print(f"{'='*70}\n")

    print(f"\nSelf-Play Loop complete! Evaluation history saved to '{history_file}'.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local Iterative Self-Play RL Loop')
    parser.add_argument('--iterations',     type=int, default=3)
    parser.add_argument('--games-per-iter', type=int, default=100)
    parser.add_argument('--eval-games',     type=int, default=50)
    parser.add_argument('--rollouts',       type=int, default=50)
    parser.add_argument('--determinizations', type=int, default=10)
    parser.add_argument('--retrain-epochs', type=int, default=5)
    parser.add_argument('--history-file',   type=str, default='eval_history.json')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    run_self_play_loop(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        eval_games=args.eval_games,
        rollouts=args.rollouts,
        determinizations=args.determinizations,
        retrain_epochs=args.retrain_epochs,
        buffer_window=args.buffer_window,
        history_file=args.history_file,
    )
