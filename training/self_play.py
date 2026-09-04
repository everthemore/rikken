"""
training/self_play.py — Phase 3 Self-Play Engine for Distributed HPC / SLURM.

Runs games where all 4 seats are controlled by NeuralAgents (BVN + BN + ISMCTS).
Generates high-quality self-play data for iterative policy improvement.

Features:
  - Multi-process / SLURM array job worker identification.
  - Generates sharded .npz datasets formatted identically to Phase 1 for seamless retraining.
  - Tracks detailed contract win-rates and tournament statistics.
  - Supports mixed tournaments (e.g. Neural vs Heuristic baseline, or New Model vs Old Model).

Usage:
    python -m training.self_play --games 1000 --rollouts 200 --worker-id 0 --output-dir data/self_play/
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from networks.bvn import NUM_CONTRACTS


import os
import argparse
import time
import logging
import numpy as np
from typing import Optional, List, Dict

from engine.state import RikkenState, Contract, Phase
from engine.game import RikkenGame
from agents.neural_agent import NeuralAgent
from agents.heuristic import HeuristicAgent
from training.data_gen import _pack_shard
import config

log = logging.getLogger(__name__)


def run_self_play_game(
    agents: List[NeuralAgent | HeuristicAgent],
    game: RikkenGame,
    rng: np.random.Generator,
) -> dict:
    """Run one game of self-play and collect state-action-outcome records."""
    state = game.reset()
    bid_records = []
    play_records = []
    prev_bid_history = np.zeros((4, NUM_CONTRACTS), dtype=np.int8)

    # ---- Bidding ----
    while state.phase == Phase.BIDDING:
        p = state.current_player
        action = agents[p].act(state)

        bid_records.append({
            'player': p,
            'hand': state.hands[p].copy(),
            'bid_history': prev_bid_history.copy(),
            'bid_taken': action,
        })

        if action != int(Contract.PAS):
            prev_bid_history[p, action] = 1

        state, reward = game.step(state, action)
        if reward is not None and state.phase == Phase.TERMINAL:
            return {'bid_records': [], 'play_records': [], 'contract': 0, 'winner': -1}

    # ---- Declaration ----
    if state.phase == Phase.TRICK_TAKING and state.trick_count == 0:
        d = state.declarer
        if Contract.is_trump_contract(state.contract):
            trump = agents[d].declare_trump(state)
            vraagaas = -1
            if state.contract in (Contract.RIK, Contract.RIK_BETER):
                vraagaas = agents[d].declare_vraagaas(state, trump)
            state = game.declare(state, trump_suit=trump, vraagaas_suit=vraagaas)

    # ---- Trick-taking ----
    while state.phase == Phase.TRICK_TAKING:
        p = state.current_player
        opponents = [q for q in range(4) if q != p]
        opp_hands = np.stack([state.hands[q] for q in opponents], axis=0)

        trick_onehot = np.zeros(52, dtype=np.int8)
        for card in state.current_trick:
            if card >= 0:
                trick_onehot[card] = 1

        play_records.append({
            'player': p,
            'own_hand': state.hands[p].copy(),
            'played': state.played_cards.copy(),
            'bid_history': prev_bid_history.copy(),
            'trick': trick_onehot.copy(),
            'void_matrix': state.void_matrix.copy(),
            'opp_hands': opp_hands.copy(),
            'trick_count': state.trick_count,
        })

        action = agents[p].act(state)
        state, reward = game.step(state, action)

    # Outcome assignment
    for rec in bid_records:
        p = rec['player']
        reward_p = float(game.get_reward(state, p))
        rec['outcome'] = reward_p
        rec['won'] = 1.0 if reward_p > 0 else 0.0

    declarer_won = (state.reward > 0) if state.reward is not None else False
    return {
        'bid_records': bid_records,
        'play_records': play_records,
        'contract': int(state.contract) if state.contract else -1,
        'declarer': state.declarer,
        'declarer_won': declarer_won,
    }


def generate_self_play(
    n_games: int,
    bvn_path: str = 'checkpoints/bvn_best.pt',
    bn_path: str = 'checkpoints/bn_best.pt',
    rollouts: int = 100,
    determinizations: int = 10,
    worker_id: int = 0,
    output_dir: str = 'data/self_play',
    device: str = config.DEVICE,
    seed: Optional[int] = None,
    epsilon: float = 0.15,
) -> None:
    """Run a batch of self-play games and save the shard."""
    os.makedirs(output_dir, exist_ok=True)
    actual_seed = (config.NUMPY_SEED + worker_id * 10000 + int(time.time() % 1000)) if seed is None else seed
    rng = np.random.default_rng(actual_seed)
    game = RikkenGame(rng=rng)

    # Initialize 4 NeuralAgents
    bvn_model = bvn_path if os.path.exists(bvn_path) else None
    bn_model  = bn_path  if os.path.exists(bn_path)  else None

    if bvn_model is None:
        log.warning(f"BVN checkpoint '{bvn_path}' not found — using fallback bidding heuristic.")
    if bn_model is None:
        log.warning(f"BN checkpoint '{bn_path}' not found — using uniform determinization.")

    agents = [
        NeuralAgent(
            seat=p,
            game=game,
            bvn=bvn_model,
            bn=bn_model,
            n_determinizations=determinizations,
            n_rollouts=rollouts,
            device=device,
            rng=rng,
            epsilon=epsilon,
        )
        for p in range(4)
    ]

    all_records = []
    contract_stats = {}
    wins = 0
    t0 = time.time()

    print(f"[Worker {worker_id:03d}] Starting {n_games:,} self-play games (Rollouts: {rollouts}, Det: {determinizations})...")

    games_completed = 0
    total_deals = 0
    max_deals = n_games * 15

    while games_completed < n_games and total_deals < max_deals:
        total_deals += 1
        res = run_self_play_game(agents, game, rng)
        if res.get('contract', 0) == 0 or not res.get('play_records'):
            # All-pass redeal: re-deal cards until an actual contract is played
            continue

        for rec in res['bid_records']:
            rec['type'] = 'bid'
            all_records.append(rec)
        for rec in res['play_records']:
            rec['type'] = 'play'
            all_records.append(rec)

        games_completed += 1
        c = res['contract']
        cname = Contract(c).name if c >= 0 else 'UNKNOWN'
        contract_stats[cname] = contract_stats.get(cname, 0) + 1
        if res.get('declarer_won'):
            wins += 1

        if games_completed % max(1, n_games // 5) == 0 or games_completed == n_games:
            elapsed = time.time() - t0
            rate = games_completed / elapsed
            print(f"[Worker {worker_id:03d}] Game {games_completed:5d}/{n_games} (Deals: {total_deals}) | {rate:.1f} g/s | WinRate: {wins/games_completed:.1%}")
            # Incremental save so data is always preserved even if interrupted
            shard_path = os.path.join(output_dir, f"self_play_shard_{worker_id:04d}.npz")
            _pack_shard(all_records, shard_path)

    shard_path = os.path.join(output_dir, f"self_play_shard_{worker_id:04d}.npz")
    _pack_shard(all_records, shard_path)
    print(f"[Worker {worker_id:03d}] Completed {games_completed} played games (Deals: {total_deals}). Saved {len(all_records)} records to {shard_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-Play RL Data Generator')
    parser.add_argument('--games',            type=int, default=100)
    parser.add_argument('--rollouts',         type=int, default=100)
    parser.add_argument('--determinizations', type=int, default=10)
    parser.add_argument('--worker-id',        type=int, default=0)
    parser.add_argument('--output-dir',       type=str, default='data/self_play')
    parser.add_argument('--bvn',              type=str, default='checkpoints/bvn_best.pt')
    parser.add_argument('--bn',               type=str, default='checkpoints/bn_best.pt')
    parser.add_argument('--device',           type=str, default=config.DEVICE)
    parser.add_argument('--seed',             type=int, default=None)
    parser.add_argument('--epsilon',          type=float, default=0.15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_self_play(
        n_games=args.games,
        bvn_path=args.bvn,
        bn_path=args.bn,
        rollouts=args.rollouts,
        determinizations=args.determinizations,
        worker_id=args.worker_id,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        epsilon=args.epsilon,
    )
