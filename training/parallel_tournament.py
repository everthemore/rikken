"""
training/parallel_tournament.py — High-throughput parallel tournament runner.
Runs N games across CPU workers rotating team seats to guarantee zero bias.
"""

import os
import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath('.'))

from engine.game import RikkenGame
from engine.state import RikkenState, Contract, Phase
from agents.heuristic import HeuristicAgent
from agents.neural_agent import NeuralAgent


def _worker_task(args):
    worker_id, n_games, bvn_path, bn_path, rollouts, determinizations, seed = args
    rng = np.random.default_rng(seed)
    game = RikkenGame(rng=rng, use_early_stop=False)

    neural_agents = [
        NeuralAgent(
            seat=0, game=game,
            bvn=bvn_path if os.path.exists(bvn_path) else None,
            bn=bn_path if os.path.exists(bn_path) else None,
            n_rollouts=rollouts,
            n_determinizations=determinizations,
            device="cpu",
            rng=rng,
        ),
        NeuralAgent(
            seat=2, game=game,
            bvn=bvn_path if os.path.exists(bvn_path) else None,
            bn=bn_path if os.path.exists(bn_path) else None,
            n_rollouts=rollouts,
            n_determinizations=determinizations,
            device="cpu",
            rng=rng,
        )
    ]
    heuristic_agents = [HeuristicAgent(seat=1, rng=rng), HeuristicAgent(seat=3, rng=rng)]

    records = []
    for g in range(n_games):
        team_a_seats = (0, 2) if (g % 2 == 0) else (1, 3)
        team_b_seats = (1, 3) if (g % 2 == 0) else (0, 2)

        active_agents = [None] * 4
        for i, p in enumerate(team_a_seats):
            neural_agents[i].set_seat(p)
            active_agents[p] = neural_agents[i]
        for i, p in enumerate(team_b_seats):
            heuristic_agents[i].set_seat(p)
            active_agents[p] = heuristic_agents[i]

        state = game.reset()
        while state.phase == Phase.BIDDING:
            p = state.current_player
            act = active_agents[p].act(state)
            state, r = game.step(state, act)
            if r is not None and state.phase == Phase.TERMINAL:
                break

        if state.phase == Phase.TERMINAL:
            records.append({'redeal': True})
            continue

        d = state.declarer
        trump = active_agents[d].declare_trump(state)
        vraagaas = active_agents[d].declare_vraagaas(state, trump) if state.contract in (Contract.RIK, Contract.RIK_BETER) else -1
        state = game.declare(state, trump, vraagaas)

        while state.phase == Phase.TRICK_TAKING:
            p = state.current_player
            act = active_agents[p].act(state)
            state, r = game.step(state, act)

        is_neural_decl = (d in team_a_seats)
        decl_won = (state.reward is not None and state.reward > 0)
        neural_won = (decl_won if is_neural_decl else not decl_won)

        records.append({
            'redeal': False,
            'contract': state.contract.name,
            'is_neural_decl': is_neural_decl,
            'decl_won': decl_won,
            'neural_won': neural_won,
            'tricks_won': state.tricks_won.copy(),
            'declarer_seat': d,
        })

    return records


def run_parallel_tournament(
    n_games: int = 5000,
    bvn_path: str = "checkpoints/bvn_best.pt",
    bn_path: str = "checkpoints/bn_best.pt",
    rollouts: int = 20,
    determinizations: int = 5,
    n_workers: int = None,
    seed: int = 42,
) -> Dict[str, Any]:
    if n_workers is None:
        n_workers = min(8, cpu_count())

    print(f"\n=================================================================")
    print(f"  PARALLEL TOURNAMENT: Neural Agent (0.50 Threshold) vs Baseline")
    print(f"  Games: {n_games:,} | Workers: {n_workers} | Rollouts: {rollouts} | Dets: {determinizations}")
    print(f"=================================================================")

    start_time = time.time()
    games_per_worker = n_games // n_workers
    remainder = n_games % n_workers

    tasks = []
    for w in range(n_workers):
        count = games_per_worker + (1 if w < remainder else 0)
        tasks.append((w, count, bvn_path, bn_path, rollouts, determinizations, seed + w * 1000))

    with Pool(n_workers) as pool:
        all_results = pool.map(_worker_task, tasks)

    flat_records = [r for worker_records in all_results for r in worker_records]
    elapsed = time.time() - start_time

    # Aggregate metrics
    valid = [r for r in flat_records if not r['redeal']]
    total_valid = len(valid)
    redeals = len(flat_records) - total_valid

    neural_wins = sum(1 for r in valid if r['neural_won'])
    neural_decl_games = [r for r in valid if r['is_neural_decl']]
    neural_def_games = [r for r in valid if not r['is_neural_decl']]

    decl_wins = sum(1 for r in neural_decl_games if r['decl_won'])
    def_wins = sum(1 for r in neural_def_games if not r['decl_won'])

    overall_wr = (neural_wins / max(1, total_valid)) * 100
    decl_wr = (decl_wins / max(1, len(neural_decl_games))) * 100 if neural_decl_games else 0
    def_wr = (def_wins / max(1, len(neural_def_games))) * 100 if neural_def_games else 0

    print(f"\n-----------------------------------------------------------------")
    print(f"  TOURNAMENT FINAL RESULTS ({n_games:,} games, {elapsed:.1f}s | {n_games/elapsed:.1f} games/s)")
    print(f"-----------------------------------------------------------------")
    print(f"  Total Valid Matches:               {total_valid:,} (Redeals: {redeals})")
    print(f"  Overall Match Win Rate (Neural):   {overall_wr:.1f}%  ({neural_wins:,}/{total_valid:,})")
    print(f"  When Neural Declares (Offense):    {decl_wr:.1f}%  ({decl_wins:,}/{len(neural_decl_games):,})")
    print(f"  When Neural Defends (Defense):     {def_wr:.1f}%  ({def_wins:,}/{len(neural_def_games):,})")
    print(f"-----------------------------------------------------------------")

    # Contract Breakdown
    contract_stats = {}
    for r in valid:
        c = r['contract']
        if c not in contract_stats:
            contract_stats[c] = {'neural_bids': 0, 'neural_wins': 0, 'opp_bids': 0, 'opp_wins': 0}
        if r['is_neural_decl']:
            contract_stats[c]['neural_bids'] += 1
            if r['decl_won']:
                contract_stats[c]['neural_wins'] += 1
        else:
            contract_stats[c]['opp_bids'] += 1
            if r['decl_won']:
                contract_stats[c]['opp_wins'] += 1

    print("\n  CONTRACT BREAKDOWN:")
    print(f"  {'Contract':<16} | {'Neural Bids':<12} | {'Neural Win %':<14} | {'Opponent Bids':<14} | {'Opponent Win %':<14}")
    print(f"  {'-'*16}-+-{'-'*12}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")
    for c, stats in sorted(contract_stats.items(), key=lambda x: x[1]['neural_bids'], reverse=True):
        n_b = stats['neural_bids']
        n_wr = f"{(stats['neural_wins']/n_b)*100:.1f}%" if n_b > 0 else "—"
        o_b = stats['opp_bids']
        o_wr = f"{(stats['opp_wins']/o_b)*100:.1f}%" if o_b > 0 else "—"
        print(f"  {c:<16} | {n_b:<12} | {n_wr:<14} | {o_b:<14} | {o_wr:<14}")
    print("=" * 65)

    return {
        'total_games': total_valid,
        'overall_win_rate': overall_wr / 100,
        'declarer_win_rate': decl_wr / 100,
        'defender_win_rate': def_wr / 100,
        'contract_stats': contract_stats,
    }


if __name__ == "__main__":
    run_parallel_tournament(n_games=5000)
