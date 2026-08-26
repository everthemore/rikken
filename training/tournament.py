"""
training/tournament.py — Benchmark evaluator: Neural Agent vs Heuristic Baseline.

Runs structured evaluation tournaments rotating seats every game to ensure
zero positional bias. Logs detailed per-contract win rates, trick differentials,
and match histories.
"""

from __future__ import annotations
import os
import sys
import time
import json
import logging
import argparse
import numpy as np
from typing import Optional, List, Dict

import config
from engine.game import RikkenGame
from engine.state import RikkenState, Contract, Phase
from engine.card import card_to_str
from agents.heuristic import HeuristicAgent
from agents.neural_agent import NeuralAgent

log = logging.getLogger(__name__)


def play_tournament_match(
    neural_agents: List[NeuralAgent],
    heuristic_agents: List[HeuristicAgent],
    game: RikkenGame,
    rng: np.random.Generator,
    team_a_seats: tuple[int, int] = (0, 2),
) -> dict:
    """Play one tournament match between Neural (Team A) and Heuristic (Team B)."""
    active_agents = [None] * 4

    for i, p in enumerate(team_a_seats):
        agent = neural_agents[i]
        agent.set_seat(p)
        active_agents[p] = agent

    team_b_seats = tuple(s for s in range(4) if s not in team_a_seats)
    for i, p in enumerate(team_b_seats):
        agent = heuristic_agents[i]
        agent.set_seat(p)
        active_agents[p] = agent

    state = game.reset()

    # Bidding
    while state.phase == Phase.BIDDING:
        p = state.current_player
        action = active_agents[p].act(state)
        state, reward = game.step(state, action)
        if reward is not None and state.phase == Phase.TERMINAL:
            return {'contract': 0, 'declarer': -1, 'winner': -1, 'redeal': True, 'tricks_won': np.zeros(4)}

    # Declaration
    if state.phase == Phase.TRICK_TAKING and state.trick_count == 0:
        d = state.declarer
        if Contract.is_trump_contract(state.contract):
            trump = active_agents[d].declare_trump(state)
            vraagaas = -1
            if state.contract in (Contract.RIK, Contract.RIK_BETER):
                vraagaas = active_agents[d].declare_vraagaas(state, trump)
            state = game.declare(state, trump_suit=trump, vraagaas_suit=vraagaas)

    # Trick-taking
    while state.phase == Phase.TRICK_TAKING:
        p = state.current_player
        action = active_agents[p].act(state)
        state, reward = game.step(state, action)
        if reward is not None:
            break

    declarer = state.declarer
    declarer_is_team_a = (declarer in team_a_seats)
    declarer_won = (state.reward > 0)

    if declarer_is_team_a:
        team_a_won = declarer_won
    else:
        team_a_won = not declarer_won

    return {
        'contract': int(state.contract),
        'declarer': declarer,
        'declarer_is_team_a': declarer_is_team_a,
        'declarer_won': declarer_won,
        'team_a_won': team_a_won,
        'tricks_won': state.tricks_won.copy(),
        'declarer_tricks': state.declarer_tricks,
        'redeal': False,
    }


def run_tournament(
    n_games: int = 100,
    bvn_path: str = 'checkpoints/bvn_best.pt',
    bn_path: str = 'checkpoints/bn_best.pt',
    rollouts: int = 50,
    determinizations: int = 10,
    device: str = config.DEVICE,
    seed: int = 42,
) -> dict:
    """Run evaluation tournament with comprehensive per-contract & trick metrics."""
    rng = np.random.default_rng(seed)
    game = RikkenGame(rng=rng)

    neural_agents = [
        NeuralAgent(seat=p, game=game, bvn=bvn_path, bn=bn_path,
                    n_determinizations=determinizations, n_rollouts=rollouts,
                    device=device, rng=rng)
        for p in [0, 2]
    ]

    heuristic_agents = [
        HeuristicAgent(seat=p, rng=rng)
        for p in [1, 3]
    ]

    team_a_wins = 0
    total_valid_games = 0
    redeals = 0

    neural_declared_count = 0
    neural_declared_wins = 0
    neural_declared_tricks = []

    neural_defended_count = 0
    neural_defended_wins = 0
    heuristic_declared_tricks = []

    # Per-contract granular tracking
    contract_breakdown = {}
    for c in Contract:
        if c > Contract.PAS:
            contract_breakdown[c.name] = {
                'total_bids': 0,
                'neural_bids': 0,
                'neural_wins': 0,
                'heuristic_bids': 0,
                'heuristic_wins': 0,
            }

    print(f"
{'='*65}")
    print(f"  TOURNAMENT: Neural Agent (BVN+BN+ISMCTS) vs Heuristic Baseline")
    print(f"  Games: {n_games} | Rollouts: {rollouts} | Determinizations: {determinizations}")
    print(f"{'='*65}")

    t0 = time.time()

    for g in range(n_games):
        team_a_seats = (0, 2) if (g % 2 == 0) else (1, 3)
        res = play_tournament_match(neural_agents, heuristic_agents, game, rng, team_a_seats)

        if res['redeal']:
            redeals += 1
            continue

        total_valid_games += 1
        if res['team_a_won']:
            team_a_wins += 1

        cname = Contract(res['contract']).name
        if cname in contract_breakdown:
            cb = contract_breakdown[cname]
            cb['total_bids'] += 1
            if res['declarer_is_team_a']:
                cb['neural_bids'] += 1
                if res['declarer_won']:
                    cb['neural_wins'] += 1
            else:
                cb['heuristic_bids'] += 1
                if res['declarer_won']:
                    cb['heuristic_wins'] += 1

        if res['declarer_is_team_a']:
            neural_declared_count += 1
            neural_declared_tricks.append(res['declarer_tricks'])
            if res['declarer_won']:
                neural_declared_wins += 1
        else:
            neural_defended_count += 1
            heuristic_declared_tricks.append(res['declarer_tricks'])
            if not res['declarer_won']:
                neural_defended_wins += 1

        if (g + 1) % max(1, n_games // 10) == 0 or (g + 1) == n_games:
            elapsed = time.time() - t0
            win_rate = team_a_wins / max(total_valid_games, 1)
            print(f"  Game {g+1:4d}/{n_games} | Neural Win Rate: {win_rate:6.1%} | {total_valid_games/elapsed:4.1f} games/s")

    elapsed = time.time() - t0
    win_rate = team_a_wins / max(total_valid_games, 1)
    decl_wr = neural_declared_wins / max(neural_declared_count, 1)
    def_wr  = neural_defended_wins / max(neural_defended_count, 1)

    # Compute win rate percentages per contract
    contracts_summary = {}
    for cname, cb in contract_breakdown.items():
        if cb['total_bids'] > 0:
            n_wr = (cb['neural_wins'] / cb['neural_bids']) if cb['neural_bids'] > 0 else None
            h_wr = (cb['heuristic_wins'] / cb['heuristic_bids']) if cb['heuristic_bids'] > 0 else None
            contracts_summary[cname] = {
                'total_bids': cb['total_bids'],
                'neural_bids': cb['neural_bids'],
                'neural_win_rate': round(n_wr, 3) if n_wr is not None else None,
                'heuristic_bids': cb['heuristic_bids'],
                'heuristic_win_rate': round(h_wr, 3) if h_wr is not None else None,
            }

    print(f"
{'-'*65}")
    print(f"  TOURNAMENT FINAL RESULTS ({total_valid_games} games, {elapsed:.1f}s)")
    print(f"{'-'*65}")
    print(f"  Overall Match Win Rate (Neural):  {win_rate:6.1%}  ({team_a_wins}/{total_valid_games})")
    print(f"  When Neural Declares (Offense):   {decl_wr:6.1%}  ({neural_declared_wins}/{neural_declared_count})")
    print(f"  When Neural Defends (Defense):    {def_wr:6.1%}  ({neural_defended_wins}/{neural_defended_count})")
    print(f"{'-'*65}")

    metrics = {
        'total_games': total_valid_games,
        'redeals': redeals,
        'neural_win_rate': round(win_rate, 4),
        'declarer_win_rate': round(decl_wr, 4),
        'defender_win_rate': round(def_wr, 4),
        'neural_declared_count': neural_declared_count,
        'neural_defended_count': neural_defended_count,
        'neural_avg_tricks_declared': round(float(np.mean(neural_declared_tricks)), 2) if neural_declared_tricks else None,
        'heuristic_avg_tricks_declared': round(float(np.mean(heuristic_declared_tricks)), 2) if heuristic_declared_tricks else None,
        'contract_breakdown': contracts_summary,
        'elapsed_seconds': round(elapsed, 1),
    }
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tournament Evaluator')
    parser.add_argument('--games',            type=int, default=50)
    parser.add_argument('--rollouts',         type=int, default=50)
    parser.add_argument('--determinizations', type=int, default=10)
    parser.add_argument('--bvn',              type=str, default='checkpoints/bvn_best.pt')
    parser.add_argument('--bn',               type=str, default='checkpoints/bn_best.pt')
    parser.add_argument('--device',           type=str, default=config.DEVICE)
    parser.add_argument('--seed',             type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    run_tournament(
        n_games=args.games,
        bvn_path=args.bvn,
        bn_path=args.bn,
        rollouts=args.rollouts,
        determinizations=args.determinizations,
        device=args.device,
        seed=args.seed,
    )
