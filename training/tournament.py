"""
training/tournament.py — Benchmark Tournament between Agent Generations.

Pits NeuralAgents (or checkpoint models) against HeuristicAgents or previous
model checkpoints in head-to-head match play with seat rotation to eliminate
positional bias.

Metrics Tracked:
  - Overall Win Rate (Neural vs Opponent)
  - Declarer Win Rate (when declaring)
  - Defender Win Rate (when defending against opponent declarer)
  - Contracts bid and success rates
  - Average tricks taken per game

Usage:
    python -m training.tournament --games 100 --rollouts 50 --bvn checkpoints/bvn_best.pt --bn checkpoints/bn_best.pt
"""

from __future__ import annotations
import os
import argparse
import time
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

from engine.state import RikkenState, Contract, Phase
from engine.game import RikkenGame
from engine.card import NUM_SUITS, cards_in_suit, rank_of, card_id, ACE_RANK, NUM_RANKS
from agents.neural_agent import NeuralAgent
from agents.heuristic import HeuristicAgent
from agents.random_agent import RandomAgent
import config

log = logging.getLogger(__name__)


def play_tournament_match(
    team_a_agents: List[NeuralAgent | HeuristicAgent],
    team_b_agents: List[NeuralAgent | HeuristicAgent],
    game: RikkenGame,
    rng: np.random.Generator,
    team_a_seats: Tuple[int, int] = (0, 2), # Seats for Team A (Partners)
) -> dict:
    """
    Play one game where Team A occupies two seats (e.g. 0 & 2) and Team B occupies (1 & 3).
    """
    # Assemble 4 agents in seat order and synchronize seat indices
    active_agents = [None] * 4
    for p in range(4):
        if p in team_a_seats:
            agent = team_a_agents[0] if p == team_a_seats[0] else team_a_agents[1]
        else:
            other_seats = [s for s in range(4) if s not in team_a_seats]
            agent = team_b_agents[0] if p == other_seats[0] else team_b_agents[1]
        agent.seat = p
        if hasattr(agent, 'ismcts'):
            agent.ismcts.seat = p
        active_agents[p] = agent

    state = game.reset()

    # Bidding
    while state.phase == Phase.BIDDING:
        p = state.current_player
        action = active_agents[p].act(state)
        state, reward = game.step(state, action)
        if reward is not None and state.phase == Phase.TERMINAL:
            return {'contract': 0, 'declarer': -1, 'winner': -1, 'redeal': True}

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

    # Determine winning team
    declarer = state.declarer
    declarer_is_team_a = (declarer in team_a_seats)
    declarer_won = (state.reward > 0)

    # Reward for Team A
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
    """Run an evaluation tournament: Neural Team (Seats 0, 2) vs Heuristic Team (Seats 1, 3)."""
    rng = np.random.default_rng(seed)
    game = RikkenGame(rng=rng)

    # Team A: Neural Agents
    neural_agents = [
        NeuralAgent(seat=p, game=game, bvn=bvn_path, bn=bn_path,
                    n_determinizations=determinizations, n_rollouts=rollouts,
                    device=device, rng=rng)
        for p in [0, 2]
    ]

    # Team B: Heuristic Baseline Agents
    heuristic_agents = [
        HeuristicAgent(seat=p, rng=rng)
        for p in [1, 3]
    ]

    team_a_wins = 0
    total_valid_games = 0
    neural_declared_count = 0
    neural_declared_wins = 0
    neural_defended_count = 0
    neural_defended_wins = 0
    contract_stats = {}

    print(f"\n{'='*65}")
    print(f"  TOURNAMENT: Neural Agent (BVN+BN+ISMCTS) vs Heuristic Baseline")
    print(f"  Games: {n_games} | Rollouts: {rollouts} | Determinizations: {determinizations}")
    print(f"{'='*65}")

    t0 = time.time()

    for g in range(n_games):
        # Rotate seats every game to ensure zero positional bias
        team_a_seats = (0, 2) if (g % 2 == 0) else (1, 3)
        res = play_tournament_match(neural_agents, heuristic_agents, game, rng, team_a_seats)

        if res['redeal']:
            continue

        total_valid_games += 1
        if res['team_a_won']:
            team_a_wins += 1

        if res['declarer_is_team_a']:
            neural_declared_count += 1
            if res['declarer_won']:
                neural_declared_wins += 1
        else:
            neural_defended_count += 1
            if not res['declarer_won']:
                neural_defended_wins += 1

        cname = Contract(res['contract']).name
        contract_stats[cname] = contract_stats.get(cname, 0) + 1

        if (g + 1) % max(1, n_games // 10) == 0 or (g + 1) == n_games:
            elapsed = time.time() - t0
            win_rate = team_a_wins / max(total_valid_games, 1)
            print(f"  Game {g+1:4d}/{n_games} | Neural Win Rate: {win_rate:6.1%} | {total_valid_games/elapsed:4.1f} games/s")

    elapsed = time.time() - t0
    win_rate = team_a_wins / max(total_valid_games, 1)
    decl_wr = neural_declared_wins / max(neural_declared_count, 1)
    def_wr  = neural_defended_wins / max(neural_defended_count, 1)

    print(f"\n{'-'*65}")
    print(f"  TOURNAMENT FINAL RESULTS ({total_valid_games} games, {elapsed:.1f}s)")
    print(f"{'-'*65}")
    print(f"  Overall Match Win Rate (Neural):  {win_rate:6.1%}  ({team_a_wins}/{total_valid_games})")
    print(f"  When Neural Declares (Offense):   {decl_wr:6.1%}  ({neural_declared_wins}/{neural_declared_count})")
    print(f"  When Neural Defends (Defense):    {def_wr:6.1%}  ({neural_defended_wins}/{neural_defended_count})")
    print(f"{'-'*65}")

    metrics = {
        'total_games': total_valid_games,
        'neural_win_rate': win_rate,
        'declarer_win_rate': decl_wr,
        'defender_win_rate': def_wr,
        'contracts': contract_stats,
        'elapsed_seconds': elapsed,
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
