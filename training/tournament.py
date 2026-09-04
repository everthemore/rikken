"""
training/tournament.py — Benchmark evaluator: Individual 1-vs-3 Tournament.

Evaluates an agent on an individual seat basis (1 Evaluated Agent vs 3 Opponents),
rotating both seat position (0..3) and dealer position (0..3) across 16-game
symmetric blocks to guarantee zero positional bias.

Supports:
  1. 1 Neural Agent vs 3 Heuristic Agents (fixed baseline anchor).
  2. 1 Neural Agent vs 3 Previous Generation Neural Agents (direct AlphaZero policy gating).
"""

from __future__ import annotations
import os
import sys
import time
import json
import logging
import argparse
import numpy as np
from typing import Optional, List, Dict, Any

import config
from engine.game import RikkenGame
from engine.state import RikkenState, Contract, Phase
from agents.heuristic import HeuristicAgent
from agents.neural_agent import NeuralAgent

log = logging.getLogger(__name__)


def play_tournament_match(
    eval_agent: NeuralAgent,
    opp_agents: List[NeuralAgent | HeuristicAgent],
    game: RikkenGame,
    rng: np.random.Generator,
    eval_seat: int = 0,
    dealer: Optional[int] = None,
) -> dict:
    """
    Play one tournament match with eval_agent in eval_seat and opp_agents in the other 3 seats.
    """
    active_agents = [None] * 4
    eval_agent.set_seat(eval_seat)
    active_agents[eval_seat] = eval_agent

    opp_seats = [s for s in range(4) if s != eval_seat]
    for i, s in enumerate(opp_seats):
        opp_agents[i].set_seat(s)
        active_agents[s] = opp_agents[i]

    state = game.reset(dealer=dealer)

    # Bidding
    while state.phase == Phase.BIDDING:
        p = state.current_player
        action = active_agents[p].act(state)
        state, reward = game.step(state, action)
        if reward is not None and state.phase == Phase.TERMINAL:
            return {
                'contract': 0,
                'declarer': -1,
                'redeal': True,
                'eval_won': False,
                'eval_reward': 0.0,
                'tricks_won': np.zeros(4),
            }

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

    eval_reward = float(state.rewards[eval_seat])
    eval_won = (eval_reward > 0)

    is_declarer = bool(state.declarer == eval_seat or state.declarer_mask[eval_seat])
    is_partner = bool(state.partner == eval_seat)
    is_defender = not is_declarer and not is_partner

    return {
        'contract': int(state.contract),
        'declarer': state.declarer,
        'eval_seat': eval_seat,
        'is_declarer': is_declarer,
        'is_partner': is_partner,
        'is_defender': is_defender,
        'eval_won': eval_won,
        'eval_reward': eval_reward,
        'tricks_won': state.tricks_won.copy(),
        'eval_tricks': int(state.tricks_won[eval_seat]),
        'declarer_tricks': state.declarer_tricks,
        'redeal': False,
    }


def run_tournament(
    n_games: int = 400,
    bvn_path: str = 'checkpoints/bvn_best.pt',
    bn_path: str = 'checkpoints/bn_best.pt',
    opp_type: str = 'heuristic',
    opp_bvn_path: Optional[str] = None,
    opp_bn_path: Optional[str] = None,
    rollouts: int = 50,
    determinizations: int = 10,
    device: str = config.DEVICE,
    seed: int = 42,
) -> dict:
    """
    Run evaluation tournament with individual 1-vs-3 seat and dealer rotation.
    """
    rng = np.random.default_rng(seed)
    game = RikkenGame(rng=rng)

    # Primary evaluated agent
    eval_agent = NeuralAgent(
        seat=0,
        game=game,
        bvn=bvn_path,
        bn=bn_path,
        n_determinizations=determinizations,
        n_rollouts=rollouts,
        device=device,
        rng=rng,
        epsilon=0.0,
    )

    # 3 Opponent agents
    if opp_type == 'neural':
        opp_bvn = opp_bvn_path or bvn_path
        opp_bn = opp_bn_path or bn_path
        opp_agents = [
            NeuralAgent(
                seat=i + 1,
                game=game,
                bvn=opp_bvn,
                bn=opp_bn,
                n_determinizations=determinizations,
                n_rollouts=rollouts,
                device=device,
                rng=rng,
                epsilon=0.0,
            )
            for i in range(3)
        ]
        opp_label = f"Neural Opponents ({os.path.basename(str(opp_bvn))})"
    else:
        opp_agents = [HeuristicAgent(seat=i + 1, rng=rng) for i in range(3)]
        opp_label = "Heuristic Baseline"

    eval_wins = 0
    total_valid_games = 0
    redeals = 0

    decl_games, decl_wins = 0, 0
    part_games, part_wins = 0, 0
    def_games, def_wins = 0, 0

    eval_declared_tricks = []
    opp_declared_tricks = []

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

    print("\n" + "=" * 68)
    print(f"  TOURNAMENT: 1 Neural vs 3 {opp_label}")
    print(f"  Games: {n_games} | Rollouts: {rollouts} | Det: {determinizations}")
    print("=" * 68)

    t0 = time.time()

    valid_games = 0
    total_deals = 0
    max_deals = n_games * 4

    while valid_games < n_games and total_deals < max_deals:
        eval_seat = valid_games % 4
        dealer = (valid_games // 4) % 4
        total_deals += 1

        res = play_tournament_match(
            eval_agent=eval_agent,
            opp_agents=opp_agents,
            game=game,
            rng=rng,
            eval_seat=eval_seat,
            dealer=dealer,
        )

        if res['redeal']:
            redeals += 1
            continue

        valid_games += 1
        if res['eval_won']:
            eval_wins += 1

        cname = Contract(res['contract']).name
        if cname in contract_breakdown:
            cb = contract_breakdown[cname]
            cb['total_bids'] += 1
            if res['is_declarer']:
                cb['neural_bids'] += 1
                if res['eval_won']:
                    cb['neural_wins'] += 1
            else:
                cb['heuristic_bids'] += 1
                if not res['eval_won']:
                    cb['heuristic_wins'] += 1

        if res['is_declarer']:
            decl_games += 1
            eval_declared_tricks.append(res['declarer_tricks'])
            if res['eval_won']:
                decl_wins += 1
        elif res['is_partner']:
            part_games += 1
            if res['eval_won']:
                part_wins += 1
        else:
            def_games += 1
            opp_declared_tricks.append(res['declarer_tricks'])
            if res['eval_won']:
                def_wins += 1

        if valid_games % max(1, n_games // 10) == 0 or valid_games == n_games:
            elapsed = time.time() - t0
            current_wr = eval_wins / max(valid_games, 1)
            rate = valid_games / max(elapsed, 0.001)
            print(f"  Game {valid_games:4d}/{n_games} (Deals: {total_deals}) | Win Rate: {current_wr:6.1%} | {rate:4.1f} g/s")

    elapsed = time.time() - t0
    win_rate = eval_wins / max(valid_games, 1)
    decl_wr = (decl_wins / decl_games) if decl_games > 0 else 0.0
    part_wr = (part_wins / part_games) if part_games > 0 else 0.0
    def_wr = (def_wins / def_games) if def_games > 0 else 0.0

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

    print("\n" + "-" * 68)
    print(f"  TOURNAMENT FINAL RESULTS ({valid_games} games, {elapsed:.1f}s)")
    print("-" * 68)
    print(f"  Overall Win Rate:          {win_rate:6.1%}  ({eval_wins}/{valid_games})")
    print(f"  When Declarer (Solo/Lead): {decl_wr:6.1%}  ({decl_wins}/{decl_games})")
    print(f"  When Partner (Maatje):     {part_wr:6.1%}  ({part_wins}/{part_games})")
    print(f"  When Defender:             {def_wr:6.1%}  ({def_wins}/{def_games})")
    print("-" * 68)

    metrics = {
        'total_games': valid_games,
        'redeals': redeals,
        'neural_win_rate': round(win_rate, 4),
        'declarer_win_rate': round(decl_wr, 4),
        'partner_win_rate': round(part_wr, 4),
        'defender_win_rate': round(def_wr, 4),
        'neural_declared_count': decl_games,
        'neural_defended_count': def_games,
        'neural_partner_count': part_games,
        'neural_avg_tricks_declared': round(float(np.mean(eval_declared_tricks)), 2) if eval_declared_tricks else None,
        'heuristic_avg_tricks_declared': round(float(np.mean(opp_declared_tricks)), 2) if opp_declared_tricks else None,
        'contract_breakdown': contracts_summary,
        'opp_type': opp_type,
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
    parser.add_argument('--opp-type',         type=str, default='heuristic', choices=['heuristic', 'neural'])
    parser.add_argument('--opp-bvn',          type=str, default=None)
    parser.add_argument('--opp-bn',           type=str, default=None)
    parser.add_argument('--device',           type=str, default=config.DEVICE)
    parser.add_argument('--seed',             type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    run_tournament(
        n_games=args.games,
        bvn_path=args.bvn,
        bn_path=args.bn,
        opp_type=args.opp_type,
        opp_bvn_path=args.opp_bvn,
        opp_bn_path=args.opp_bn,
        rollouts=args.rollouts,
        determinizations=args.determinizations,
        device=args.device,
        seed=args.seed,
    )
