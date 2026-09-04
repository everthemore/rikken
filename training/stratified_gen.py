"""
training/stratified_gen.py — Stratified Contract Valuation Data Generator.

Systematically forces every contract type to be played across random deals to
map out the counterfactual Expected Value function Q(hand, contract).
Guarantees balanced representation across all 15 Rikken contracts.
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
from engine.card import ACE_RANK, HEARTS_SUIT, card_id
from agents.neural_agent import NeuralAgent
from agents.heuristic import HeuristicAgent
from training.data_gen import _pack_shard
import config

log = logging.getLogger(__name__)


def setup_forced_declaration(
    state: RikkenState,
    declarer: int,
    target_contract: Contract,
    agent: NeuralAgent | HeuristicAgent,
    game: RikkenGame,
) -> RikkenState:
    """Configures trump and partner for a forced contract declaration."""
    s = state.copy()
    s.contract = target_contract
    s.declarer = declarer
    s.declarer_mask[declarer] = True

    hand = s.hands[declarer]

    # Special partner contracts
    if target_contract == Contract.TROELA:
        from engine.rules import find_troela_partner
        s.partner = find_troela_partner(s, declarer)
        s.trump_suit = HEARTS_SUIT  # Hearts is traditional trump for Troela or longest suit
    elif target_contract == Contract.MOELA:
        from engine.rules import find_moela_partner
        s.partner = find_moela_partner(s, declarer)
        s.trump_suit = HEARTS_SUIT
    elif target_contract == Contract.RIK_BETER:
        s.partner = -1
        s.trump_suit = HEARTS_SUIT
    elif Contract.is_trump_contract(target_contract):
        # Pick longest suit as trump
        suit_lengths = [int(np.sum(hand[suit*13:(suit+1)*13])) for suit in range(4)]
        s.trump_suit = int(np.argmax(suit_lengths))
        s.partner = -1
    else:
        # No-trump contracts (Misere, Piek, Open Misere, Open Piek)
        s.trump_suit = -1
        s.partner = -1

    # Declare vraagaas for Rik & Rik Beter
    if target_contract in (Contract.RIK, Contract.RIK_BETER):
        vraagaas_suit = -1
        for suit in range(4):
            if suit != s.trump_suit and hand[suit * 13 + ACE_RANK] == 0:
                vraagaas_suit = suit
                break
        if vraagaas_suit != -1:
            for p in range(4):
                if p != declarer and s.hands[p][vraagaas_suit * 13 + ACE_RANK] == 1:
                    s.partner = p
                    break
        s.vraagaas_suit = vraagaas_suit

    s.phase = Phase.TRICK_TAKING
    s.current_player = declarer
    s.trick_leader = declarer
    return s


def run_stratified_game(
    target_contract: Contract,
    agents: List[NeuralAgent | HeuristicAgent],
    game: RikkenGame,
    rng: np.random.Generator,
) -> dict:
    """Run one game where declarer is forced to play target_contract."""
    state = game.reset()
    bid_records = []
    play_records = []

    # Select the player whose dealt hand is most suited for target_contract
    if target_contract == Contract.TROELA:
        for _ in range(50):
            ace_counts = [int(np.sum(state.hands[p][12::13])) for p in range(4)]
            if max(ace_counts) >= 3:
                declarer = int(np.argmax(ace_counts))
                break
            state = game.reset()
        else:
            declarer = 0
    elif target_contract == Contract.MOELA:
        for _ in range(200):
            moela_candidates = [p for p in range(4) if int(np.sum(state.hands[p][12::13])) == 4 and int(np.sum(state.hands[p][11::13])) >= 3]
            if moela_candidates:
                declarer = moela_candidates[0]
                break
            state = game.reset()
        else:
            declarer = 0
    elif Contract.is_trump_contract(target_contract):
        # Pick player with strongest/longest trump suit
        best_p, best_len = 0, 0
        for p in range(4):
            max_suit = max(int(np.sum(state.hands[p][s*13:(s+1)*13])) for s in range(4))
            if max_suit > best_len:
                best_len = max_suit
                best_p = p
        declarer = best_p
    elif target_contract in (Contract.MISERE, Contract.OPEN_MISERE):
        # Pick player with lowest cards / fewest honors
        honor_counts = [int(np.sum(state.hands[p][9::13]) + np.sum(state.hands[p][10::13]) + np.sum(state.hands[p][11::13]) + np.sum(state.hands[p][12::13])) for p in range(4)]
        declarer = int(np.argmin(honor_counts))
    elif target_contract in (Contract.PIEK, Contract.OPEN_PIEK):
        # Pick player with most high honors
        honor_counts = [int(np.sum(state.hands[p][9::13]) + np.sum(state.hands[p][10::13]) + np.sum(state.hands[p][11::13]) + np.sum(state.hands[p][12::13])) for p in range(4)]
        declarer = int(np.argmax(honor_counts))
    else:
        declarer = int(rng.integers(0, 4))

    # Record bidding action
    prev_bid_history = np.zeros((4, NUM_CONTRACTS), dtype=np.int8)
    prev_bid_history[declarer, int(target_contract)] = 1

    bid_records.append({
        'player': declarer,
        'hand': state.hands[declarer].copy(),
        'bid_history': np.zeros((4, NUM_CONTRACTS), dtype=np.int8),
        'bid_taken': int(target_contract),
    })

    # Other players pass
    for p in range(4):
        if p != declarer:
            bid_records.append({
                'player': p,
                'hand': state.hands[p].copy(),
                'bid_history': prev_bid_history.copy(),
                'bid_taken': int(Contract.PAS),
            })

    # Set up declaration
    state = setup_forced_declaration(state, declarer, target_contract, agents[declarer], game)

    # Trick-taking phase
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

    # Assign terminal rewards
    for rec in bid_records:
        p = rec['player']
        reward_p = float(game.get_reward(state, p))
        rec['outcome'] = reward_p
        rec['won'] = 1.0 if reward_p > 0 else 0.0

    declarer_won = (state.reward > 0) if state.reward is not None else False
    return {
        'bid_records': bid_records,
        'play_records': play_records,
        'contract': int(target_contract),
        'declarer': declarer,
        'declarer_won': declarer_won,
    }


def generate_stratified_dataset(
    contracts_per_batch: Optional[List[int]] = None,
    games_per_contract: int = 100,
    worker_id: int = 0,
    output_dir: str = 'data/stratified',
    rollouts: int = 50,
    determinizations: int = 5,
    seed: Optional[int] = None,
) -> None:
    """Generates a balanced stratified dataset across specified contracts."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed or (config.NUMPY_SEED + worker_id * 1000))
    game = RikkenGame(rng=rng)

    target_contracts = contracts_per_batch or [
        int(Contract.RIK),
        int(Contract.RIK_BETER),
        int(Contract.ACHT_ALLEEN),
        int(Contract.NEGEN_ALLEEN),
        int(Contract.TIEN_ALLEEN),
        int(Contract.ELF_ALLEEN),
        int(Contract.TWAALF_ALLEEN),
        int(Contract.DERTIEN_ALLEEN),
        int(Contract.MISERE),
        int(Contract.PIEK),
        int(Contract.OPEN_MISERE),
        int(Contract.OPEN_PIEK),
        int(Contract.TROELA),
        int(Contract.MOELA),
    ]

    agents = [
        NeuralAgent(
            seat=p,
            game=game,
            n_determinizations=determinizations,
            n_rollouts=rollouts,
            device="cpu",
            rng=rng,
        )
        for p in range(4)
    ]

    all_records = []
    t0 = time.time()
    total_games = len(target_contracts) * games_per_contract

    print(f"[Worker {worker_id:03d}] Generating {total_games} stratified games ({games_per_contract} per contract)...")

    for c_id in target_contracts:
        c = Contract(c_id)
        wins = 0
        for _ in range(games_per_contract):
            res = run_stratified_game(c, agents, game, rng)
            for rec in res['bid_records']:
                rec['type'] = 'bid'
                all_records.append(rec)
            for rec in res['play_records']:
                rec['type'] = 'play'
                all_records.append(rec)
            if res['declarer_won']:
                wins += 1

        wr = (wins / games_per_contract) * 100
        print(f"  Contract {c.name:<16} | Played: {games_per_contract:4d} | Decl Win: {wr:5.1f}%")

    shard_path = os.path.join(output_dir, f"stratified_shard_{worker_id:04d}.npz")
    _pack_shard(all_records, shard_path)
    dt = time.time() - t0
    print(f"[Worker {worker_id:03d}] Completed in {dt:.1f}s -> Saved {shard_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stratified Contract Data Generator")
    parser.add_argument('--games-per-contract', type=int, default=50)
    parser.add_argument('--worker-id', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='data/stratified')
    parser.add_argument('--rollouts', type=int, default=50)
    parser.add_argument('--determinizations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    generate_stratified_dataset(
        games_per_contract=args.games_per_contract,
        worker_id=args.worker_id,
        output_dir=args.output_dir,
        rollouts=args.rollouts,
        determinizations=args.determinizations,
        seed=args.seed,
    )
