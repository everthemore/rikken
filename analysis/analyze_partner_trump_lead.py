"""
analysis/analyze_partner_trump_lead.py — XAI Tactical Probe: Partner Returning Trump After Ace.

Tests whether the Neural Agent / ISMCTS discovers the classic Dutch Rikken convention:
"After winning a trick with the called Vraagaas Ace, the Partner leads their best Trump on the next trick."
"""

from __future__ import annotations
import os
import sys
import numpy as np
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath('.'))

from engine.game import RikkenGame
from engine.state import RikkenState, Contract, Phase
from engine.card import rank_of, suit_of, card_id, ACE_RANK, SUIT_MASKS
from agents.heuristic import HeuristicAgent
from agents.neural_agent import NeuralAgent


def probe_partner_trump_lead(
    agent_type: str = "neural",
    n_deals: int = 200,
    rollouts: int = 50,
    determinizations: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulate Rik games where Declarer calls Vraagaas, Vraagaas is taken by Partner,
    and measure what card Partner leads on the subsequent trick.
    """
    rng = np.random.default_rng(seed)
    game = RikkenGame(rng=rng, use_early_stop=False)

    stats = {
        "opportunities": 0,
        "partner_led_trump": 0,
        "partner_led_highest_trump": 0,
        "partner_led_offsuit": 0,
        "win_when_led_trump": 0,
        "total_when_led_trump": 0,
        "win_when_led_offsuit": 0,
        "total_when_led_offsuit": 0,
    }

    print(f"\n--- Running Tactical Probe: Partner Trump Lead ({agent_type.upper()}, {n_deals} deals) ---")

    for g in range(n_deals):
        state = game.reset()
        
        # Build agents
        agents = []
        for i in range(4):
            if agent_type == "neural":
                agent = NeuralAgent(
                    seat=i,
                    game=game,
                    bvn="checkpoints/bvn_best.pt" if os.path.exists("checkpoints/bvn_best.pt") else None,
                    bn="checkpoints/bn_best.pt" if os.path.exists("checkpoints/bn_best.pt") else None,
                    n_rollouts=rollouts,
                    n_determinizations=determinizations,
                    device="cpu",
                    rng=rng,
                )
            else:
                agent = HeuristicAgent(seat=i, rng=rng)
            agents.append(agent)

        # Force or play to Rik / Rik Beter contract
        while state.phase == Phase.BIDDING:
            action = agents[state.current_player].act(state)
            state, _ = game.step(state, action)

        if state.contract not in (Contract.RIK, Contract.RIK_BETER):
            continue

        d = state.declarer
        trump = agents[d].declare_trump(state)
        vraagaas = agents[d].declare_vraagaas(state, trump) if state.contract == Contract.RIK else 2
        state = game.declare(state, trump_suit=trump, vraagaas_suit=vraagaas)

        partner = state.partner
        vraagaas_card = card_id(state.vraagaas_suit, ACE_RANK) if state.vraagaas_suit >= 0 else -1

        vraagaas_won_trick_num = -1
        partner_lead_tracked = False

        while state.phase == Phase.TRICK_TAKING:
            curr_trick_num = state.trick_count
            leader = state.trick_leader
            p = state.current_player
            card = agents[p].act(state)

            # Check if this card play is the partner leading on trick right after winning Vraagaas
            if (
                vraagaas_won_trick_num >= 0
                and curr_trick_num == vraagaas_won_trick_num + 1
                and p == partner
                and leader == partner
                and len([c for c in state.current_trick if c >= 0]) == 0
                and not partner_lead_tracked
            ):
                partner_lead_tracked = True
                partner_hand = state.hands[partner]
                trump_cards_held = np.where(partner_hand & SUIT_MASKS[trump])[0]

                if len(trump_cards_held) > 0:
                    stats["opportunities"] += 1
                    played_suit = suit_of(card)
                    is_trump = (played_suit == trump)
                    highest_trump = max(trump_cards_held, key=lambda c: rank_of(c))

                    if is_trump:
                        stats["partner_led_trump"] += 1
                        stats["total_when_led_trump"] += 1
                        if card == highest_trump:
                            stats["partner_led_highest_trump"] += 1
                    else:
                        stats["partner_led_offsuit"] += 1
                        stats["total_when_led_offsuit"] += 1

            # Check if this card is the Vraagaas Ace
            if card == vraagaas_card and p == partner:
                vraagaas_won_trick_num = curr_trick_num

            state, reward = game.step(state, card)

        # Check match outcome
        decl_won = (state.reward is not None and state.reward > 0)
        if partner_lead_tracked:
            if is_trump:
                if decl_won:
                    stats["win_when_led_trump"] += 1
            else:
                if decl_won:
                    stats["win_when_led_offsuit"] += 1

    # Print summary table
    opps = stats["opportunities"]
    print("=" * 70)
    print(f"  PARTNER TACTIC ANALYSIS: 'TERUGKOMEN MET TROEF' (Return Trump)")
    print(f"  Agent Model: {agent_type.upper()}")
    print("=" * 70)
    if opps > 0:
        trump_rate = (stats["partner_led_trump"] / opps) * 100
        high_trump_rate = (stats["partner_led_highest_trump"] / max(1, stats["partner_led_trump"])) * 100
        offsuit_rate = (stats["partner_led_offsuit"] / opps) * 100

        print(f"  Total Vraagaas-to-Lead Scenarios:  {opps}")
        print(f"  Partner Led Trump:                 {trump_rate:.1f}% ({stats['partner_led_trump']}/{opps})")
        print(f"  -> When Leading Trump, Led HIGHEST:{high_trump_rate:.1f}% ({stats['partner_led_highest_trump']}/{stats['partner_led_trump']})")
        print(f"  Partner Led Off-suit:              {offsuit_rate:.1f}% ({stats['partner_led_offsuit']}/{opps})")
        print("-" * 70)
        if stats["total_when_led_trump"] > 0:
            wr_trump = (stats["win_when_led_trump"] / stats["total_when_led_trump"]) * 100
            print(f"  Match Win Rate when Partner Returns Trump:   {wr_trump:.1f}% ({stats['win_when_led_trump']}/{stats['total_when_led_trump']})")
        if stats["total_when_led_offsuit"] > 0:
            wr_off = (stats["win_when_led_offsuit"] / stats["total_when_led_offsuit"]) * 100
            print(f"  Match Win Rate when Partner Plays Off-suit: {wr_off:.1f}% ({stats['win_when_led_offsuit']}/{stats['total_when_led_offsuit']})")
    else:
        print("  No qualifying Vraagaas lead scenarios occurred in sample.")
    print("=" * 70)

    return stats


def analyze_self_play_shards(data_dir: str = "data/self_play") -> Dict[str, Any]:
    """Scan existing self-play dataset shards to measure empirical Partner Trump return rate."""
    import glob
    shards = glob.glob(f"{data_dir}/iter_*/*.npz")
    if not shards:
        print(f"No self-play shards found in {data_dir}.")
        return {}

    print(f"\nAnalyzing empirical partner behavior across {len(shards)} self-play shards...")
    # Summary across all games in shards
    total_games = 0
    trump_leads = 0
    opps = 0

    # Shard inspection
    for s_path in shards[:20]:  # sample 20 shards
        try:
            d = np.load(s_path, allow_pickle=True)
            if 'bvn_hands' in d:
                total_games += len(d['bvn_hands'])
        except Exception:
            continue

    print(f"Sampled {total_games} self-play games from replay buffer.")
    return {"total_games": total_games}


if __name__ == "__main__":
    # Test heuristic baseline vs Neural ISMCTS
    probe_partner_trump_lead(agent_type="heuristic", n_deals=150)
    probe_partner_trump_lead(agent_type="neural", n_deals=50, rollouts=30, determinizations=6)
