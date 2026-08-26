"""
analysis/extract_tactical_rules.py — Tactical Trick-Taking Rule Extraction.
Simulates matches to measure the exact win-rate differentials of specific tactical choices.
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from engine.game import RikkenGame
from engine.state import Contract, Phase
from engine.card import rank_of, suit_of, cards_in_suit, SUIT_MASKS
from agents.heuristic import HeuristicAgent
from agents.neural_agent import NeuralAgent

def evaluate_tactical_rules(n_games: int = 500):
    print(f"Simulating {n_games} tactical games to extract in-game tactical rules...")
    game = RikkenGame(rng=np.random.default_rng(42))
    
    # Track tactical events
    stats = {
        'trump_contract_games': 0,
        'declarer_drew_trump_early_wins': 0,
        'declarer_drew_trump_early_total': 0,
        'declarer_did_not_draw_trump_wins': 0,
        'declarer_did_not_draw_trump_total': 0,
        'defender_led_through_declarer_wins': 0,
        'defender_led_through_declarer_total': 0,
        'defender_led_into_declarer_wins': 0,
        'defender_led_into_declarer_total': 0,
    }

    # Simulate and record tactics
    for g in range(n_games):
        state = game.reset()
        agents = [HeuristicAgent(seat=i) for i in range(4)]
        
        while state.phase == Phase.BIDDING:
            state, r = game.step(state, agents[state.current_player].act(state))
            if r is not None and state.phase == Phase.TERMINAL:
                break
        
        if state.phase != Phase.TRICK_TAKING:
            continue
            
        d = state.declarer
        trump = agents[d].declare_trump(state)
        vraagaas = agents[d].declare_vraagaas(state, trump) if state.contract == Contract.RIK else -1
        state = game.declare(state, trump, vraagaas)
        
        is_trump_contract = Contract.is_trump_contract(state.contract)
        if is_trump_contract:
            stats['trump_contract_games'] += 1
            
        trick_1_lead_suit = -1
        drew_trump_t1 = False
        
        while state.phase == Phase.TRICK_TAKING:
            p = state.current_player
            act = agents[p].act(state)
            
            # Record trick 1 lead
            if state.trick_count == 0 and len(state.current_trick) == 0:
                if is_trump_contract and p == d:
                    if suit_of(act) == trump:
                        drew_trump_t1 = True
            
            state, r = game.step(state, act)
            if r is not None:
                break
                
        declarer_won = (state.reward > 0)
        if is_trump_contract:
            if drew_trump_t1:
                stats['declarer_drew_trump_early_total'] += 1
                if declarer_won:
                    stats['declarer_drew_trump_early_wins'] += 1
            else:
                stats['declarer_did_not_draw_trump_total'] += 1
                if declarer_won:
                    stats['declarer_did_not_draw_trump_wins'] += 1

    print("\n" + "=" * 78)
    print("  EXPLAINABLE AI (XAI): EXTRACTED TACTICAL RULES")
    print("=" * 78)
    
    if stats['declarer_drew_trump_early_total'] > 0:
        t_draw_wr = stats['declarer_drew_trump_early_wins'] / stats['declarer_drew_trump_early_total']
        print(f"\n1. Declarer Trump-Drawing Tactic:")
        print(f"   - Leading Trump immediately on Trick 1: Win Rate = {t_draw_wr:.1%} ({stats['declarer_drew_trump_early_wins']}/{stats['declarer_drew_trump_early_total']})")
    
    if stats['declarer_did_not_draw_trump_total'] > 0:
        no_t_draw_wr = stats['declarer_did_not_draw_trump_wins'] / stats['declarer_did_not_draw_trump_total']
        print(f"   - Side-suit development before drawing trumps: Win Rate = {no_t_draw_wr:.1%} ({stats['declarer_did_not_draw_trump_wins']}/{stats['declarer_did_not_draw_trump_total']})")

    return stats

if __name__ == '__main__':
    evaluate_tactical_rules()
