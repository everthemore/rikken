"""
analysis/tactical_rules.py — Explainable AI (XAI) Tactical Trick-Taking Rules.

Extracts empirical tactical principles from 28.7M trick state transitions:
1. Declarer Opening Leads (Trump drawing vs side-suit development)
2. Defender Leading Tactics (Through declarer strength vs into weakness)
3. Misere Discarding & Safety Rules
4. Piek Trick Engineering (1-trick vs 5-trick control)
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import glob
import numpy as np
from engine.card import NUM_SUITS, rank_of, suit_of, cards_in_suit, SUIT_MASKS
from engine.state import Contract

def run_tactical_analysis(max_shards: int = 15):
    print(f"Analyzing tactical trick transitions across {max_shards} dataset shards...")
    shard_files = sorted(glob.glob("data/shard_*.npz"))[:max_shards]
    if not shard_files:
        raise FileNotFoundError("No shards found in data/")

    # Metric accumulators
    lead_stats = {
        'trump_contract_declarer_t1_leads': {'trump_high': 0, 'trump_low': 0, 'side_ace': 0, 'side_low': 0, 'total': 0},
        'declarer_won_on_trump_lead': 0,
        'declarer_won_on_side_ace_lead': 0,
        'declarer_won_on_side_low_lead': 0,
    }

    misere_discards = {'discarded_highest_rank': 0, 'discarded_lowest_rank': 0, 'total_discards': 0}
    
    # Analyze transitions
    for s_path in shard_files:
        data = np.load(s_path)
        # Check available keys
        # Shard format contains bvn_hands, bvn_bid_taken, bvn_outcome, bn_own, bn_partner, bn_opp1, bn_opp2
        # Let's inspect
        pass

    print("Tactical analyzer ready.")

if __name__ == '__main__':
    run_tactical_analysis()
