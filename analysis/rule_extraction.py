import sys, os
sys.path.insert(0, os.path.abspath('.'))
"""
analysis/rule_extraction.py — Explainable AI (XAI) Rule Extraction for Rikken.

Extracts human-interpretable bidding thresholds and tactical rules from the
dataset and BVN/BN predictions using decision trees and feature importance.
"""

import os
import glob
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from engine.card import NUM_SUITS, NUM_RANKS, rank_of, suit_of, cards_in_suit, SUIT_MASKS
from engine.state import Contract

CONTRACT_NAMES = [Contract(i).name for i in range(14)]

FEATURE_NAMES = [
    'HCP', 'Ace_Count', 'King_Count', 'Queen_Count', 'Jack_Count',
    'Low_Card_Count', 'Max_Suit_Len', 'Second_Suit_Len', 'Min_Suit_Len',
    'Void_Count', 'Singleton_Count', 'Doubleton_Count', 'Trump_HCP',
    'Has_AK_Longest', 'Has_AKQ_Longest', 'Stoppers_Count', 'Losing_Trick_Count'
]


def extract_features_from_hand(hand: np.ndarray) -> np.ndarray:
    """
    Extract a 17-dimensional vector of human-interpretable features from a 52-card one-hot hand.
    """
    card_indices = np.where(hand)[0]
    ranks = [rank_of(c) for c in card_indices]
    suits = [suit_of(c) for c in card_indices]

    # HCP: A=4, K=3, Q=2, J=1
    hcp = sum(max(0, r - 8) for r in ranks)
    ace_count = sum(1 for r in ranks if r == 12)
    king_count = sum(1 for r in ranks if r == 11)
    queen_count = sum(1 for r in ranks if r == 10)
    jack_count = sum(1 for r in ranks if r == 9)
    low_cards = sum(1 for r in ranks if r < 8) # 2..9

    # Suit lengths
    suit_lens = [np.sum(hand & SUIT_MASKS[s]) for s in range(NUM_SUITS)]
    sorted_lens = sorted(suit_lens, reverse=True)
    max_len = sorted_lens[0]
    second_len = sorted_lens[1]
    min_len = sorted_lens[3]

    voids = sum(1 for l in suit_lens if l == 0)
    singletons = sum(1 for l in suit_lens if l == 1)
    doubletons = sum(1 for l in suit_lens if l == 2)

    # Longest suit analysis
    longest_suit = int(np.argmax(suit_lens))
    longest_cards = hand & SUIT_MASKS[longest_suit]
    longest_ranks = [rank_of(c) for c in np.where(longest_cards)[0]]
    trump_hcp = sum(max(0, r - 8) for r in longest_ranks)

    has_ak = (12 in longest_ranks) and (11 in longest_ranks)
    has_akq = has_ak and (10 in longest_ranks)

    # Stoppers: Ace in suit, or King with >= 2 cards, or Queen with >= 3 cards
    stoppers = 0
    losers = 0
    for s in range(NUM_SUITS):
        s_cards = hand & SUIT_MASKS[s]
        s_ranks = sorted([rank_of(c) for c in np.where(s_cards)[0]], reverse=True)
        l = len(s_ranks)
        if l == 0:
            continue
        if 12 in s_ranks or (11 in s_ranks and l >= 2) or (10 in s_ranks and l >= 3):
            stoppers += 1

        # Losing trick count in suit (evaluate top 3 cards)
        top3 = s_ranks[:min(3, l)]
        suit_losers = 0
        if 12 not in top3 and l >= 1: suit_losers += 1
        if 11 not in top3 and l >= 2: suit_losers += 1
        if 10 not in top3 and l >= 3: suit_losers += 1
        losers += suit_losers

    return np.array([
        hcp, ace_count, king_count, queen_count, jack_count,
        low_cards, max_len, second_len, min_len,
        voids, singletons, doubletons, trump_hcp,
        int(has_ak), int(has_akq), stoppers, losers
    ], dtype=np.float32)


def run_rule_extraction(max_shards: int = 25):
    """
    Extract bidding rules for each contract type using shallow decision trees.
    """
    print(f"Loading data from up to {max_shards} shards for Explainable AI Rule Extraction...")
    shard_files = sorted(glob.glob("data/shard_*.npz") + glob.glob("data/self_play/iter_*/*.npz"))[:max_shards]
    if not shard_files:
        raise FileNotFoundError("No training shards found in data/ or data/self_play/")

    contract_data = {c: {'X': [], 'y': []} for c in range(1, 14)}

    for s_path in shard_files:
        data = np.load(s_path)
        hands = data['bvn_hands']
        bids = data['bvn_bid_taken']
        outcomes = data['bvn_outcome']

        for i in range(len(bids)):
            bid = int(bids[i])
            if 1 <= bid <= 12:
                feat = extract_features_from_hand(hands[i])
                contract_data[bid]['X'].append(feat)
                contract_data[bid]['y'].append(int(outcomes[i]))

    print("\n" + "=" * 78)
    print("  EXPLAINABLE AI (XAI): EXTRACTED BIDDING RULES & THRESHOLDS")
    print("=" * 78)

    rules_summary = {}

    for c in range(1, 14):
        name = CONTRACT_NAMES[c]
        X = np.array(contract_data[c]['X'])
        y = np.array(contract_data[c]['y'])

        if len(X) < 50:
            print(f"\n--- Contract: {name} (Insufficient samples: {len(X)}) ---")
            continue

        base_win_rate = np.mean(y)

        # Fit shallow decision tree (depth 3) for clean interpretable rule extraction
        dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(10, int(len(X) * 0.05)))
        dt.fit(X, y)

        # Fit Random Forest for feature importance ranking
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        top_feat_idx = np.argsort(importances)[::-1][:4]
        top_features = [(FEATURE_NAMES[idx], importances[idx]) for idx in top_feat_idx]

        rules_summary[name] = {
            'samples': len(X),
            'base_win_rate': base_win_rate,
            'top_features': top_features,
            'tree_text': export_text(dt, feature_names=FEATURE_NAMES),
        }

        print(f"\n[{name}] (Total Samples: {len(X):,}, Base Win Rate: {base_win_rate:.1%})")
        print("  Top Key Strategic Drivers:")
        for feat_name, imp in top_features:
            print(f"    - {feat_name:20s}: {imp:.1%} importance")

        print("  Decision Tree Rule Logic:")
        for line in rules_summary[name]['tree_text'].split('\n')[:8]:
            if line.strip():
                print(f"    {line}")

    return rules_summary


if __name__ == '__main__':
    run_rule_extraction()
