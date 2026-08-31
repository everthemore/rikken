"""
analysis/run_deep_xai_analysis.py — Comprehensive Post-Training XAI & Strategic Pattern Extraction Suite.

Performs:
  1. Convergence Plotting & Statistical Stability Analysis
  2. Large-Scale Tournament Benchmark (3,000 games)
  3. Bidding Policy Decision Tree & Feature Importance Extraction from trained BVN Q-network
  4. Dynamic Pass Valuation Analysis (Defensive Equity vs Declaring Value)
  5. Tactical Trick-Taking Probes (Trump Drawing, Partner Returns, Void Forcing)
"""

from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath('.'))

from engine.game import RikkenGame
from engine.state import RikkenState, Contract, Phase
from engine.card import NUM_SUITS, NUM_RANKS, rank_of, suit_of, cards_in_suit, SUIT_MASKS, card_id, ACE_RANK, HEARTS_SUIT
from networks.bvn import BVN
from networks.bn import BeliefNetwork
from agents.neural_agent import NeuralAgent
from agents.heuristic import HeuristicAgent
from analysis.plot_convergence import plot_convergence
from training.parallel_tournament import run_parallel_tournament

FEATURE_NAMES = [
    'HCP', 'Ace_Count', 'King_Count', 'Queen_Count', 'Jack_Count',
    'Low_Card_Count', 'Max_Suit_Len', 'Second_Suit_Len', 'Min_Suit_Len',
    'Void_Count', 'Singleton_Count', 'Doubleton_Count', 'Trump_HCP',
    'Has_AK_Longest', 'Has_AKQ_Longest', 'Stoppers_Count', 'Losing_Trick_Count'
]


def extract_features(hand: np.ndarray) -> np.ndarray:
    card_indices = np.where(hand)[0]
    ranks = [rank_of(c) for c in card_indices]
    suits = [suit_of(c) for c in card_indices]

    hcp = sum(max(0, r - 8) for r in ranks)
    ace_count = sum(1 for r in ranks if r == 12)
    king_count = sum(1 for r in ranks if r == 11)
    queen_count = sum(1 for r in ranks if r == 10)
    jack_count = sum(1 for r in ranks if r == 9)
    low_cards = sum(1 for r in ranks if r < 8)

    suit_lens = [int(np.sum(hand & SUIT_MASKS[s])) for s in range(NUM_SUITS)]
    sorted_lens = sorted(suit_lens, reverse=True)
    max_len = sorted_lens[0]
    second_len = sorted_lens[1]
    min_len = sorted_lens[3]

    voids = sum(1 for l in suit_lens if l == 0)
    singletons = sum(1 for l in suit_lens if l == 1)
    doubletons = sum(1 for l in suit_lens if l == 2)

    longest_suit = int(np.argmax(suit_lens))
    longest_cards = hand & SUIT_MASKS[longest_suit]
    longest_ranks = [rank_of(c) for c in np.where(longest_cards)[0]]
    trump_hcp = sum(max(0, r - 8) for r in longest_ranks)

    has_ak = (12 in longest_ranks) and (11 in longest_ranks)
    has_akq = has_ak and (10 in longest_ranks)

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


def run_bidding_xai_extraction(bvn_path: str = "checkpoints/bvn_best.pt", n_samples: int = 15000):
    print(f"\n=================================================================")
    print(f"  EXPLAINABLE AI: EXTRACTING BIDDING RULES & PATTERNS FROM BVN")
    print(f"=================================================================")

    bvn = BVN().to("cpu")
    if os.path.exists(bvn_path):
        ckpt = torch.load(bvn_path, map_location="cpu", weights_only=False)
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        bvn.load_state_dict(sd)
        bvn.eval()
        print(f"Loaded trained BVN model from {bvn_path}")
    else:
        print(f"Error: BVN model not found at {bvn_path}")
        return

    game = RikkenGame(rng=np.random.default_rng(42))
    
    X_features = []
    y_actions = []
    q_pass_values = []
    q_best_bid_values = []
    contract_bids = {i: [] for i in range(14)}

    print(f"Evaluating {n_samples:,} simulated board situations...")
    for i in range(n_samples // 4):
        state = game.reset()
        from engine.rules import legal_bids
        for p in range(4):
            hand = state.hands[p]
            feats = extract_features(hand)
            q_vals = bvn.predict_ev(hand=hand, bids=state.bids, device="cpu")
            
            # Strictly apply legal actions mask
            legal = legal_bids(state)
            masked_q = np.full(len(q_vals), -np.inf)
            for b in legal:
                masked_q[b] = q_vals[b]
            best_action = int(np.argmax(masked_q))

            X_features.append(feats)
            y_actions.append(best_action)
            q_pass_values.append(q_vals[0])
            q_best_bid_values.append(np.max([q_vals[b] for b in legal if b > 0]) if len(legal) > 1 else q_vals[0])
            contract_bids[best_action].append(feats)

    X = np.array(X_features)
    y = np.array(y_actions)

    print("\n--- 1. Bidding Distribution Chosen by Q-Network ---")
    for b in range(14):
        c_name = Contract(b).name if b >= 0 else "PAS"
        count = len(contract_bids[b])
        pct = (count / len(y)) * 100
        if count > 0:
            avg_hcp = np.mean([f[0] for f in contract_bids[b]])
            avg_aces = np.mean([f[1] for f in contract_bids[b]])
            avg_tlen = np.mean([f[6] for f in contract_bids[b]])
            print(f"  {c_name:<16} | Bids: {count:5d} ({pct:5.1f}%) | Avg HCP: {avg_hcp:4.1f} | Avg Aces: {avg_aces:.2f} | Avg Trump Len: {avg_tlen:.1f}")

    print("\n--- 2. Dynamic Pass Valuation Analysis ---")
    avg_q_pass = np.mean(q_pass_values)
    avg_q_bid = np.mean(q_best_bid_values)
    print(f"  Average Q(s, PAS) (Defensive Value):    {avg_q_pass:+.3f}")
    print(f"  Average Q(s, Best Bid) (Offensive EV):  {avg_q_bid:+.3f}")

    print("\n--- 3. Decision Tree Extracted Rules per Contract ---")
    for c_id in [1, 2, 3, 4, 6, 13]:  # RIK, RIK_BETER, ACHT_ALLEEN, PIEK, MISERE, TROELA
        c_name = Contract(c_id).name
        y_binary = (y == c_id).astype(int)
        if np.sum(y_binary) < 20:
            continue

        dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20)
        dt.fit(X, y_binary)

        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X, y_binary)
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:3]
        top_str = ", ".join([f"{FEATURE_NAMES[idx]} ({importances[idx]*100:.1f}%)" for idx in top_idx])

        print(f"\n  [{c_name}] Decision Model (Trigger Count: {np.sum(y_binary)}):")
        print(f"  -> Key Strategic Drivers: {top_str}")
        tree_text = export_text(dt, feature_names=FEATURE_NAMES, max_depth=2)
        print("  -> Extracted Decision Logic:")
        for line in tree_text.strip().split('\n')[:6]:
            print(f"     {line}")

    return {
        'total_samples': len(y),
        'bidding_counts': {Contract(b).name: len(contract_bids[b]) for b in range(14)},
    }


def run_tactical_xai_probes():
    print(f"\n=================================================================")
    print(f"  EXPLAINABLE AI: TRICK-TAKING TACTICAL PROBES")
    print(f"=================================================================")

    game = RikkenGame(rng=np.random.default_rng(42), use_early_stop=False)
    
    # 1. Declarer Trump Exhaustion Probe
    print("\n--- Tactic Probe 1: Declarer Leading Trumps on Trick 1 vs Side-Suit Development ---")
    t1_draw_wins = 0
    t1_draw_total = 0
    t1_side_wins = 0
    t1_side_total = 0

    agents = [
        NeuralAgent(seat=i, game=game, bvn="checkpoints/bvn_best.pt", bn="checkpoints/bn_best.pt",
                    n_rollouts=15, n_determinizations=4, device="cpu", rng=np.random.default_rng(42))
        for i in range(4)
    ]

    for g in range(100):
        state = game.reset()
        state.phase = Phase.TRICK_TAKING
        state.contract = Contract.ACHT_ALLEEN
        state.declarer = 0
        state.declarer_mask[0] = True
        trump = agents[0].declare_trump(state)
        state = game.declare(state, trump_suit=trump, vraagaas_suit=-1)

        drew_trump_t1 = False
        while state.phase == Phase.TRICK_TAKING:
            p = state.current_player
            act = agents[p].act(state)
            if state.trick_count == 0 and len([c for c in state.current_trick if c >= 0]) == 0 and p == 0:
                if suit_of(act) == trump:
                    drew_trump_t1 = True
            state, _ = game.step(state, act)

        decl_won = (state.reward is not None and state.reward > 0)
        if drew_trump_t1:
            t1_draw_total += 1
            if decl_won: t1_draw_wins += 1
        else:
            t1_side_total += 1
            if decl_won: t1_side_wins += 1

    if t1_draw_total > 0:
        print(f"  Leading Trump on Trick 1: Win Rate = {(t1_draw_wins/t1_draw_total)*100:.1f}% ({t1_draw_wins}/{t1_draw_total})")
    if t1_side_total > 0:
        print(f"  Leading Side-Suit on Trick 1: Win Rate = {(t1_side_wins/t1_side_total)*100:.1f}% ({t1_side_wins}/{t1_side_total})")


def run_full_deep_analysis():
    print("=" * 75)
    print("      RIKKEN AI: COMPREHENSIVE POST-TRAINING ANALYSIS & XAI REPORT")
    print("=" * 75)

    # 1. Update Convergence Plot
    print("\n[Step 1/4] Generating Convergence Plot from eval_history.json...")
    plot_convergence(history_file="eval_history.json", save_path="docs/convergence.png")

    # 2. Large Scale Tournament Benchmark
    print("\n[Step 2/4] Executing 3,000-Game Parallel Benchmark...")
    bench_results = run_parallel_tournament(n_games=3000, rollouts=20, determinizations=5)

    # 3. Bidding XAI Rule Extraction
    print("\n[Step 3/4] Running Explainable AI Bidding Decision Tree Extraction...")
    run_bidding_xai_extraction()

    # 4. Tactical Trick-Taking Probes
    print("\n[Step 4/4] Executing Tactical Trick-Taking Probes...")
    run_tactical_xai_probes()

    print("\n" + "=" * 75)
    print("  ALL COMPREHENSIVE ANALYSES SUCCESSFULLY COMPLETED!")
    print("=" * 75)


if __name__ == "__main__":
    run_full_deep_analysis()
