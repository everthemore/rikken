# analysis/run_all_analysis.py
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath('.'))

def run_pipeline():
    print("=" * 70)
    print("  RIKKEN AI: COMPLETE EVALUATION & XAI ANALYSIS PIPELINE")
    print("=" * 70)

    # 1. Inspect eval_history.json
    history_file = "eval_history.json"
    if os.path.exists(history_file):
        print("\n[1/3] POLICY EVOLUTION SUMMARY (eval_history.json):")
        print("-" * 70)
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
            
            print(f"{'Gen':<6} | {'Overall WR':<12} | {'Offense WR':<12} | {'Defense WR':<12} | {'Timestamp':<19}")
            print("-" * 70)
            for entry in history:
                gen = f"Gen {entry.get('iteration', '?')}"
                if entry.get('type') == 'baseline':
                    gen = "Baseline"
                wr = f"{entry.get('neural_win_rate', 0)*100:.1f}%"
                off = f"{entry.get('declarer_win_rate', 0)*100:.1f}%" if entry.get('declarer_win_rate') is not None else "-"
                defe = f"{entry.get('defender_win_rate', 0)*100:.1f}%" if entry.get('defender_win_rate') is not None else "-"
                ts = entry.get('timestamp', '')
                print(f"{gen:<6} | {wr:<12} | {off:<12} | {defe:<12} | {ts:<19}")
        except Exception as e:
            print(f"Error loading eval_history.json: {e}")
    else:
        print("\n[1/3] No eval_history.json found.")

    # 2. Run XAI Bidding Rule Extraction
    print("\n[2/3] EXPLAINABLE AI: EXTRACTING BIDDING RULES & THRESHOLDS:")
    print("-" * 70)
    try:
        from analysis.rule_extraction import run_rule_extraction
        run_rule_extraction(max_shards=25)
    except Exception as e:
        print(f"Rule extraction note: {e}")

    # 3. Run In-game Tactical Rule Extraction
    print("\n[3/3] IN-GAME TACTICAL IMPACT EVALUATION:")
    print("-" * 70)
    try:
        from analysis.extract_tactical_rules import evaluate_tactical_rules
        evaluate_tactical_rules(n_games=300)
    except Exception as e:
        print(f"Tactical evaluation note: {e}")

    print("\n" + "=" * 70)
    print("  ANALYSIS PIPELINE COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
