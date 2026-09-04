"""
analysis/plot_convergence.py — Multi-generation RL Convergence Visualizer.
Reads eval_history.json and plots:
  1. Overall Win Rate vs Generation
  2. Offense (Declarer) vs Defense Win Rate
  3. Average Tricks Won per Hand
  4. Contract Breakdown Win Rates
"""

from __future__ import annotations
import os
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(history_file: str = "eval_history.json", save_path: str = "docs/convergence.png"):
    if not os.path.exists(history_file):
        print(f"Error: {history_file} not found.")
        return

    with open(history_file, "r") as f:
        data = json.load(f)

    if not data:
        print("eval_history.json is empty.")
        return

    # Extract metrics across iterations
    iters = [d["iteration"] for d in data]
    overall_wr = [d["neural_win_rate"] * 100 for d in data]
    decl_wr = [d.get("declarer_win_rate", 0) * 100 for d in data]
    part_wr = [d.get("partner_win_rate", 0) * 100 if d.get("partner_win_rate") is not None else None for d in data]
    def_wr = [d.get("defender_win_rate", 0) * 100 for d in data]
    neural_tricks = [d.get("neural_avg_tricks_declared", 0) for d in data]
    heur_tricks = [d.get("heuristic_avg_tricks_declared", 0) for d in data]

    # Head-to-head vs prior generation (if present)
    h2h_iters = [d["iteration"] for d in data if d.get("neural_vs_prev_win_rate") is not None]
    h2h_wr = [d["neural_vs_prev_win_rate"] * 100 for d in data if d.get("neural_vs_prev_win_rate") is not None]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    # Plot 1: Overall Win Rate Progression & Head-to-Head
    ax1 = axes[0, 0]
    ax1.plot(iters, overall_wr, marker="o", linewidth=2.5, color="#2563eb", label="vs Heuristic Baseline")
    if h2h_iters:
        ax1.plot(h2h_iters, h2h_wr, marker="D", linewidth=2.0, color="#dc2626", linestyle="-.", label="vs Prior Gen (Head-to-Head)")
    ax1.axhline(50, color="gray", linestyle="--", alpha=0.7, label="Parity (50%)")
    ax1.set_title("1. Win Rate vs Generation", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Generation / Iteration")
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_ylim(20, 100)
    ax1.legend()

    # Plot 2: Role-based Performance (Declarer vs Partner vs Defender)
    ax2 = axes[0, 1]
    ax2.plot(iters, decl_wr, marker="s", linewidth=2, color="#10b981", label="As Declarer (Solo/Lead)")
    valid_part = [(it, pw) for it, pw in zip(iters, part_wr) if pw is not None]
    if valid_part:
        p_iters, p_vals = zip(*valid_part)
        ax2.plot(p_iters, p_vals, marker="o", linewidth=2, color="#06b6d4", label="As Partner (Maatje)")
    ax2.plot(iters, def_wr, marker="^", linewidth=2, color="#f59e0b", label="As Defender")
    ax2.set_title("2. Win Rate by Role (Declarer / Partner / Defender)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Generation / Iteration")
    ax2.set_ylabel("Win Rate (%)")
    ax2.set_ylim(20, 100)
    ax2.legend()

    # Plot 3: Average Tricks Declared
    ax3 = axes[1, 0]
    ax3.plot(iters, neural_tricks, marker="o", linewidth=2, color="#8b5cf6", label="Neural Agent")
    ax3.plot(iters, heur_tricks, marker="x", linewidth=2, color="#ef4444", linestyle="--", label="Heuristic Baseline")
    ax3.set_title("3. Average Tricks Won When Declaring", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Generation / Iteration")
    ax3.set_ylabel("Average Tricks")
    ax3.legend()

    # Plot 4: Latest Contract Win Rate Breakdown
    ax4 = axes[1, 1]
    latest = data[-1]
    cb = latest.get("contract_breakdown", {})
    contracts = []
    win_rates = []
    for c_name, c_info in cb.items():
        if c_info.get("total_bids", 0) > 0 and c_info.get("neural_win_rate") is not None:
            contracts.append(c_name)
            win_rates.append(c_info["neural_win_rate"] * 100)

    if contracts:
        y_pos = np.arange(len(contracts))
        bars = ax4.barh(y_pos, win_rates, color="#3b82f6", alpha=0.85)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(contracts, fontsize=9)
        ax4.set_xlabel("Neural Win Rate (%)")
        ax4.set_xlim(0, 105)
        ax4.set_title(f"4. Contract Performance (Gen {latest['iteration']})", fontsize=12, fontweight="bold")
        for bar, wr in zip(bars, win_rates):
            ax4.text(wr + 1.5, bar.get_y() + bar.get_height()/2, f"{wr:.0f}%", va="center", fontsize=8, fontweight="bold")
    else:
        ax4.text(0.5, 0.5, "Contract data building...", ha="center", va="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"Convergence plot saved to {save_path}")


if __name__ == "__main__":
    plot_convergence()
