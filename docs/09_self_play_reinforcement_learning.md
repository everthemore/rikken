# 09. Self-Play Reinforcement Learning & Evaluation Methodology

This document details the AlphaZero-inspired **Expert Iteration (ExIt)** reinforcement learning loop, the **Dual-Head Bidding Value Network**, and the **True Individual (1-vs-3) Evaluation Tournament** for Dutch Rikken.

---

## 1. The Reinforcement Learning Framework (AlphaZero / ExIt)

Rikken AI trains through pure self-play policy iteration with **zero external human data**:

```
              ┌──────────────────────────────────────────────────┐
              │                                                  │
              ▼                                                  │
     [ Policy Improvement ]                                      │
  4 Neural Agents play games via ISMCTS                          │
  (MCTS lookahead search plays stronger                          │
   than the raw neural network alone)                            │
              │                                                  │
              ▼                                                  │
  Self-Play Experience Collection                                │
  (States: hands, bid_history, voids -> Actions -> Match Reward) │
              │                                                  │
              ▼                                                  │
     [ Policy Evaluation ]                                       │
  Dual-Head BVN & Belief Network Retrained                       │
  on rolling replay buffer (BUFFER_WINDOW = 5 generations)       │
              │                                                  │
              └──────────────────────────────────────────────────┘
```

1. **Policy Improvement (Search)**:
   During self-play, agents use **Information Set MCTS (ISMCTS)** with Belief Network card determinization and rollout lookahead. Lookahead search acts as a policy improvement operator that discovers better moves than the raw network.
2. **Policy Evaluation (Deep Learning)**:
   The neural networks distill the outcomes of those games into instant intuitions (evaluating private hands in milliseconds during bidding).
3. **Rolling Replay Buffer (`BUFFER_WINDOW = 5`)**:
   Retraining samples across a sliding window of the latest 5 generations (up to 75,000 games) to ensure smooth policy evolution without catastrophic forgetting or overfitting to a single iteration.

---

## 2. The Dual-Head BVR Architecture: Solving the Expected Value Paradox

### 2.1 The Expected Value Trap in Card Games
In standard Dutch Rikken scoring, stakes scale steeply with contract difficulty:
* **RIK / RIK BETER**: $\pm 1$ pt per opponent (Net $\pm 1$ pt each)
* **PIEK**: $\pm 2$ pts per opponent (Net $\pm 6$ pts solo)
* **OPEN PIEK / MISÈRE**: $\pm 3$ pts per opponent (Net $\pm 9$ pts solo)

Because payoffs are symmetric ($+S$ on win, $-S$ on loss), expected points scale with $S$:
$$\mathbb{E}[\text{Points}] = S \times (2 P(\text{Win}) - 1)$$

If an agent bids by maximizing **linear Expected Value ($\mathbb{E}[\text{Points}]$)**:
* An **80% lock on RIK** yields: $1 \times (2 \times 0.80 - 1) = \mathbf{+0.60\text{ pts}}$
* A **55% risky coin-flip on OPEN PIEK** yields: $9 \times (2 \times 0.55 - 1) = \mathbf{+0.90\text{ pts}}$

The linear EV formula mathematically incentivized the agent to gamble on high-stake solo contracts (like OPEN PIEK) whenever win chance was slightly over 50%, completely refusing to bid safe contracts like RIK!

### 2.2 The Dual-Head Solution
To eliminate reckless gambling while preserving expected score analytics, the BVN features **two distinct heads**:

1. **Win-Probability Head ($P_\theta(\text{Win} \mid s, a) \in [0.0, 1.0]$)**:
   - Output: `Linear -> Sigmoid`
   - Loss: Binary Cross-Entropy on whether the player won points ($y_{\text{won}} \in \{0.0, 1.0\}$)
   - **Policy Action**: $\text{Bid} = \arg\max_{a \in \text{Legal}(s)} P_\theta(\text{Win} \mid s, a)$
   - Result: The bot bids the contract with the **highest probability of winning** ($80\% \text{ RIK} > 55\% \text{ PIEK}$), naturally selecting natural human-like contracts.
2. **Expected-Value Head ($\mathbb{E}_\theta[\text{Points} \mid s, a] \in [-1.0, +1.0]$)**:
   - Output: `Linear -> Tanh`
   - Loss: Huber regression on normalized game payoffs ($\div 10$)
   - Used for the UI advice panel and strategic score evaluation.

---

## 3. Dealing, Position & Opening Lead Rules

1. **Clockwise Dealer Rotation**:
   The dealer rotates clockwise on each deal: `dealer = (dealer + 1) % 4`.
2. **Voorhand Opening Lead**:
   The player to the left of the dealer (`voorhand = (dealer + 1) % 4`) is **Voorhand**:
   * Voorhand speaks first in the bidding auction.
   * **Voorhand always leads Trick 1**, regardless of who declared the contract. (On subsequent tricks, the trick winner leads).

---

## 4. True Individual (1-vs-3) Evaluation Tournament

In Rikken, **there are no fixed teams**. Unlike Bridge (where North/South and East/West are permanent partners), partnerships in Rikken are dynamic:
* In `RIK`, the declarer calls an Ace (`vraagaas`). The holder of that Ace becomes their partner (*maatje*).
* In `PIEK`, `MISÈRE`, and Solo contracts, it is 1 against 3.
* In `OPEN PIEK` / `MISÈRE`, multiple players can join the contract simultaneously without being partners.

### 4.1 The 1-vs-3 Tournament Structure (`training/tournament.py`)
To prevent partnership contamination, the tournament pits:
**1 Evaluated Agent vs 3 Opponents** across 400 games.

### 4.2 16-Game Symmetric Block Rotation
To ensure absolute mathematical fairness with zero seat or dealer bias:
```python
for g in range(n_games):
    eval_seat = g % 4          # Rotates evaluated agent through seats 0, 1, 2, 3
    dealer = (g // 4) % 4      # Cycles dealer through 0, 1, 2, 3
    voorhand = (dealer + 1) % 4
```
In every 16 games:
Every possible combination of `(eval_seat, dealer)` is played **exactly once**. Across 400 games, each position is tested 25 times.

### 4.3 Granular Role Tracking
Outcome is determined strictly by the evaluated seat's individual zero-sum payoff (`state.rewards[eval_seat] > 0`):
* **Declarer Win Rate**: When the agent declared the contract.
* **Partner (*Maatje*) Win Rate**: When an opponent declared and called the agent's Ace.
* **Defender Win Rate**: When the agent was on defense.
* **Simultaneous Piek/Misère**: Evaluated purely on the agent's own score, unaffected by whether the other declarer succeeded or failed.

---

## 5. Dual-Track Benchmarking (AlphaZero Gating)

During GPU retraining on ALICE (`cluster/submit_retrain.slurm`), each generation undergoes two rigorous benchmark matches:

1. **Track 1: vs Heuristic Baseline Anchor (400 Games)**:
   - Evaluates 1 Neural vs 3 Rule-Based Heuristics.
   - Provides an unmoving, stationary measuring stick across all generations.
2. **Track 2: Head-to-Head vs Previous Generation Champion (200 Games)**:
   - Evaluates 1 Candidate (`bvn_final.pt`, Gen $N$) vs 3 Champions (`bvn_gen_{N-1}.pt`, Gen $N-1$).
   - Direct AlphaZero gating: verifies policy improvement ($> 50\%$) before model promotion.
   - Logs `neural_vs_prev_win_rate` to `eval_history.json`.

---

## 6. Convergence Visualization (`analysis/plot_convergence.py`)

The visualizer generates `docs/convergence.png` featuring 4 synchronized panels:
1. **Win Rate vs Generation**: Displays the Baseline Anchor curve and Head-to-Head Prior Generation line against the 50% parity mark.
2. **Win Rate by Role**: Separate trajectories for Declarer (Solo/Lead), Partner (*Maatje*), and Defender.
3. **Average Tricks Won**: Tricks captured when declaring vs opponent baseline.
4. **Contract Breakdown**: Granular win rates per contract type (Rik, Rik Beter, Piek, Misère, Troela).
