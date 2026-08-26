# 07. Phase 1 Dataset Analysis & Heuristic Strategy Report

This document provides a comprehensive analysis of the **1,000,000-game dataset** generated during Phase 1 for bootstrapping the Neural Networks (BVN & Belief Network).

---

## 1. Executive Summary

| Metric | Value | Description |
|:---|:---:|:---|
| **Total Games Simulated** | **1,000,000** | 100 compressed `.npz` shards × 10,000 games/shard |
| **Total Bidding Decisions** | **4,632,787** | Supervised training pairs for the BVN |
| **Total Trick State Transitions** | **28,716,172** | Supervised training pairs for the Belief Network |
| **Average Bids per Game** | **4.63** | Average auction length before contract finalization |
| **Average Trick Transitions per Game** | **28.72** | Average card-by-card state updates per game |
| **Dataset Size on Disk** | **435.3 MB** | Compressed NumPy (`int8` bitmasks / one-hot tensors) |
| **Generation Time** | **29.8 min** | Generated via 4-worker multiprocessing on Apple Silicon |

---

## 2. Player Strategy Specification (`HeuristicAgent`)

The 1,000,000 games were generated using 4 identical, independent instances of the rule-based `HeuristicAgent`. The agent implements standard Dutch trick-taking card heuristics across bidding, declaration, and trick-taking.

### 2.1 Hand Evaluation & Bidding Logic
Every player evaluates their private 13-card hand using a high-card point (HCP) and distributional bonus formula:

$$\text{Strength} = \sum \text{HCP} + \sum \text{LengthBonus} + \text{VoidBonus}$$

- **High Card Points**: Ace = 4, King = 3, Queen = 2, Jack = 1.
- **Length Bonus**: $+2$ points for every card beyond the 4th in any suit (e.g., a 6-card suit gives $+4$).
- **Void Bonus**: $+1$ point for a complete void in any side suit.

#### Bidding Ladder Thresholds:
1. **Misère Check**: Hand holds zero Aces, no cards $\ge 10$ (no J, Q, K, A) across $\ge 3$ suits $\implies$ Bid `MISERE`.
2. **Piek Check**: One dominant suit ($\ge 7$ cards) and short singletons in others $\implies$ Bid `PIEK`.
3. **Troela Check**: If holding $\ge 3$ Aces $\implies$ Bid `TROELA` with probability `TROELA_CALL_RATE = 0.70`.
4. **Alleen Ladder**:
   - $\text{Strength} \ge 22$ and $\ge 3$ Aces $\implies$ `SOLO_SLIM` (13 tricks)
   - $\text{Strength} \ge 19$ and $\ge 2$ Aces $\implies$ `TWAALF_ALLEEN` (12 tricks)
   - $\text{Strength} \ge 17$ and $\ge 2$ Aces $\implies$ `ELF_ALLEEN` (11 tricks)
   - $\text{Strength} \ge 15$ and $\ge 2$ Aces $\implies$ `TIEN_ALLEEN` (10 tricks)
   - $\text{Strength} \ge 13$ and $\ge 1$ Ace $\implies$ `NEGEN_ALLEEN` (9 tricks)
   - $\text{Strength} \ge 11$ and $\ge 1$ Ace $\implies$ `ACHT_ALLEEN` (8 tricks)
   - $\text{Strength} \ge 9$ and $\ge 1$ Ace $\implies$ `RIK` (8 tricks with partner)
   - Otherwise $\implies$ `PAS`.

### 2.2 Declaration Strategy
- **Trump Choice**: The declarer selects the suit with the largest card count and highest rank density. For `TROELA`, the partner's 4th Ace suit is strictly excluded.
- **Vraagaas Choice (Rik)**: The declarer requests the Ace of their weakest side-suit (the suit where partner assistance is most needed).

### 2.3 Trick-Taking Tactics
1. **Leading**:
   - If holding the trump majority, lead high trumps to strip defenders of their trumps.
   - Otherwise, lead side Aces (guaranteed trick winners) or top cards of the longest established suit.
2. **Following**:
   - Follow suit is strictly enforced.
   - If the player's partner is currently winning the trick, duck (play lowest card).
   - If attempting to win, play the **lowest winning card** ("win cheaply").
3. **Must-Trump & Discard**:
   - If void in the led suit, play the lowest trump that beats the current winner.
   - If void in both led suit and trump suit, discard the lowest card from the shortest suit.
4. **Misère / Piek Logic**:
   - Misère: Always duck; when leading, lead lowest card. When following, play the highest card that is still strictly lower than the current trick leader.
   - Piek: Aggressively win trick 1 (if aiming for 1 trick) or win 5 tricks and discard losers thereafter.

---

## 3. Empirical Dataset Statistics (1,000,000 Games)

The table below breaks down all **4,632,787 bidding decisions** recorded across the full dataset:

| Contract / Action | Action ID | Decision Count | Bidding Frequency | Bidder Win Rate | Strategic Role in Dataset |
|:---|:---:|:---:|:---:|:---:|:---|
| `PAS` | 0 | 2,999,979 | 64.76% | 90.50% | Defender baseline (defenders win when declarer fails) |
| `RIK` | 1 | 86,106 | 1.86% | 96.37% | High-equity partner contract |
| `ACHT_ALLEEN` | 2 | 126,370 | 2.73% | 92.79% | High-equity solo contract |
| `PIEK` | 3 | 196,317 | 4.24% | 78.27% | Specialized 1-or-5 trick contract |
| `NEGEN_ALLEEN` | 4 | 455,854 | 9.84% | 38.94% | Competitive solo threshold |
| `MISERE` | 5 | 102,547 | 2.21% | 64.99% | Specialized 0-trick contract |
| `TIEN_ALLEEN` | 6 | 127,365 | 2.75% | 7.93% | Challenging high solo contract |
| `ELF_ALLEEN` | 9 | 133,667 | 2.89% | 4.33% | Aggressive overbid category |
| `TWAALF_ALLEEN` | 10 | 268,965 | 5.81% | 0.24% | High-variance overbid category |
| `SOLO_SLIM` | 11 | 21,598 | 0.47% | 0.39% | 13-trick grand slam attempt |
| `TROELA` | 12 | 114,019 | 2.46% | 85.31% | 3-Ace partner contract |
| **Total** | — | **4,632,787** | **100.00%** | — | Complete Phase 1 distribution |

---

## 4. Key Insights & Machine Learning Value

### 4.1 Bidding Distribution & Overbidding Diversity
In human play, bidding 10, 11, or 12 Alleen is relatively rare. The heuristic agent used point thresholds ($\ge 15, 17, 19$) that frequently attempt these ambitious contracts.
- **Why this is advantageous for training**: If the bootstrapping agent only bid contracts with 100% win certainty, the dataset would lack negative examples and contrastive signals.
- The presence of both easy wins (`RIK`: 96.4%, `ACHT_ALLEEN`: 92.8%) and difficult/overbid hands (`TIEN_ALLEEN`: 7.9%, `TWAALF_ALLEEN`: 0.24%) provides the **Bidding Value Network (BVN)** with a smooth, continuous gradient to learn exact win-probability curves for every hand configuration.

### 4.2 Physical Clumping in the Belief Network Data
Because the dataset was generated using the **clumping shuffle** (`SHUFFLE_INTENSITY = 3` with 6-then-7 round-robin dealing):
- Card distributions in opponents' hands exhibit realistic run-lengths and suit clustering from the previous hand's collected tricks.
- The **Belief Network (BN)** trained on these ~28.7M state transitions will learn true conditional distributions $P(\text{card } c \in \text{Opponent } p \mid \text{Public State}, \text{Void Matrix})$ rather than idealized uniform Bernoulli distributions.

---

## 5. Next Step: Supervised Training (Phase 2)

With the dataset validated and analyzed, Phase 2 can proceed:
1. **Train BVN** (`python main.py train-bvn --epochs 30`): Learns the mapping $\text{Hand} + \text{Bid History} \to \mathbb{E}[\text{Payoff}]$ for all 13 contract choices.
2. **Train BN** (`python main.py train-bn --epochs 30`): Learns opponent hand probability distributions across all 28.7M trick-taking states.
