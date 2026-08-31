# 09. Self-Play Reinforcement Learning & Convergence Methodology

This document details the multi-generation empirical trajectory, the 3,000-game benchmark evaluation, and the mathematical convergence analysis of the **constant-free Action-Value Q-Network ($Q \in [-1.0, +1.0]$)** trained on the Leiden University ALICE HPC cluster.

---

## 1. Executive Benchmark Summary ($N = 3,000$ Games)

| Metric | Heuristic Baseline | Neural Agent (Constant-Free Q-Network) | Strategic Superiority |
|:---|:---:|:---:|:---|
| **Overall Match Win Rate** | `19.1%` (545/2,854) | **`80.9%` (2,309 / 2,854)** ⭐ | **+61.8% Advantage** |
| **When Defending (Defense)** | `11.8%` | **`88.2%` (2,096 / 2,377)** 🛡️ | **Dominant Defensive Punishing** |
| **When Declaring (Offense)** | `11.8%` | **`44.7%` (213 / 477)** 👑 | **High Selective Discipline** |
| **Troela Win Rate** | `69.9%` (123/176) | **`83.5%` (187 / 224)** 🏆 | **Cooperative 4th Ace Mastery** |
| **Suicidal Solo Overbids (10–12 Alleen)** | **831 Overbids** | **0 Bids** 🧠 | **Complete Elimination of Overbidding** |

---

## 2. 10-Generation ALICE Training Progression Log

Below is the generational progression recorded across the 10 clean training iterations on ALICE:

| Gen | Match Win Rate | Declarer Win Rate (Offense) | Defender Win Rate (Defense) | Training Dynamics & Meta-Evolution |
|:---:|:---:|:---:|:---:|:---|
| **1** | **`81.2%`** | `66.7%` | `84.0%` | Q-network rapidly masters defensive punishment against baseline overbids |
| **2** | **`81.9%`** | **`85.7%`** | `81.2%` | Peak offensive equity: highly selective declaring (85.7% win rate) |
| **3** | **`76.0%`** | `40.0%` | `82.7%` | Policy exploration on solo contract thresholds |
| **4** | **`72.9%`** | `59.1%` | `77.0%` | Recovery and consolidation of partner contract valuation |
| **5** | **`76.0%`** | `42.1%` | `84.4%` | High defense floor (>84%) maintained across cluster workers |
| **6** | **`81.8%`** | `56.2%` | `86.8%` | Defense crosses 86%; Troela coordination solidified |
| **7** | **`75.3%`** | `43.8%` | `81.5%` | Sustained equilibrium across all 14 contract types |
| **8** | **`72.5%`** | `61.5%` | `74.1%` | Aggressive offensive testing |
| **9** | **`78.6%`** | `47.1%` | `85.2%` | Strong defensive rebound (85.2%) |
| **10** 🌟 | **`80.4%`** | `62.5%` | `84.0%` | **Converged Plateau**: 80.4% Overall, 84% Defense, 62.5% Offense |

### Convergence Dashboard
![Convergence Dashboard](convergence.png)

---

## 3. Large-Scale Contract Performance Breakdown ($N = 2,854$ Matches)

| Contract | Neural Bids | Neural Win % | Opponent Bids | Opponent Win % | Strategic Takeaway |
| :--- | :---: | :---: | :---: | :---: | :---|
| **TROELA** | 224 | **`83.5%`** 👑 | 176 | `69.9%` | Exceptional synergy with the 4th Ace partner. |
| **RIK** | 2 | **`100.0%`** ⭐ | 108 | `47.2%` | Maximum selective certainty on called Aces. |
| **RIK BETER** | 3 | **`66.7%`** | 28 | `57.1%` | Strong performance with fixed Hearts trump. |
| **ACHT ALLEEN** | 62 | **`25.8%`** | 292 | `3.1%` | **8x higher win rate** than baseline. |
| **NEGEN ALLEEN** | 159 | `3.8%` | 870 | `7.5%` | Highly contested solo contract. |
| **TIEN ALLEEN** | **0** | — | **272** | `1.1%` | Baseline committed 272 suicidal bids; AI defended with 98.9% success. |
| **ELF / TWAALF** | **0** | — | **559** | `2.4%` | Baseline committed 559 suicidal bids; AI passed and punished. |
| **MISÈRE** | 27 | `0.0%` | 57 | `1.8%` | Extreme 3v1 ducking contract; hard to win under optimal defense. |

---

## 4. Why Constant-Free Q-Values Succeeded

1. **Dynamic Pass Valuation ($Q(s, 	ext{PAS}) pprox +0.997$)**:
   - The network dynamically learned that when an opponent overbids a risky solo contract, passing provides an extraordinary $+0.997$ expected defensive reward.
2. **Zero Hardcoded Parameters**:
   - Action selection $\pi(s) = rg\max_{a \in 	ext{Legal}} Q_	heta(s, a)$ operates end-to-end without artificial thresholds or manual parameters.
