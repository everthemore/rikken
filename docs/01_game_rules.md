# 01. Game Rules & Mechanics

## 1. Overview
Rikken is a traditional Dutch 4-player trick-taking card game played with a standard 52-card deck (no jokers). It features a bidding ladder ranging from cooperative partner contracts to solo and negative contracts (Misère / Piek).

---

## 2. Dealing & Card Hierarchy
- **Deck**: 52 cards, 4 suits (♣ Clubs, ♦ Diamonds, ♥ Hearts, ♠ Spades), 13 ranks each ($2 < 3 < \dots < 10 < \text{J} < \text{Q} < \text{K} < \text{A}$).
- **Dealing**: 13 cards per player, dealt in two rounds: 6 cards each, then 7 cards each.
- **Clumping Shuffle**: Real-world Rikken uses an imperfect riffle shuffle where previous tricks remain partially clumped, creating realistic non-uniform suit distributions.

---

## 3. The Bidding Ladder

Contracts in ascending order:

| Rank | Contract | Type | Trump | Target | Strategic Description |
|:---:|:---|:---:|:---:|:---:|:---|
| 0 | **PAS** | — | — | — | Pass (no bid) |
| 1 | **RIK** | Partner | Caller's choice (♣/♦/♠) | 8+ tricks | Declarer calls a non-held Ace |
| 2 | **RIK BETER** | Partner | **Hearts (♥) Fixed** | 8+ tricks | Beats regular Rik; calls a non-held Ace |
| 3 | **ACHT ALLEEN** | Solo | Caller's choice | 8+ tricks | Solo against 3 defenders |
| 4 | **PIEK** | Solo / Multi | **None** | **Exact 1 OR Exact 5 tricks** | Target 1 trick; if 2nd won, pivot to 5 tricks |
| 5 | **NEGEN ALLEEN** | Solo | Caller's choice | 9+ tricks | Solo against 3 defenders |
| 6 | **MISÈRE** | Solo / Multi | **None** | **Exact 0 tricks** | Multi-player co-bidding allowed |
| 7 | **TIEN ALLEEN** | Solo | Caller's choice | 10+ tricks | Solo against 3 defenders |
| 8 | **OPEN PIEK** | Solo / Multi | **None** | **Exact 1 OR Exact 5 tricks** | Cards face-up after Trick 1 |
| 9 | **OPEN MISÈRE** | Solo / Multi | **None** | **Exact 0 tricks** | Cards face-up after Trick 1 |
| 10 | **ELF ALLEEN** | Solo | Caller's choice | 11+ tricks | Solo against 3 defenders |
| 11 | **TWAALF ALLEEN** | Solo | Caller's choice | 12+ tricks | Solo against 3 defenders |
| 12 | **SOLO SLIM** | Solo | Caller's choice | **13 tricks** | Grand Slam (all tricks) |
| 13 | **TROELA** | Partner | Caller's choice | 8+ tricks | Mandatory bid with 3 or 4 Aces |

---

### 3.1 Piek Dynamic Pivot Strategy
In Piek, the contract is won if the declarer takes **either exactly 1 trick OR exactly 5 tricks**:
1. **Phase 1 (Target: 1 trick)**: Declarer attempts to win trick 1 (burning their high honor), then ducks all remaining tricks.
2. **Phase 2 (Pivot to 5 tricks)**: If forced to win a 2nd trick, declarer **has not lost yet**. They immediately pivot to winning tricks up to 5!
3. **Phase 3 (Lock at 5 tricks)**: Once 5 tricks are won, declarer switches back to Misère ducking to ensure they do not win a fatal 6th trick.
4. **Loss Conditions**: Winning 0 tricks, winning 2/3/4 tricks at end of game, or winning $\ge 6$ tricks.

---

### 3.2 Simultaneous Co-Bidding (Multi-Player Misère & Piek)
- When a player bids **`MISÈRE`**, **`PIEK`**, **`OPEN_MISÈRE`**, or **`OPEN_PIEK`**, other players can also bid the **exact same contract** without having to overbid.
- Multiple declarers play simultaneously against the remaining defenders; each declarer's win/loss is scored independently.

---

## 4. Trick-Taking Priority Rules
On each trick:
1. **Aasvragen Priority**: In `RIK` / `RIK_BETER`, when the Declarer leads the called Ace's suit for the first time, the partner **must play the called Ace**.
2. **Must Follow Suit**: Players must play a card of the led suit if they hold one.
3. **Must Trump (Bekennen/Kopen)**: If void in the led suit, a player **must play a trump card** if they hold one.
4. **Free Discard**: Only if void in both the led suit and the trump suit may a player discard any off-suit card.

---

## 5. Official Dutch Rikken Scoring Schedule & Payoffs

Rikken is played as a **zero-sum point exchange game** ($\sum_{p=0}^3 	ext{Points}[p] = 0$). In the engine and neural training pipeline, points are normalized by $\div 10.0$ for numerical stability:

| Contract | Team Structure | Target Tricks | Declarer Points | Defender Points (each) | Normalized Reward ($\div 10$) | Strategic Risk / Reward |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **RIK / RIK BETER** | 2 vs 2 | 8+ | **$\pm 1.0$** (each) | **$\mp 1.0$** (each) | **$\pm 0.10$** | Low risk baseline cooperative contract |
| **TROELA** | 2 vs 2 | 8+ | **$\pm 1.0$** (each) | **$\mp 1.0$** (each) | **$\pm 0.10$** | Cooperative contract with 4th Ace |
| **ACHT ALLEEN** | 1 vs 3 | 8+ | **$\pm 3.0$** | **$\mp 1.0$** | **$\pm 0.30$** (Decl) / **$\mp 0.10$** (Defs) | 3x stakes solo attack |
| **NEGEN ALLEEN** | 1 vs 3 | 9+ | **$\pm 4.0$** | **$\mp 1.33$** | **$\pm 0.40$** / **$\mp 0.133$** | Higher solo contract |
| **TIEN ALLEEN** | 1 vs 3 | 10+ | **$\pm 5.0$** | **$\mp 1.67$** | **$\pm 0.50$** / **$\mp 0.167$** | High solo contract |
| **ELF ALLEEN** | 1 vs 3 | 11+ | **$\pm 6.0$** | **$\mp 2.00$** | **$\pm 0.60$** / **$\mp 0.200$** | Near-slam solo |
| **TWAALF ALLEEN** | 1 vs 3 | 12+ | **$\pm 7.0$** | **$\mp 2.33$** | **$\pm 0.70$** / **$\mp 0.233$** | Near-slam solo |
| **SOLO SLIM (13)** | 1 vs 3 | 13 | **$\pm 8.0$** | **$\mp 2.67$** | **$\pm 0.80$** / **$\mp 0.267$** | Grand Slam (all 13 tricks) |
| **MISÈRE** | 1 vs 3 | 0 | **$\pm 9.0$** | **$\mp 3.00$** | **$\pm 0.90$** / **$\mp 0.300$** | High penalty 0-trick contract |
| **PIEK** | 1 vs 3 | 1 or 5 | **$\pm 9.0$** | **$\mp 3.00$** | **$\pm 0.90$** / **$\mp 0.300$** | High penalty 1/5-trick contract |
| **OPEN MISÈRE / PIEK** | 1 vs 3 | 0 / (1 or 5) | **$\pm 10.0$** | **$\mp 3.33$** | **$\pm 1.00$** / **$\mp 0.333$** | Open face-up contract |

### 5.1 Multi-Player Scoring (Simultaneous Misère / Piek)
When multiple players co-bid Misère or Piek, each declarer wins $+9$ (or loses $-9$) independently against the remaining defenders:
- If 2 players bid Misère and both win: each declarer gets $+9$, and the 2 defenders each pay $-9$.
- If 1 declarer wins and 1 loses: the winning declarer gets $+9$, the losing declarer pays $-9$, and defenders get $0$ net.
