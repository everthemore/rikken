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

| Rank | Contract | Type | Trump | Target | Notes |
|:---:|:---|:---:|:---:|:---:|:---|
| 0 | **PAS** | — | — | — | Pass |
| 1 | **RIK** | Partner | Caller's choice (♣/♦/♠) | 8+ tricks | Declarer calls a non-held Ace |
| 2 | **RIK BETER** | Partner | **Hearts (♥) Fixed** | 8+ tricks | Beats regular Rik; calls an Ace |
| 3 | **ACHT ALLEEN** | Solo | Caller's choice | 8+ tricks | Solo against 3 defenders |
| 4 | **PIEK** | Solo / Multi | **None** | **Exact 1 trick** | Misère with 1 master winner |
| 5 | **NEGEN ALLEEN** | Solo | Caller's choice | 9+ tricks | Solo against 3 defenders |
| 6 | **MISÈRE** | Solo / Multi | **None** | **Exact 0 tricks** | Multi-player co-bidding allowed |
| 7 | **TIEN ALLEEN** | Solo | Caller's choice | 10+ tricks | Solo against 3 defenders |
| 8 | **OPEN PIEK** | Solo / Multi | **None** | **Exact 1 trick** | Cards face-up after Trick 1 |
| 9 | **OPEN MISÈRE** | Solo / Multi | **None** | **Exact 0 tricks** | Cards face-up after Trick 1 |
| 10 | **ELF ALLEEN** | Solo | Caller's choice | 11+ tricks | Solo against 3 defenders |
| 11 | **TWAALF ALLEEN** | Solo | Caller's choice | 12+ tricks | Solo against 3 defenders |
| 12 | **SOLO SLIM** | Solo | Caller's choice | **13 tricks** | Grand Slam (all tricks) |
| 13 | **TROELA** | Partner | Caller's choice | 8+ tricks | Mandatory bid with 3 or 4 Aces |

### Simultaneous Co-Bidding (Multi-Player Misère & Piek)
- When a player bids **`MISÈRE`**, **`PIEK`**, **`OPEN_MISÈRE`**, or **`OPEN_PIEK`**, other players can also bid the **exact same contract** without having to overbid.
- Multiple declarers play simultaneously against the remaining defenders; each declarer's win/loss is scored independently.

---

## 4. Trick-Taking Priority Rules
On each trick:
1. **Aasvragen Priority**: In `RIK` / `RIK_BETER`, when the Declarer leads the called Ace's suit for the first time, the partner **must play the called Ace**.
2. **Must Follow Suit**: Players must play a card of the led suit if they hold one.
3. **Must Trump (Bekennen/Kopen)**: If void in the led suit, a player **must play a trump card** if they hold one.
4. **Free Discard**: Only if void in both the led suit and the trump suit may a player discard any off-suit card.
