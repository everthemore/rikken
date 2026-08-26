# 01. Complete Game Rules & Custom Mechanics

*Rikken* is a classic Dutch 4-player trick-taking card game with imperfect information, hidden partnerships, and a rich bidding hierarchy. This document details the exact rules implemented in this engine, including all custom house rules.

---

## 1. Deck, Dealing & Physical Clumping

### 1.1 The Deck
- Standard 52-card international deck (no jokers).
- **Suits**: Clubs (♣), Diamonds (♦), Hearts (♥), Spades (♠).
- **Ranks** (in ascending order): `2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace`.
- Cards are encoded as integers $c \in [0, 51]$ with $c = 13 	imes 	ext{suit} + 	ext{rank}$.

### 1.2 The Dealing Protocol (6-then-7 Round Robin)
Dealing is performed strictly in round-robin fashion over two distinct rounds:
1. **Round 1 (6 cards each)**:
   - Player 0 receives deck indices `[0:6]`
   - Player 1 receives deck indices `[6:12]`
   - Player 2 receives deck indices `[12:18]`
   - Player 3 receives deck indices `[18:24]`
2. **Round 2 (7 cards each)**:
   - Player 0 receives deck indices `[24:31]`
   - Player 1 receives deck indices `[31:38]`
   - Player 2 receives deck indices `[38:45]`
   - Player 3 receives deck indices `[45:52]`

### 1.3 Clumping Shuffle Mechanism
In real-world play, cards from previous tricks remain clumped in groups of 4. A human dealer rarely shuffles more than 2 to 4 times. 
- The engine reconstructs the deck directly from `trick_sequence` of the previous game.
- It applies `SHUFFLE_INTENSITY = 3` imperfect riffle passes using a geometric distribution ($\mu pprox 2$ cards dropped per thumb release) with cut jitter ($\pm 5$ cards).
- This produces realistic card clumping distributions essential for the Belief Network to learn real-world Bayesian hand inference.

---

## 2. Bidding Phase & Hierarchy

Bidding proceeds clockwise starting from Player 0. A player may either **Pass** (`PAS`) or submit a bid higher than the current highest bid.
- Once a player passes, they cannot bid again in that hand (`passed[p] = True`).
- If all 4 players pass, the hand is a **Redeal** with reward `0.0`.

### 2.1 Bidding Hierarchy (Ascending)

| Level | Contract Name | Team / Solo | Trump | Goal / Win Condition | Notes |
|:---:|:---|:---:|:---:|:---|:---|
| 0 | `PAS` | - | - | Pass (no action) | |
| 1 | `RIK` | Partner (called Ace) | Yes (Declarer choice) | Win $\ge 8$ tricks | Secret partner holding Vraagaas |
| 2 | `ACHT_ALLEEN` (8 Alleen) | Solo | Yes (Declarer choice) | Win $\ge 8$ tricks | 1 vs 3 |
| 3 | `PIEK` (Custom Rule) | Solo | No trump | Win **exactly 1** OR **exactly 5** tricks | `win = (tricks==1) or (tricks==5)` |
| 4 | `NEGEN_ALLEEN` (9 Alleen) | Solo | Yes (Declarer choice) | Win $\ge 9$ tricks | 1 vs 3 |
| 5 | `MISERE` | Solo | No trump | Win **exactly 0** tricks | Loss if declarer wins $\ge 1$ trick |
| 6 | `TIEN_ALLEEN` (10 Alleen) | Solo | Yes (Declarer choice) | Win $\ge 10$ tricks | 1 vs 3 |
| 7 | `OPEN_PIEK` | Solo | No trump | Win **exactly 1** OR **5** tricks | Hand exposed after Trick 1 |
| 8 | `OPEN_MISERE` | Solo | No trump | Win **exactly 0** tricks | Hand exposed after Trick 1 |
| 9 | `ELF_ALLEEN` (11 Alleen) | Solo | Yes (Declarer choice) | Win $\ge 11$ tricks | 1 vs 3 |
| 10 | `TWAALF_ALLEEN` (12 Alleen) | Solo | Yes (Declarer choice) | Win $\ge 12$ tricks | 1 vs 3 |
| 11 | `SOLO_SLIM` (13 Alleen) | Solo | Yes (Declarer choice) | Win **all 13** tricks | Loss if defenders take 1 trick |
| 12 | `TROELA` (Special) | Partner (4th Ace) | Partner choice | Win $\ge 8$ tricks | Requires holding $\ge 3$ Aces |

### 2.2 Special Contract Rules
1. **Troela**:
   - Optional call when holding 3 (or 4) Aces.
   - The holder of the 4th Ace becomes the partner.
   - The partner dictates the trump suit, which **cannot be the suit of the 4th Ace**.
   - Team must win $\ge 8$ tricks.
2. **Rik & Aasvragen ("Vraagaas")**:
   - The declarer calls for an Ace they do not hold (`vraagaas_suit`).
   - The holder of this Ace is the secret partner.
   - **Aasvragen Trigger Rule**: When the Declarer *leads* the `vraagaas_suit` in a trick, Aasvragen is triggered. The partner is then strictly locked (`legal_plays` restricted) to playing that Ace on that trick. Once played, the lock is released.

---

## 3. Trick-Taking Phase & Play Rules

1. **Follow Suit (Mandatory)**: If a player holds cards of the led suit, they must play one of them.
2. **Must-Trump Rule (Custom House Rule)**:
   - If a player cannot follow suit, and they hold one or more cards of the trump suit, **they MUST play a trump card**.
   - A player may only discard an off-suit, non-trump card if they are void in **both** the led suit and the trump suit.
3. **Trick Resolution**:
   - The highest trump card played wins the trick.
   - If no trump is played, the highest card of the led suit wins.
   - The winner of the trick leads the next trick.
4. **Open Contracts**:
   - In `OPEN_PIEK` and `OPEN_MISERE`, the declarer's remaining cards are publicly revealed (`cards_face_up`) immediately following the completion of Trick 1.

---

## 4. Payoff & Objectives
- Payoffs are strictly binary zero-sum:
  - Declarer team win: $+1.0$ (defenders $-1.0$)
  - Declarer team loss: $-1.0$ (defenders $+1.0$)
  - Redeal: $0.0$
- Overtricks / undertricks do not grant extra score.
