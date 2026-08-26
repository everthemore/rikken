# 02. Human Heuristics & Strategy Guide

This document captures the classic strategic principles of Rikken and how they are codified into the Phase 1 Rule-Based Heuristic Agent.

---

## 1. Hand Evaluation & Bidding Strategy

### 1.1 Point-Count & Distributional Strength
Human players evaluate hands based on high card points (HCP) combined with suit length bonuses:
- **High Cards**: Ace = 4 pts, King = 3 pts, Queen = 2 pts, Jack = 1 pt.
- **Long Suit Bonus**: +2 pts for each card beyond the 4th in any suit (5-card suit = +2, 6-card = +4, 7-card = +6).
- **Void Bonus**: +1 pt for complete void in a side suit.

### 1.2 Bidding Thresholds
- **Rik** (Strength $\ge 9$, $\ge 1$ Ace): Look for 1 long suit (5+ cards) and a weakness in an unheld Ace suit to call partner.
- **8 Alleen** (Strength $\ge 11$, $\ge 1$ Ace): Strong trump suit (5+ cards with A/K) and side-suit stoppers.
- **9 Alleen to 12 Alleen**: Progressively higher requirements (13, 15, 17, 19 pts) with 2+ Aces and 6+ trumps.
- **Solo Slim (13)** (Strength $\ge 22$, $\ge 3$ Aces): Unstoppable trump chain and all high side-suit bosses.
- **Misère**: Requires zero Aces, no face cards (J, Q, K) in $\ge 3$ suits, and safe low exits (2s, 3s, 4s).
- **Piek**: Very skewed distribution — 1 ultra-long suit (7+ cards) and singletons in other suits, allowing controlled trick taking (targeting exactly 1 or 5).

---

## 2. Trick-Taking Tactics

### 2.1 Leading Strategy (Declarer)
1. **Drawing Trump**: When holding trump majority, lead high trumps early to strip opponents of their trumps and prevent them from ruffing side winners.
2. **Cashing Aces**: Cash side-suit Aces while opponents are still likely to follow suit.
3. **Establishing Long Suits**: Lead from the longest side suit to force out high defender cards, turning low cards into winners once opponents are void.

### 2.2 Following & Defending Strategy
1. **Win Cheaply**: When attempting to win a trick, play the lowest winning card necessary (e.g. if the 10 is winning, play Jack rather than Ace).
2. **Duck for Partner**: If your partner is currently winning the trick, play your lowest card (or slough a worthless loser) to conserve power.
3. **Must-Trump Exploitation**: Defenders can lead suits in which Declarer is known to be void to force them to spend high trumps ("forcing game").

### 2.3 Misère Tactics
- Declarer always leads their lowest card.
- When following, declarer plays the highest card that is still strictly lower than the current trick maximum.
- Defenders lead suits in which Declarer is longest to trap their high-middle cards (7s, 8s, 9s).

### 2.4 Piek Tactics
- Target 1: Win early with a high boss card, then duck every subsequent trick.
- Target 5: If forced past 1 trick, aggressively play to win exactly 5 tricks by establishing long suit control, then slough remaining cards.
