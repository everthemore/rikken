"""
engine/deck.py — Clumping shuffle and deal logic.

The shuffle is designed to mimic realistic human table-shuffling between games,
where tricks are collected in order and the resulting clumps are only partially
dispersed by a small number of riffle passes.

Clumping calibration:
  After tricks are collected, the deck consists of 13 consecutive groups of 4
  cards (each group = one trick, ordered by winner's collection). The deal then
  distributes them in 6-then-7 round-robin batches. A single riffle pass
  approximately halves the run-length of consecutive trick-clumps. Three passes
  (SHUFFLE_INTENSITY=3) leaves mild but realistic clustering — matching what one
  sees empirically when a dealer does 3 casual riffles.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from engine.card import NUM_CARDS
import config


# ---------------------------------------------------------------------------
# Deck reconstruction after a game
# ---------------------------------------------------------------------------

def reconstruct_deck_from_game(
    trick_sequence: Optional[list] = None,
    remaining_hands: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rebuild the physical deck order as it would exist after gathering tricks and hands.

    In authentic Rikken:
      1. Tricks won are collected in 4-card packets in trick-play order.
      2. If a hand terminates early (early stop or concession) or in all-pass redeals,
         players discard their remaining cards onto the pile. Players hold their cards
         sorted by suit and rank in hand, preserving natural suit clusters.
    """
    deck = []
    if trick_sequence:
        for _winner, cards in trick_sequence:
            deck.extend(cards)

    if remaining_hands is not None and len(deck) < NUM_CARDS:
        for p in range(4):
            held_cards = np.where(remaining_hands[p] == 1)[0]
            # Sort by suit first (0..3), then rank (0..12) as held in human hands
            held_sorted = sorted(held_cards, key=lambda c: (c // 13, c % 13))
            deck.extend(held_sorted)

    if len(deck) != NUM_CARDS:
        # Fallback: standard ordered deck (suits grouped together)
        return np.arange(NUM_CARDS, dtype=np.int8)
    return np.array(deck, dtype=np.int8)


def reconstruct_deck_from_tricks(trick_sequence: list) -> np.ndarray:
    """Backward-compatible wrapper for reconstruct_deck_from_game."""
    return reconstruct_deck_from_game(trick_sequence=trick_sequence)


# ---------------------------------------------------------------------------
# Riffle shuffle (imperfect, geometric bias)
# ---------------------------------------------------------------------------

def _riffle_pass(deck: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Perform one imperfect riffle shuffle pass.

    Algorithm:
      1. Split the deck near the middle with ±5-card jitter.
      2. Interleave the two halves using a geometric distribution to model
         thumb-release packets of 1–4 cards at a time (realistic finger pressure).

    This preserves some of the original run structure while adding randomness.
    """
    n = len(deck)
    mid = n // 2 + rng.integers(-5, 6)  # random cut near center
    mid = np.clip(mid, 8, n - 8)        # keep both halves non-trivial

    left  = deck[:mid].copy()
    right = deck[mid:].copy()

    result = []
    li, ri = 0, 0
    # Geometric distribution with p=0.5 → mean packet size ≈ 2 cards
    while li < len(left) and ri < len(right):
        packet_size = min(int(rng.geometric(0.5)), len(left) - li, len(right) - ri)
        # Randomly choose which half releases next
        if rng.random() < 0.5:
            result.extend(left[li:li + packet_size])
            li += packet_size
        else:
            result.extend(right[ri:ri + packet_size])
            ri += packet_size

    # Flush remaining cards
    result.extend(left[li:])
    result.extend(right[ri:])

    return np.array(result, dtype=np.int8)


def clumping_shuffle(
    prev_trick_sequence: Optional[list] = None,
    rng: Optional[np.random.Generator] = None,
    shuffle_intensity: int = config.SHUFFLE_INTENSITY,
    prev_remaining_hands: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Shuffle the deck with realistic clumping from the previous game's tricks.

    Steps:
      1. Reconstruct the deck from previous trick-collection order (clumped).
         If no previous game exists, start from a standard ordered deck.
      2. Apply `shuffle_intensity` riffle passes.
      3. Apply a random cut.

    Args:
        prev_trick_sequence: Output of `state.trick_sequence` from the last game,
                             or None for the very first game.
        rng:                 NumPy random generator (seeded externally for reproducibility).
        shuffle_intensity:   Number of riffle passes (3 = light shuffle, 7 = well shuffled).

    Returns:
        np.ndarray of shape (52,) — shuffled card order ready for dealing.
    """
    if rng is None:
        rng = np.random.default_rng()

    deck = reconstruct_deck_from_game(
        trick_sequence=prev_trick_sequence,
        remaining_hands=prev_remaining_hands,
    )

    for _ in range(shuffle_intensity):
        deck = _riffle_pass(deck, rng)

    # Random cut
    cut_point = rng.integers(1, NUM_CARDS)
    deck = np.roll(deck, -cut_point)

    return deck


# ---------------------------------------------------------------------------
# Deal
# ---------------------------------------------------------------------------

def deal(deck: np.ndarray) -> np.ndarray:
    """
    Deal 52 cards to 4 players in round-robin batches of 6 then 7.

    Round 1 (6 cards each, positions 0-23):
        P0 ← deck[0:6], P1 ← deck[6:12], P2 ← deck[12:18], P3 ← deck[18:24]
    Round 2 (7 cards each, positions 24-51):
        P0 ← deck[24:31], P1 ← deck[31:38], P2 ← deck[38:45], P3 ← deck[45:52]

    Args:
        deck: np.ndarray of shape (52,) — shuffled card order.

    Returns:
        hands: np.ndarray of shape (4, 52), dtype int8
               hands[p, c] = 1 iff player p holds card c.
    """
    hands = np.zeros((4, 52), dtype=np.int8)

    # Round 1: 6 cards each
    for p in range(4):
        start = p * 6
        for card in deck[start:start + 6]:
            hands[p, card] = 1

    # Round 2: 7 cards each
    offset = 24
    for p in range(4):
        start = offset + p * 7
        for card in deck[start:start + 7]:
            hands[p, card] = 1

    # Sanity check: each player has exactly 13 cards
    assert np.all(hands.sum(axis=1) == 13), \
        f"Deal error: card counts = {hands.sum(axis=1).tolist()}"
    assert hands.sum() == 52, "Deal error: total cards ≠ 52"

    return hands
