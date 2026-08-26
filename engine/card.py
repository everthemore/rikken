"""
engine/card.py — Card constants, encoding, and helper functions.

Card identity is a single int8 in [0, 51]:
    card_id = suit_index * 13 + rank_index

Suits:  0=Clubs (C), 1=Diamonds (D), 2=Hearts (H), 3=Spades (S)
Ranks:  0=2, 1=3, ..., 11=King, 12=Ace
"""

from __future__ import annotations
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Suit / Rank lookup tables
# ---------------------------------------------------------------------------
SUIT_NAMES = ['C', 'D', 'H', 'S']          # Clubs, Diamonds, Hearts, Spades
SUIT_FULL  = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
RANK_NAMES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

NUM_SUITS = 4
NUM_RANKS = 13
NUM_CARDS = 52
NUM_PLAYERS: int = 4
HEARTS_SUIT: int = 2

# Pre-built index arrays (avoids repeated computation in hot loops)
_SUIT_OF = np.array([c // 13 for c in range(52)], dtype=np.int8)  # card → suit
_RANK_OF = np.array([c % 13  for c in range(52)], dtype=np.int8)  # card → rank

# Suit mask arrays:  suit_mask[s] is a bool[52] with True for all cards of suit s
SUIT_MASKS: list[np.ndarray] = [
    np.array([c // 13 == s for c in range(52)], dtype=bool)
    for s in range(NUM_SUITS)
]

# Rank mask arrays: rank_mask[r] is a bool[52] with True for all cards of rank r
RANK_MASKS: list[np.ndarray] = [
    np.array([c % 13 == r for c in range(52)], dtype=bool)
    for r in range(NUM_RANKS)
]

# Ordered deck (standard order, used as the reset baseline)
FULL_DECK: np.ndarray = np.arange(52, dtype=np.int8)

# ACE rank index and ACE card IDs per suit
ACE_RANK: int = 12
ACES: np.ndarray = np.array([s * 13 + ACE_RANK for s in range(NUM_SUITS)], dtype=np.int8)


# ---------------------------------------------------------------------------
# Inline helpers (all operate on scalar ints or NumPy int8 arrays)
# ---------------------------------------------------------------------------

def suit_of(card: int) -> int:
    """Return the suit index (0-3) of a card id."""
    return int(_SUIT_OF[card])


def rank_of(card: int) -> int:
    """Return the rank index (0-12, where 12=Ace) of a card id."""
    return int(_RANK_OF[card])


def card_id(suit: int, rank: int) -> int:
    """Construct a card id from suit and rank indices."""
    return int(suit * 13 + rank)


def card_to_str(card: int) -> str:
    """Human-readable representation, e.g. 'A♠', 'T♥', '2♣'."""
    suit_symbols = ['♣', '♦', '♥', '♠']
    rank_str = RANK_NAMES[rank_of(card)]
    return f"{rank_str}{suit_symbols[suit_of(card)]}"


def str_to_card(s: str) -> int:
    """Parse a card string like 'A♠' or 'AS' or '10H' into a card id.

    Accepts Unicode suit symbols (♣♦♥♠) or letter codes (CDHS), case-insensitive.
    """
    s = s.strip()
    suit_map = {'♣': 0, '♦': 1, '♥': 2, '♠': 3,
                'C': 0, 'D': 1, 'H': 2, 'S': 3,
                'c': 0, 'd': 1, 'h': 2, 's': 3}
    # last character is the suit
    suit_char = s[-1]
    rank_str  = s[:-1].upper()
    if suit_char not in suit_map:
        raise ValueError(f"Unknown suit character '{suit_char}' in '{s}'")
    suit = suit_map[suit_char]
    rank_map = {r: i for i, r in enumerate(RANK_NAMES)}
    rank_map['T'] = rank_map['10']  # convenience alias
    if rank_str not in rank_map:
        raise ValueError(f"Unknown rank '{rank_str}' in '{s}'")
    return card_id(suit, rank_map[rank_str])


def hand_to_str(hand_vec: np.ndarray) -> str:
    """Convert a one-hot int8[52] hand vector to a sorted human-readable string."""
    cards = np.where(hand_vec)[0]
    return ' '.join(card_to_str(int(c)) for c in sorted(cards))


def cards_in_suit(hand_vec: np.ndarray, suit: int) -> np.ndarray:
    """Return indices of cards in `hand_vec` that belong to `suit`."""
    mask = hand_vec.astype(bool) & SUIT_MASKS[suit]
    return np.where(mask)[0].astype(np.int8)


def highest_card_in_suit(hand_vec: np.ndarray, suit: int) -> Optional[int]:
    """Return the highest card (by rank) of `suit` in `hand_vec`, or None."""
    candidates = cards_in_suit(hand_vec, suit)
    if len(candidates) == 0:
        return None
    return int(candidates[np.argmax(_RANK_OF[candidates])])


def lowest_card_in_suit(hand_vec: np.ndarray, suit: int) -> Optional[int]:
    """Return the lowest card (by rank) of `suit` in `hand_vec`, or None."""
    candidates = cards_in_suit(hand_vec, suit)
    if len(candidates) == 0:
        return None
    return int(candidates[np.argmin(_RANK_OF[candidates])])


def beats(attacker: int, defender: int, trump_suit: int, led_suit: int) -> bool:
    """Return True if `attacker` beats `defender` given trump and led suit context.

    Rules:
      - Any trump beats any non-trump.
      - Among same suit, higher rank wins.
      - Non-trump, non-led-suit cards never win (return False).
    """
    a_suit, a_rank = suit_of(attacker), rank_of(attacker)
    d_suit, d_rank = suit_of(defender), rank_of(defender)

    a_is_trump = (a_suit == trump_suit)
    d_is_trump = (d_suit == trump_suit)

    if a_is_trump and not d_is_trump:
        return True
    if d_is_trump and not a_is_trump:
        return False
    if a_suit == d_suit:
        return a_rank > d_rank
    # attacker is off-suit, off-trump → never beats
    return False


def trick_winner(trick: np.ndarray, leader: int, trump_suit: int) -> int:
    """Return the player index who wins the trick.

    Args:
        trick:      int8[4] — card played by each player (ordered by seat).
                    A value of -1 means that player has not yet played.
        leader:     Seat index of the player who led the trick.
        trump_suit: Suit index of the trump suit (-1 if no trump).

    Returns:
        Seat index (0-3) of the trick winner.
    """
    led_card = trick[leader]
    led_suit = suit_of(led_card)

    winning_seat = leader
    winning_card = led_card

    for offset in range(1, 4):
        seat = (leader + offset) % 4
        card = trick[seat]
        if card < 0:
            continue  # not yet played (shouldn't happen at resolution)
        if beats(card, winning_card, trump_suit, led_suit):
            winning_seat = seat
            winning_card = card

    return winning_seat
