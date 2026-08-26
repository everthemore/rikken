"""
engine/early_stop.py — Basic and advanced early stopping for ISMCTS rollouts.

Two tiers of pruning:
  1. Basic: Threshold checks on trick counts after every trick.
  2. Advanced: Public-determinism check using the Void Matrix — a sound
     lower-bound on guaranteed future tricks.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from engine.card import suit_of, rank_of, SUIT_MASKS, NUM_SUITS, card_id
from engine.state import RikkenState, Contract


# ---------------------------------------------------------------------------
# Basic Early Stopping
# ---------------------------------------------------------------------------

def check_basic_early_stop(state: RikkenState) -> Optional[float]:
    """
    Check simple trick-count thresholds that guarantee the contract outcome.

    Called after each completed trick.

    Returns:
        +1.0 if Declarer wins early, -1.0 if Declarer loses early, None otherwise.
        All values are from the Declarer's perspective.
    """
    c = state.contract
    d_tricks = state.declarer_tricks
    def_tricks = state.defender_tricks
    remaining = state.remaining_tricks

    # ---- Rik / Rik Beter / Partner contracts (target 8+) ----
    if c in (Contract.RIK, Contract.RIK_BETER, Contract.TROELA):
        if d_tricks >= 8:
            return +1.0
        if def_tricks >= 6:
            return -1.0

    # ---- Alleen contracts ----
    elif c == Contract.ACHT_ALLEEN:
        if d_tricks >= 8:
            return +1.0
        if def_tricks >= 6:
            return -1.0

    elif c == Contract.NEGEN_ALLEEN:
        if d_tricks >= 9:
            return +1.0
        if def_tricks >= 5:
            return -1.0

    elif c == Contract.TIEN_ALLEEN:
        if d_tricks >= 10:
            return +1.0
        if def_tricks >= 4:
            return -1.0

    elif c == Contract.ELF_ALLEEN:
        if d_tricks >= 11:
            return +1.0
        if def_tricks >= 3:
            return -1.0

    elif c == Contract.TWAALF_ALLEEN:
        if d_tricks >= 12:
            return +1.0
        if def_tricks >= 2:
            return -1.0

    elif c == Contract.SOLO_SLIM:
        if d_tricks == 13:
            return +1.0
        if def_tricks >= 1:
            return -1.0

    # ---- Misere / Open Misere: win exactly 0 tricks ----
    elif c in (Contract.MISERE, Contract.OPEN_MISERE):
        if np.any(state.declarer_mask):
            # Check if all active declarers have lost (taken >= 1 trick)
            active_decls = np.where(state.declarer_mask)[0]
            if all(state.tricks_won[p] >= 1 for p in active_decls):
                return -1.0
        else:
            if d_tricks >= 1:
                return -1.0
        if remaining == 0:
            return +1.0

    # ---- Piek / Open Piek: win exactly 1 trick ----
    elif c in (Contract.PIEK, Contract.OPEN_PIEK):
        if np.any(state.declarer_mask):
            active_decls = np.where(state.declarer_mask)[0]
            if all(state.tricks_won[p] >= 2 for p in active_decls):
                return -1.0
        else:
            if d_tricks >= 2:
                # Overshot 1 trick target
                return -1.0
        if remaining == 0:
            return +1.0 if (d_tricks == 1) else -1.0

    return None


# ---------------------------------------------------------------------------
# Advanced Early Stopping (Public Determinism)
# ---------------------------------------------------------------------------

def _count_free_tricks(
    state: RikkenState,
    seat: int,
    remaining_cards: np.ndarray,
) -> int:
    free_tricks = 0
    trump = state.trump_suit
    hand = state.hands[seat]

    if trump >= 0:
        trump_held = np.where(hand & SUIT_MASKS[trump])[0]
        if len(trump_held) > 0:
            other_trumps = np.where(remaining_cards & SUIT_MASKS[trump])[0]
            if len(other_trumps) == 0:
                free_tricks += len(trump_held)

    for s in range(NUM_SUITS):
        if s == trump:
            continue
        suit_held = np.where(hand & SUIT_MASKS[s])[0]
        if len(suit_held) == 0:
            continue
        others_in_suit = np.where(remaining_cards & SUIT_MASKS[s])[0]
        if len(others_in_suit) == 0:
            if trump < 0:
                free_tricks += len(suit_held)
            else:
                defenders_with_trump = [
                    p for p in range(4)
                    if p != seat and p != state.partner and not state.void_matrix[p, trump]
                ]
                if len(defenders_with_trump) == 0:
                    free_tricks += len(suit_held)

    return free_tricks


def check_advanced_early_stop(state: RikkenState) -> Optional[float]:
    basic_res = check_basic_early_stop(state)
    if basic_res is not None:
        return basic_res

    c = state.contract
    target = Contract.target_tricks(c)
    if target is None or target <= 0:
        return None

    if c in (Contract.PIEK, Contract.OPEN_PIEK, Contract.MISERE, Contract.OPEN_MISERE):
        return None

    current_cards = np.zeros(52, dtype=bool)
    for card in state.current_trick:
        if card >= 0:
            current_cards[card] = True
    all_seen = state.played_cards.astype(bool) | current_cards

    decl = state.declarer
    if decl < 0:
        return None

    unseen_by_decl = ~all_seen & ~state.hands[decl].astype(bool)
    decl_free = _count_free_tricks(state, decl, unseen_by_decl)

    partner_free = 0
    if state.partner >= 0 and state.partner != decl and state.partner_revealed:
        unseen_by_partner = ~all_seen & ~state.hands[state.partner].astype(bool)
        partner_free = _count_free_tricks(state, state.partner, unseen_by_partner)

    total_guaranteed_decl = state.declarer_tricks + decl_free + partner_free
    if total_guaranteed_decl >= target:
        return +1.0

    return None


def should_early_stop(state: RikkenState) -> Tuple[bool, Optional[float]]:
    res = check_advanced_early_stop(state)
    if res is not None:
        return True, res
    return False, None
