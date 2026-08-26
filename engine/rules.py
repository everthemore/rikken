"""
engine/rules.py — Legal action computation, void matrix updates, and Aasvragen logic.

This is the hottest module in the engine — it is called on every step during
ISMCTS rollouts. All operations are kept branchless / vectorized where possible.
"""

from __future__ import annotations
import numpy as np
from typing import List

from engine.card import (
    suit_of, rank_of, ACE_RANK, SUIT_MASKS, NUM_SUITS,
    cards_in_suit, highest_card_in_suit, lowest_card_in_suit, card_id,
)
from engine.state import RikkenState, Contract, Phase


# ---------------------------------------------------------------------------
# Legal actions — bidding phase
# ---------------------------------------------------------------------------

def legal_bids(state: RikkenState) -> List[int]:
    """
    Return list of legal bid Contract values for state.current_player.

    Rules:
      - A player who has already passed (passed[p]=True) has no legal actions.
      - Any player may PAS (Contract.PAS = 0) unless they have already passed.
      - Any player may bid any Contract value strictly greater than state.highest_bid.
      - Multi-player contracts (Piek, Misere, Open Piek, Open Misere) can also be
        co-bid at the SAME level as state.highest_bid.
      - TROELA is only available if the player holds 3 or more Aces.

    Returns:
        List of int (Contract values).
    """
    p = state.current_player
    if state.passed[p]:
        return []

    legal = [int(Contract.PAS)]

    hand = state.hands[p]
    ace_count = int(np.sum(hand[12::13]))

    for contract in Contract:
        if contract <= Contract.PAS:
            continue
        if contract == Contract.TROELA:
            if ace_count < 3:
                continue

        c_int = int(contract)
        high_int = int(state.highest_bid)

        # Strictly higher contract is always eligible
        if c_int > high_int:
            legal.append(c_int)
        # Simultaneous co-bidding allowed for Misere / Piek
        elif c_int == high_int and Contract.is_multi_player_allowed(contract):
            # Only if this player hasn't already bid it
            if state.bids[p] != c_int:
                legal.append(c_int)

    return legal


# ---------------------------------------------------------------------------
# Legal actions — trick-taking phase
# ---------------------------------------------------------------------------

def legal_plays(state: RikkenState) -> np.ndarray:
    """
    Return a boolean mask (int8[52]) of legal cards to play for state.current_player.

    Priority:
      1. Aasvragen override: if triggered and player is the partner -> only vraagaas card.
      2. Follow suit if possible.
      3. Must trump if void in led suit and holds trump.
      4. Free discard if void in both led suit and trump.

    For the trick leader (first card of trick): all cards in hand are legal.
    """
    p = state.current_player
    hand = state.hands[p]
    trump = state.trump_suit

    # --- Aasvragen override ---
    if (state.aasvragen_triggered
            and p == state.partner
            and state.vraagaas_suit >= 0
            and not state.partner_revealed):
        vraagaas_card = card_id(state.vraagaas_suit, ACE_RANK)
        if hand[vraagaas_card]:
            mask = np.zeros(52, dtype=np.int8)
            mask[vraagaas_card] = 1
            return mask

    # --- Trick leader: all cards legal ---
    if all(c == -1 for c in state.current_trick):
        return hand.copy()

    # --- Determine led suit ---
    led_card = state.current_trick[state.trick_leader]
    if led_card < 0:
        return hand.copy()
    led_suit = suit_of(led_card)

    # --- Priority 2: Follow suit ---
    follow_cards = hand & SUIT_MASKS[led_suit]
    if np.any(follow_cards):
        return follow_cards.astype(np.int8)

    # --- Priority 3: Must trump ---
    if trump >= 0:
        trump_cards = hand & SUIT_MASKS[trump]
        if np.any(trump_cards):
            return trump_cards.astype(np.int8)

    # --- Priority 4: Free discard ---
    return hand.copy()


# ---------------------------------------------------------------------------
# Void matrix inference update
# ---------------------------------------------------------------------------

def update_void_matrix(
    state: RikkenState,
    player: int,
    card_played: int,
    led_suit: Optional[int] = None,
) -> None:
    """
    Update state.void_matrix in-place based on a card played by `player`.
    """
    if led_suit is None:
        if player == state.trick_leader:
            return
        led_card = state.current_trick[state.trick_leader]
        if led_card < 0:
            return
        led_suit = suit_of(led_card)

    played_suit = suit_of(card_played)
    trump = state.trump_suit

    if played_suit != led_suit:
        state.void_matrix[player, led_suit] = True

        if trump >= 0 and played_suit != trump:
            state.void_matrix[player, trump] = True


# ---------------------------------------------------------------------------
# Aasvragen trigger detection
# ---------------------------------------------------------------------------

def check_and_trigger_aasvragen(
    state: RikkenState,
    leader: int,
    card_led: int,
) -> None:
    """
    Check whether `leader` leading `card_led` triggers the one-time Aasvragen restriction.
    """
    if not Contract.is_partner_contract(state.contract):
        return
    if state.contract == Contract.TROELA:
        return
    if state.aasvragen_triggered:
        return
    if state.vraagaas_suit < 0:
        return
    if leader != state.declarer:
        return

    if suit_of(card_led) == state.vraagaas_suit:
        state.aasvragen_triggered = True
        state.void_matrix[leader, state.vraagaas_suit] = False


# ---------------------------------------------------------------------------
# Troela partner identification
# ---------------------------------------------------------------------------

def find_troela_partner(state: RikkenState, declarer: int) -> int:
    """
    In Troela, declarer holds 3 Aces. The player holding the 4th Ace is partner.
    """
    for p in range(4):
        if p == declarer:
            continue
        aces_held = int(np.sum(state.hands[p][12::13]))
        if aces_held >= 1:
            return p
    return (declarer + 2) % 4
