"""
agents/heuristic.py — Rule-based heuristic agent for Rikken bootstrapping.

Implements standard Dutch Rikken bidding and trick-taking rules of thumb.
Used to bootstrap Phase 1 dataset generation and benchmark neural agents.
"""

from __future__ import annotations
import numpy as np
import logging
from typing import Optional

from engine.card import (
    NUM_SUITS, NUM_RANKS, ACE_RANK, HEARTS_SUIT,
    suit_of, rank_of, cards_in_suit, highest_card_in_suit,
    lowest_card_in_suit, SUIT_MASKS, card_id,
)
from engine.state import RikkenState, Contract, Phase
from engine.rules import legal_bids, legal_plays

log = logging.getLogger(__name__)

HCP_VALUES = {12: 4, 11: 3, 10: 2, 9: 1}  # A=4, K=3, Q=2, J=1


class HeuristicAgent:
    """
    Rule-based agent implementing human Rikken heuristics.
    """

    def __init__(
        self,
        seat: int,
        troela_call_rate: float = 0.70,
        rng: Optional[np.random.Generator] = None,
    ):
        self.seat = seat
        self.troela_call_rate = troela_call_rate
        self.rng = rng or np.random.default_rng()

    def set_seat(self, seat: int) -> None:
        self.seat = seat

    def act(self, state: RikkenState) -> int:
        if state.phase == Phase.BIDDING:
            return self._bid(state)
        elif state.phase == Phase.TRICK_TAKING:
            return self._play(state)
        else:
            raise RuntimeError(f"act() called in phase {state.phase}")

    def declare_trump(self, state: RikkenState) -> int:
        hand = state.hands[self.seat]
        if state.contract == Contract.RIK_BETER:
            return HEARTS_SUIT
        suit_scores = []
        for suit in range(NUM_SUITS):
            count = int(np.sum(hand & SUIT_MASKS[suit]))
            hcp = sum(HCP_VALUES.get(rank_of(c), 0) for c in np.where(hand & SUIT_MASKS[suit])[0])
            suit_scores.append(count * 2.0 + hcp)
        return int(np.argmax(suit_scores))

    def declare_vraagaas(self, state: RikkenState, trump_suit: int) -> int:
        hand = state.hands[self.seat]
        candidates = []
        for suit in range(NUM_SUITS):
            if suit == trump_suit:
                continue
            ace = card_id(suit, ACE_RANK)
            if not hand[ace]:
                candidates.append(suit)
        if not candidates:
            candidates = [s for s in range(NUM_SUITS) if s != trump_suit]
        return int(self.rng.choice(candidates))

    # -------------------------------------------------------------------------
    # Bidding heuristics
    # -------------------------------------------------------------------------
    def _bid(self, state: RikkenState) -> int:
        legal = legal_bids(state)
        if not legal or legal == [int(Contract.PAS)]:
            return int(Contract.PAS)

        p = self.seat
        hand = state.hands[p]
        ace_count = int(np.sum(hand[12::13]))

        # 1. Troela check (3 Aces)
        if Contract.TROELA in [Contract(b) for b in legal if b > 0]:
            if ace_count >= 3 and self.rng.random() < self.troela_call_rate:
                return int(Contract.TROELA)

        # 2. Hand evaluation
        strength, best_suit = self._evaluate_hand(hand)
        desired_bid = self._strength_to_bid(hand, strength, state)

        # 3. Choose the highest legal bid <= desired_bid
        best_legal = int(Contract.PAS)
        for b in legal:
            if b <= desired_bid:
                if b > best_legal:
                    best_legal = b
                elif b == best_legal and Contract.is_multi_player_allowed(Contract(b)):
                    best_legal = b

        return best_legal

    def _evaluate_hand(self, hand: np.ndarray) -> tuple[float, int]:
        hcp = sum(HCP_VALUES.get(rank_of(c), 0) for c in np.where(hand)[0])
        suit_lengths = [int(np.sum(hand & SUIT_MASKS[s])) for s in range(NUM_SUITS)]
        longest_suit = int(np.argmax(suit_lengths))
        max_length = suit_lengths[longest_suit]

        length_bonus = max(0, max_length - 4) * 1.5
        void_bonus = sum(2.0 for l in suit_lengths if l == 0)
        singleton_bonus = sum(1.0 for l in suit_lengths if l == 1)

        total_strength = hcp + length_bonus + void_bonus + singleton_bonus
        return total_strength, longest_suit

    def _strength_to_bid(
        self, hand: np.ndarray, strength: float, state: RikkenState
    ) -> int:
        ace_count = int(np.sum(hand[12::13]))
        hearts_count = int(np.sum(hand & SUIT_MASKS[HEARTS_SUIT]))

        # Check Misere
        if self._is_misere_hand(hand):
            return int(Contract.MISERE)

        # Check Piek (Misere with exactly 1 winner)
        if self._is_piek_hand(hand):
            return int(Contract.PIEK)

        # Strength ladder
        if strength >= 22 and ace_count >= 3:
            return int(Contract.SOLO_SLIM)
        elif strength >= 19 and ace_count >= 2:
            return int(Contract.TWAALF_ALLEEN)
        elif strength >= 17 and ace_count >= 2:
            return int(Contract.ELF_ALLEEN)
        elif strength >= 15 and ace_count >= 2:
            return int(Contract.TIEN_ALLEEN)
        elif strength >= 13 and ace_count >= 1:
            return int(Contract.NEGEN_ALLEEN)
        elif strength >= 11 and ace_count >= 1:
            return int(Contract.ACHT_ALLEEN)
        elif strength >= 9 and ace_count >= 1 and hearts_count >= 5:
            return int(Contract.RIK_BETER)
        elif strength >= 9 and ace_count >= 1:
            return int(Contract.RIK)
        else:
            return int(Contract.PAS)

    def _is_misere_hand(self, hand: np.ndarray) -> bool:
        for c in np.where(hand)[0]:
            r = rank_of(c)
            if r >= 10:  # Q, K, A
                return False
        suits_with_cards = sum(
            1 for suit in range(NUM_SUITS)
            if np.any(hand & SUIT_MASKS[suit])
        )
        return suits_with_cards >= 3

    def _is_piek_hand(self, hand: np.ndarray) -> bool:
        high_honors = 0
        middle_cards = 0
        for c in np.where(hand)[0]:
            r = rank_of(c)
            if r >= 10:  # Q, K, A
                high_honors += 1
            elif r >= 6: # 8, 9, J
                middle_cards += 1
        suits_with_cards = sum(
            1 for suit in range(NUM_SUITS)
            if np.any(hand & SUIT_MASKS[suit])
        )
        return (high_honors == 1) and (middle_cards <= 1) and (suits_with_cards >= 3)

    # -------------------------------------------------------------------------
    # Play heuristics
    # -------------------------------------------------------------------------
    def _play(self, state: RikkenState) -> int:
        legal_mask = legal_plays(state)
        legal_cards = np.where(legal_mask)[0]

        if len(legal_cards) == 1:
            return int(legal_cards[0])

        c = state.contract
        if c in (Contract.MISERE, Contract.OPEN_MISERE):
            return self._play_misere(state, legal_cards)
        elif c in (Contract.PIEK, Contract.OPEN_PIEK):
            return self._play_piek(state, legal_cards)
        else:
            return self._play_normal(state, legal_cards)

    def _play_normal(self, state: RikkenState, legal_cards: np.ndarray) -> int:
        p = self.seat
        is_leading = (state.trick_leader == p and state.current_trick[p] == -1)

        if is_leading:
            return self._lead_normal(state, legal_cards)
        else:
            return self._follow_normal(state, legal_cards)

    def _lead_normal(self, state: RikkenState, legal_cards: np.ndarray) -> int:
        p = self.seat
        trump = state.trump_suit
        hand = state.hands[p]

        # Declarer in solo/partner: draw trumps early
        if p == state.declarer and trump >= 0:
            trump_legal = [c for c in legal_cards if suit_of(c) == trump]
            if trump_legal:
                return int(max(trump_legal, key=lambda c: rank_of(c)))

        # Otherwise lead Aces in side suits
        aces = [c for c in legal_cards if rank_of(c) == ACE_RANK and suit_of(c) != trump]
        if aces:
            return int(aces[0])

        # Otherwise lead lowest card
        return int(min(legal_cards, key=lambda c: rank_of(c)))

    def _follow_normal(self, state: RikkenState, legal_cards: np.ndarray) -> int:
        return int(max(legal_cards, key=lambda c: rank_of(c)))

    def _play_misere(self, state: RikkenState, legal_cards: np.ndarray) -> int:
        # In Misere: always duck and play lowest available card
        return int(min(legal_cards, key=lambda c: rank_of(c)))

    def _play_piek(self, state: RikkenState, legal_cards: np.ndarray) -> int:
        p = self.seat
        is_decl = state.declarer_mask[p] if hasattr(state, 'declarer_mask') else (p == state.declarer)
        if is_decl:
            if state.tricks_won[p] == 0:
                # Need our 1 trick: play highest card
                return int(max(legal_cards, key=lambda c: rank_of(c)))
            else:
                # Already won our 1 trick: duck everything
                return int(min(legal_cards, key=lambda c: rank_of(c)))
        else:
            return int(min(legal_cards, key=lambda c: rank_of(c)))
