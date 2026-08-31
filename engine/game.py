"""
engine/game.py — RikkenGame orchestrator and core game-loop interface.

Implements the standard environment step() interface:
    state = game.reset()
    state, reward = game.step(state, action)
"""

from __future__ import annotations
import numpy as np
import logging
from typing import Optional, Tuple, List

from engine.card import (
    NUM_CARDS, NUM_SUITS, NUM_PLAYERS, ACE_RANK, HEARTS_SUIT,
    suit_of, rank_of, trick_winner, card_id,
)
from engine.deck import deal, clumping_shuffle
from engine.state import RikkenState, Contract, Phase
from engine.rules import (
    legal_bids, legal_plays, update_void_matrix,
    check_and_trigger_aasvragen, find_troela_partner,
)
from engine.early_stop import should_early_stop

log = logging.getLogger(__name__)


class RikkenGame:
    """
    Stateful engine wrapper managing deal, turn progression, and reward calculation.
    """

    def __init__(
        self,
        shuffle_intensity: int = 3,
        use_early_stop: bool = True,
        troela_call_rate: float = 0.70,
        rng: Optional[np.random.Generator] = None,
    ):
        self.shuffle_intensity = shuffle_intensity
        self.use_early_stop = use_early_stop
        self.troela_call_rate = troela_call_rate
        self.rng = rng or np.random.default_rng()
        self._prev_trick_seq: Optional[list] = None

    def reset(self, seed: Optional[int] = None) -> RikkenState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        deck = clumping_shuffle(
            self._prev_trick_seq,
            self.rng,
            self.shuffle_intensity,
        )
        hands = deal(deck)

        state = RikkenState.initial()
        state.hands = hands
        return state

    def legal_actions(self, state: RikkenState) -> List[int] | np.ndarray:
        if state.phase == Phase.BIDDING:
            return legal_bids(state)
        elif state.phase == Phase.TRICK_TAKING:
            return legal_plays(state)
        return []

    def step(
        self, state: RikkenState, action: int
    ) -> Tuple[RikkenState, Optional[float]]:
        s = state.copy()

        if s.phase == Phase.BIDDING:
            return self._step_bidding(s, action)
        elif s.phase == Phase.TRICK_TAKING:
            return self._step_trick(s, action)
        else:
            raise RuntimeError("step() called on a terminal state.")

    def _step_bidding(
        self, s: RikkenState, action: int
    ) -> Tuple[RikkenState, Optional[float]]:
        p = s.current_player
        contract = Contract(action)

        if contract == Contract.PAS:
            s.passed[p] = True
            s.bids[p] = int(Contract.PAS)
            s.pass_count += 1

            if s.pass_count == 4 or np.all(s.passed):
                s.phase = Phase.TERMINAL
                s.reward = 0.0
                s.rewards = np.zeros(4, dtype=np.float32)
                return s, 0.0
        else:
            s.bids[p] = int(contract)
            s.highest_bid = contract
            if s.bid_winner < 0 or int(contract) > int(s.bids[s.bid_winner]):
                s.bid_winner = p
            s.pass_count = 0

        s.current_player = self._next_bidder(s)

        if self._bidding_finished(s):
            return self._resolve_bidding(s)

        return s, None

    def _next_bidder(self, s: RikkenState) -> int:
        p = (s.current_player + 1) % 4
        for _ in range(4):
            if not s.passed[p]:
                return p
            p = (p + 1) % 4
        return s.current_player

    def _bidding_finished(self, s: RikkenState) -> bool:
        if s.highest_bid <= Contract.PAS:
            return np.all(s.passed)

        # Count active players who haven't passed
        active = [p for p in range(4) if not s.passed[p]]
        if len(active) <= 1:
            return True

        # If highest bid is multi-player allowed (Misere / Piek)
        if Contract.is_multi_player_allowed(s.highest_bid):
            # Finished if every player has either bid the highest contract or passed
            for p in range(4):
                if not s.passed[p] and s.bids[p] != int(s.highest_bid):
                    return False
            return True

        return False

    def _resolve_bidding(
        self, s: RikkenState
    ) -> Tuple[RikkenState, Optional[float]]:
        s.contract = s.highest_bid
        s.declarer = s.bid_winner
        s.declarer_mask = np.zeros(4, dtype=bool)

        if Contract.is_multi_player_allowed(s.contract):
            for p in range(4):
                if s.bids[p] == int(s.contract):
                    s.declarer_mask[p] = True
        else:
            s.declarer_mask[s.declarer] = True

        if s.contract in (Contract.RIK, Contract.RIK_BETER):
            s.partner = -1
            if s.contract == Contract.RIK_BETER:
                s.trump_suit = HEARTS_SUIT  # Hearts is fixed trump for Rik Beter
        elif s.contract == Contract.TROELA:
            s.partner = find_troela_partner(s, s.declarer)

        s.phase = Phase.TRICK_TAKING
        s.current_player = s.declarer
        s.trick_leader = s.declarer
        return s, None

    def declare(
        self,
        state: RikkenState,
        trump_suit: int = -1,
        vraagaas_suit: int = -1,
    ) -> RikkenState:
        s = state.copy()
        if s.contract == Contract.RIK_BETER:
            s.trump_suit = HEARTS_SUIT
        else:
            s.trump_suit = trump_suit
        s.vraagaas_suit = vraagaas_suit

        if Contract.is_open(s.contract):
            for p in range(4):
                if s.declarer_mask[p]:
                    s.cards_face_up[p] = s.hands[p].copy()

        if s.contract in (Contract.RIK, Contract.RIK_BETER) and vraagaas_suit >= 0:
            ace_card = card_id(vraagaas_suit, ACE_RANK)
            for p in range(4):
                if p != s.declarer and s.hands[p, ace_card]:
                    s.partner = p
                    break
        return s

    def _step_trick(
        self, s: RikkenState, card: int
    ) -> Tuple[RikkenState, Optional[float]]:
        p = s.current_player

        if not s.hands[p, card]:
            raise ValueError(f"Player {p} attempted to play card {card} but doesn't hold it.")

        if s.trick_count == 0 and len(s.current_trick[s.current_trick >= 0]) == 0:
            check_and_trigger_aasvragen(s, p, card)

        update_void_matrix(s, p, card)

        s.hands[p, card] = 0
        s.played_cards[card] = 1
        s.current_trick[p] = card

        if (s.contract in (Contract.RIK, Contract.RIK_BETER)
                and not s.partner_revealed
                and s.vraagaas_suit >= 0):
            vraagaas_card = card_id(s.vraagaas_suit, ACE_RANK)
            if card == vraagaas_card:
                s.partner = p
                s.partner_revealed = True

        if np.all(s.current_trick >= 0):
            return self._resolve_trick(s)

        s.current_player = (p + 1) % 4
        return s, None

    def _resolve_trick(
        self, s: RikkenState
    ) -> Tuple[RikkenState, Optional[float]]:
        winner = trick_winner(
            s.current_trick,
            s.trick_leader,
            s.trump_suit,
        )

        s.tricks_won[winner] += 1
        s.trick_count += 1

        trick_cards = s.current_trick.copy()
        s.trick_sequence.append((winner, trick_cards))

        s.current_trick[:] = -1
        s.trick_leader = winner
        s.current_player = winner

        if Contract.is_open(s.contract) and s.trick_count == 1:
            for p in range(4):
                if s.declarer_mask[p]:
                    s.cards_face_up[p] = s.hands[p].copy()

        if self.use_early_stop:
            early_terminal, early_reward = should_early_stop(s)
            if early_terminal:
                s.phase = Phase.TERMINAL
                s.reward = early_reward
                self._compute_rewards_vector(s)
                self._prev_trick_seq = s.trick_sequence
                return s, s.reward

        if s.trick_count == 13:
            s.phase = Phase.TERMINAL
            s.reward = self._compute_final_reward(s)
            self._compute_rewards_vector(s)
            self._prev_trick_seq = s.trick_sequence
            return s, s.reward

        return s, None

    def _compute_final_reward(self, s: RikkenState) -> float:
        c = s.contract
        d = s.declarer_tricks

        if c in (Contract.RIK, Contract.RIK_BETER, Contract.TROELA, Contract.ACHT_ALLEEN):
            won = d >= 8
        elif c == Contract.NEGEN_ALLEEN:
            won = d >= 9
        elif c == Contract.TIEN_ALLEEN:
            won = d >= 10
        elif c == Contract.ELF_ALLEEN:
            won = d >= 11
        elif c == Contract.TWAALF_ALLEEN:
            won = d >= 12
        elif c == Contract.SOLO_SLIM:
            won = d == 13
        elif c in (Contract.MISERE, Contract.OPEN_MISERE):
            won = d == 0
        elif c in (Contract.PIEK, Contract.OPEN_PIEK):
            won = (d == 1) or (d == 5)
        else:
            won = False

        return +1.0 if won else -1.0

    def _compute_rewards_vector(self, s: RikkenState) -> None:
        """
        Compute zero-sum game payoffs per player based on official Dutch Rikken scoring schedule.
        Normalized by dividing by 10.0 for neural network regression stability.
        """
        c = s.contract
        s.rewards = np.zeros(4, dtype=np.float32)

        # Base contract point schedule (Declarer payoff):
        # Rik / Rik Beter / Troela: +1 pt (2v2) -> +0.10 each
        # 8 Alleen: +3 pts (1v3) -> +0.30, defenders -0.10 each
        # 9 Alleen: +4 pts (1v3) -> +0.40, defenders -0.133 each
        # 10 Alleen: +5 pts (1v3) -> +0.50, defenders -0.167 each
        # 11 Alleen: +6 pts (1v3) -> +0.60, defenders -0.200 each
        # 12 Alleen: +7 pts (1v3) -> +0.70, defenders -0.233 each
        # Solo Slim (13): +8 pts (1v3) -> +0.80, defenders -0.267 each
        # Misere / Piek: +9 pts (1v3) -> +0.90, defenders -0.300 each
        # Open Misere / Open Piek: +10 pts (1v3) -> +1.00, defenders -0.333 each

        decl_pts_map = {
            Contract.RIK: 1.0,
            Contract.RIK_BETER: 1.0,
            Contract.TROELA: 1.0,
            Contract.ACHT_ALLEEN: 3.0,
            Contract.NEGEN_ALLEEN: 4.0,
            Contract.TIEN_ALLEEN: 5.0,
            Contract.ELF_ALLEEN: 6.0,
            Contract.TWAALF_ALLEEN: 7.0,
            Contract.SOLO_SLIM: 8.0,
            Contract.MISERE: 9.0,
            Contract.PIEK: 9.0,
            Contract.OPEN_MISERE: 10.0,
            Contract.OPEN_PIEK: 10.0,
        }

        decl_base_pt = decl_pts_map.get(c, 1.0) / 10.0

        if Contract.is_multi_player_allowed(c):
            # Multi-player contract (e.g. Misere / Piek):
            decls = np.where(s.declarer_mask)[0]
            defs = [p for p in range(4) if not s.declarer_mask[p]]
            n_defs = len(defs)

            for d in decls:
                tw = s.tricks_won[d]
                if c in (Contract.MISERE, Contract.OPEN_MISERE):
                    won = (tw == 0)
                else:  # PIEK / OPEN_PIEK
                    won = (tw == 1 or tw == 5)

                if won:
                    s.rewards[d] += decl_base_pt
                    if n_defs > 0:
                        for df in defs:
                            s.rewards[df] -= (decl_base_pt / n_defs)
                else:
                    s.rewards[d] -= decl_base_pt
                    if n_defs > 0:
                        for df in defs:
                            s.rewards[df] += (decl_base_pt / n_defs)
        else:
            decl_won = (s.reward is not None and s.reward > 0)
            if Contract.is_partner_contract(c):
                # 2 vs 2 partnership contract (Rik, Rik Beter, Troela)
                sign = +1.0 if decl_won else -1.0
                for p in range(4):
                    is_decl_team = (p == s.declarer or p == s.partner)
                    s.rewards[p] = (sign * decl_base_pt) if is_decl_team else (-sign * decl_base_pt)
            else:
                # 1 vs 3 solo contract (8 Alleen through Solo Slim)
                sign = +1.0 if decl_won else -1.0
                for p in range(4):
                    if p == s.declarer:
                        s.rewards[p] = sign * decl_base_pt
                    else:
                        s.rewards[p] = -sign * (decl_base_pt / 3.0)

    def is_terminal(self, state: RikkenState) -> bool:
        return state.is_terminal

    def get_reward(self, state: RikkenState, player: int) -> float:
        if state.reward is None:
            return 0.0
        return float(state.rewards[player])
