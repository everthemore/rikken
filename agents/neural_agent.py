"""
agents/neural_agent.py — Integrated Neural Agent combining BVN, BN, and ISMCTS.

This agent represents the final AI architecture:
  - Bidding: Queries BVN (Transformer) for Expected Values (EV) across legal bids,
             selecting the argmax EV.
  - Declaration: Uses neural heuristic / trump evaluation for contract choices.
  - Trick-taking: Runs ISMCTS with Belief Network (BN) guided determinizations.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional, List

from engine.state import RikkenState, Contract, Phase
from engine.game import RikkenGame
from engine.rules import legal_bids, legal_plays
from engine.card import NUM_SUITS, cards_in_suit, rank_of, card_id, ACE_RANK, NUM_RANKS
from networks.bvn import BVN
from networks.bn import BeliefNetwork
from agents.ismcts import ISMCTSAgent
import config


class NeuralAgent:
    """
    Complete hybrid Neural + ISMCTS agent.

    Args:
        seat:             Player index (0-3).
        game:             RikkenGame instance.
        bvn:              Pretrained BVN model (or path to checkpoint).
        bn:               Pretrained BeliefNetwork model (or path to checkpoint).
        n_determinizations: ISMCTS worlds per move.
        n_rollouts:       ISMCTS total rollouts per move.
        device:           PyTorch device ('cuda', 'mps', 'cpu').
        rng:              NumPy random generator.
    """

    def __init__(
        self,
        seat: int,
        game: RikkenGame,
        bvn: Optional[BVN | str] = None,
        bn: Optional[BeliefNetwork | str] = None,
        n_determinizations: int = config.ISMCTS_DETERMINIZATIONS,
        n_rollouts: int = config.ISMCTS_ROLLOUTS,
        device: str = config.DEVICE,
        rng: Optional[np.random.Generator] = None,
    ):
        self.seat = seat
        self.game = game
        self.device = device
        self.rng = rng or np.random.default_rng()

        # Load BVN
        if isinstance(bvn, str):
            self.bvn = BVN().to(device)
            ckpt = torch.load(bvn, map_location=device)
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            self.bvn.load_state_dict(state_dict)
            self.bvn.eval()
        else:
            self.bvn = bvn

        # Load BN
        if isinstance(bn, str):
            self.bn = BeliefNetwork().to(device)
            ckpt = torch.load(bn, map_location=device)
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            self.bn.load_state_dict(state_dict)
            self.bn.eval()
        else:
            self.bn = bn

        # Trick-taking ISMCTS
        self.ismcts = ISMCTSAgent(
            seat=seat,
            game=game,
            n_determinizations=n_determinizations,
            n_rollouts=n_rollouts,
            belief_network=self.bn,
            rng=self.rng,
        )

    def act(self, state: RikkenState) -> int:
        """Choose action during either Bidding or Trick-taking."""
        assert state.current_player == self.seat
        if state.phase == Phase.BIDDING:
            return self._act_bid(state)
        elif state.phase == Phase.TRICK_TAKING:
            return self.ismcts.act(state)
        else:
            raise RuntimeError("Act called on terminal state")

    def _act_bid(self, state: RikkenState) -> int:
        """Evaluate legal bids with BVN and select argmax expected value."""
        legal = legal_bids(state)
        if not legal:
            return int(Contract.PAS)
        if len(legal) == 1:
            return legal[0]

        if self.bvn is not None:
            hand = state.hands[self.seat]
            bid_hist = np.zeros((4, 13), dtype=np.int8)
            for p in range(4):
                if state.bids[p] >= 0:
                    bid_hist[p, state.bids[p]] = 1

            evs = self.bvn.predict_ev(hand, bid_hist, legal, device=self.device)
            # Find best legal contract
            best_bid = legal[0]
            best_ev = -float('inf')
            for b in legal:
                if evs[b] > best_ev:
                    best_ev = evs[b]
                    best_bid = b
            return best_bid
        else:
            # Fallback heuristic
            return legal[0]

    def declare_trump(self, state: RikkenState) -> int:
        """Select best trump suit."""
        from engine.state import Contract as C
        hand = state.hands[self.seat]

        forbidden_suit = -1
        if state.contract == C.TROELA and state.partner >= 0:
            for suit in range(NUM_SUITS):
                ace = card_id(suit, ACE_RANK)
                if state.hands[state.partner, ace]:
                    forbidden_suit = suit
                    break

        best_suit = -1
        best_score = -1
        for suit in range(NUM_SUITS):
            if suit == forbidden_suit:
                continue
            suit_cards = cards_in_suit(hand, suit)
            score = len(suit_cards)
            for c in suit_cards:
                score += rank_of(c) / NUM_RANKS
            if score > best_score:
                best_score = score
                best_suit = suit

        if best_suit < 0:
            for suit in range(NUM_SUITS):
                if suit != forbidden_suit:
                    return suit
        return best_suit

    def declare_vraagaas(self, state: RikkenState, trump_suit: int) -> int:
        """Select vraagaas suit."""
        hand = state.hands[self.seat]
        worst_suit = -1
        worst_score = float('inf')
        for suit in range(NUM_SUITS):
            ace = card_id(suit, ACE_RANK)
            if hand[ace]:
                continue
            suit_cards = cards_in_suit(hand, suit)
            score = len(suit_cards) + sum(rank_of(c) for c in suit_cards)
            if score < worst_score:
                worst_score = score
                worst_suit = suit
        return worst_suit if worst_suit >= 0 else (trump_suit + 1) % NUM_SUITS
