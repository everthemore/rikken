"""
agents/neural_agent.py — Neural Agent combining BVN, Belief Network, and ISMCTS.

Architecture:
  1. Bidding: Evaluates hand + history with BVN -> picks legal bid with highest EV.
  2. Trick-taking: Runs ISMCTS determinizations informed by the Belief Network.
"""

from __future__ import annotations
import os
import torch
import numpy as np
import logging
from typing import Optional, List

from engine.card import card_id, suit_of, rank_of, ACE_RANK, HEARTS_SUIT, SUIT_MASKS
from engine.state import RikkenState, Contract, Phase
from engine.game import RikkenGame
from engine.rules import legal_bids
from networks.bvn import BVN
from networks.bn import BeliefNetwork
from agents.ismcts import ISMCTSAgent
from agents.heuristic import HeuristicAgent
import config

log = logging.getLogger(__name__)


class NeuralAgent:
    """
    Combined Neural Agent for both Bidding and Trick-taking.
    """

    def __init__(
        self,
        seat: int,
        game: RikkenGame,
        bvn: Optional[str | BVN] = None,
        bn: Optional[str | BeliefNetwork] = None,
        n_determinizations: int = config.ISMCTS_DETERMINIZATIONS,
        n_rollouts: int = config.ISMCTS_ROLLOUTS,
        device: str = config.DEVICE,
        rng: Optional[np.random.Generator] = None,
        epsilon: float = 0.0,
    ):
        self.seat = seat
        self.epsilon = epsilon
        self.game = game
        self.device = device
        self.rng = rng or np.random.default_rng()

        # Load BVN
        if isinstance(bvn, str):
            if os.path.exists(bvn):
                self.bvn = BVN.from_checkpoint(bvn, device=device)
            else:
                self.bvn = None
        else:
            self.bvn = bvn

        # Load BN
        if isinstance(bn, str):
            if os.path.exists(bn):
                self.bn = BeliefNetwork.from_checkpoint(bn, device=device)
            else:
                self.bn = None
        else:
            self.bn = bn

        # Fallback Heuristic
        self.heuristic = HeuristicAgent(seat=seat, rng=self.rng)

        # Trick-taking ISMCTS
        self.ismcts = ISMCTSAgent(
            seat=seat,
            game=game,
            n_determinizations=n_determinizations,
            n_rollouts=n_rollouts,
            belief_network=self.bn,
            rng=self.rng,
        )

    def set_seat(self, seat: int) -> None:
        self.seat = seat
        if hasattr(self, 'heuristic') and self.heuristic is not None:
            self.heuristic.set_seat(seat)
        if hasattr(self, 'ismcts') and self.ismcts is not None:
            self.ismcts.set_seat(seat)

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
        """Evaluate legal bids with Q-Value BVN and select argmax expected value (constant-free)."""
        legal = legal_bids(state)
        if not legal or legal == [int(Contract.PAS)]:
            return int(Contract.PAS)

        if self.bvn is None:
            return self.heuristic.act(state)

        p = self.seat
        hand = state.hands[p]
        bids = state.bids

        # Epsilon-greedy exploration during self-play data collection
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            non_pass_legal = [b for b in legal if b != int(Contract.PAS)]
            if non_pass_legal:
                return int(self.rng.choice(non_pass_legal))

        win_probs, ev_scores = self.bvn.predict(hand=hand, bids=bids, device=self.device)

        # Mask illegal bids with -inf
        masked_win = np.full(len(win_probs), -np.inf)
        for b in legal:
            masked_win[b] = win_probs[b]

        # Select legal contract with highest Win Probability:
        best_bid = int(np.argmax(masked_win))
        return best_bid

    def evaluate_bids(self, state: RikkenState) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (win_probabilities, expected_points) for all contracts."""
        if self.bvn is None:
            win_p = np.full(NUM_CONTRACTS, 0.5, dtype=np.float32)
            ev = np.zeros(NUM_CONTRACTS, dtype=np.float32)
            return win_p, ev
        p = self.seat
        return self.bvn.predict(state.hands[p], state.bids, device=self.device)

    def declare_trump(self, state: RikkenState) -> int:
        if state.contract == Contract.RIK_BETER:
            return HEARTS_SUIT
        return self.heuristic.declare_trump(state)

    def declare_vraagaas(self, state: RikkenState, trump_suit: int) -> int:
        return self.heuristic.declare_vraagaas(state, trump_suit)
