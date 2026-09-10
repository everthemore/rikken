"""
tests/test_bidding_calibration.py — Verification of Calibrated Bidding & Heuristic ISMCTS Rollouts.
"""

from __future__ import annotations
import numpy as np
import pytest

from engine.state import RikkenState, Contract, Phase
from engine.game import RikkenGame
from engine.rules import legal_bids
from agents.neural_agent import NeuralAgent
from agents.ismcts import ISMCTSAgent


class DummyBVN:
    """Mock BVN that returns controllable win_probs and ev_scores for testing."""

    def __init__(self, win_probs: np.ndarray, ev_scores: np.ndarray):
        self._win_probs = win_probs
        self._ev_scores = ev_scores

    def predict(self, hand: np.ndarray, bids: np.ndarray, device: str = "cpu"):
        return self._win_probs.copy(), self._ev_scores.copy()


def test_calibrated_bidding_rejects_marginal_open_piek():
    """Verify that a marginal Open Piek (e.g. 58% vs 55% PAS) is rejected in favor of PAS."""
    game = RikkenGame()
    agent = NeuralAgent(seat=0, game=game)

    # Mock predictions: OPEN_PIEK (idx 8) is 0.58, PAS (idx 0) is 0.55, but margin < 0.20 and prob < 0.70
    win_probs = np.full(15, 0.20, dtype=np.float32)
    win_probs[int(Contract.PAS)] = 0.55
    win_probs[int(Contract.OPEN_PIEK)] = 0.58  # Higher than PAS, but marginal!

    ev_scores = np.full(15, -0.5, dtype=np.float32)
    ev_scores[int(Contract.PAS)] = 0.10
    ev_scores[int(Contract.OPEN_PIEK)] = 0.05

    agent.bvn = DummyBVN(win_probs, ev_scores)

    state = game.reset()
    bid = agent._act_bid(state)

    assert bid == int(Contract.PAS), f"Expected PAS on marginal Open Piek, got {Contract(bid).name}"


def test_calibrated_bidding_accepts_strong_open_piek():
    """Verify that a very high confidence Open Piek (85% vs 50% PAS, EV=+0.6) is accepted."""
    game = RikkenGame()
    agent = NeuralAgent(seat=0, game=game)

    win_probs = np.full(15, 0.20, dtype=np.float32)
    win_probs[int(Contract.PAS)] = 0.50
    win_probs[int(Contract.OPEN_PIEK)] = 0.85

    ev_scores = np.full(15, -0.5, dtype=np.float32)
    ev_scores[int(Contract.PAS)] = 0.05
    ev_scores[int(Contract.OPEN_PIEK)] = 0.60

    agent.bvn = DummyBVN(win_probs, ev_scores)

    state = game.reset()
    bid = agent._act_bid(state)

    assert bid == int(Contract.OPEN_PIEK), f"Expected OPEN_PIEK on strong hand, got {Contract(bid).name}"


def test_calibrated_bidding_partner_contract():
    """Verify that standard partner contracts (RIK) are bid when above PAS with positive EV."""
    game = RikkenGame()
    agent = NeuralAgent(seat=0, game=game)

    win_probs = np.full(15, 0.20, dtype=np.float32)
    win_probs[int(Contract.PAS)] = 0.52
    win_probs[int(Contract.RIK)] = 0.65

    ev_scores = np.full(15, -0.5, dtype=np.float32)
    ev_scores[int(Contract.PAS)] = 0.05
    ev_scores[int(Contract.RIK)] = 0.35

    agent.bvn = DummyBVN(win_probs, ev_scores)

    state = game.reset()
    bid = agent._act_bid(state)

    assert bid == int(Contract.RIK), f"Expected RIK on solid partner hand, got {Contract(bid).name}"


def test_calibrated_bidding_negative_ev_rejection():
    """Verify that even with win_prob > pas_win, negative EV contracts are rejected."""
    game = RikkenGame()
    agent = NeuralAgent(seat=0, game=game)

    win_probs = np.full(15, 0.20, dtype=np.float32)
    win_probs[int(Contract.PAS)] = 0.50
    win_probs[int(Contract.ACHT_ALLEEN)] = 0.65  # Higher win prob

    ev_scores = np.full(15, -0.5, dtype=np.float32)
    ev_scores[int(Contract.PAS)] = 0.10
    ev_scores[int(Contract.ACHT_ALLEEN)] = -0.20  # Negative EV!

    agent.bvn = DummyBVN(win_probs, ev_scores)

    state = game.reset()
    bid = agent._act_bid(state)

    assert bid == int(Contract.PAS), f"Expected PAS on negative EV, got {Contract(bid).name}"


def test_ismcts_heuristic_rollout():
    """Verify that ISMCTS heuristic rollout completes and returns a valid reward."""
    game = RikkenGame()
    ismcts = ISMCTSAgent(seat=0, game=game, n_determinizations=2, n_rollouts=10)

    # Set up a game state in trick taking
    state = game.reset()
    state.phase = Phase.TRICK_TAKING
    state.contract = Contract.RIK
    state.trump_suit = 0
    state.declarer = 0

    reward = ismcts._rollout(state)
    assert -1.0 <= reward <= 1.0, f"Expected reward in [-1.0, 1.0], got {reward}"


def test_calibrated_bidding_prioritizes_high_prob_rik_over_marginal_misere():
    """Verify that a high probability RIK (e.g. 75%) beats a marginal Misere (55%) despite lower linear EV."""
    game = RikkenGame()
    agent = NeuralAgent(seat=0, game=game)

    win_probs = np.full(15, 0.20, dtype=np.float32)
    win_probs[int(Contract.PAS)] = 0.50
    win_probs[int(Contract.RIK)] = 0.75
    win_probs[int(Contract.MISERE)] = 0.72  # Qualifies, but lower than 0.75

    ev_scores = np.full(15, -0.5, dtype=np.float32)
    ev_scores[int(Contract.PAS)] = 0.05
    ev_scores[int(Contract.RIK)] = 0.10      # +1 point net
    ev_scores[int(Contract.MISERE)] = 0.40   # +9 points net (much higher linear EV!)

    agent.bvn = DummyBVN(win_probs, ev_scores)

    state = game.reset()
    bid = agent._act_bid(state)

    assert bid == int(Contract.RIK), f"Expected RIK to beat marginal MISERE on win prob, got {Contract(bid).name}"


def test_calibrated_bidding_rejects_marginal_misere():
    """Verify that a marginal Misere (55% < 70% threshold) is rejected even with positive EV."""
    game = RikkenGame()
    agent = NeuralAgent(seat=0, game=game)

    win_probs = np.full(15, 0.20, dtype=np.float32)
    win_probs[int(Contract.PAS)] = 0.50
    win_probs[int(Contract.MISERE)] = 0.55  # Below 0.70 threshold

    ev_scores = np.full(15, -0.5, dtype=np.float32)
    ev_scores[int(Contract.PAS)] = 0.05
    ev_scores[int(Contract.MISERE)] = 0.25   # High positive EV

    agent.bvn = DummyBVN(win_probs, ev_scores)

    state = game.reset()
    bid = agent._act_bid(state)

    assert bid == int(Contract.PAS), f"Expected PAS on marginal Misere, got {Contract(bid).name}"

