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
    win_probs[int(Contract.ACHT_ALLEEN)] = 0.75  # Higher win prob (> 0.68 threshold)

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


def test_ismcts_bn_guided_determinization():
    """Verify that ISMCTS determinization uses Belief Network predictions when available."""
    game = RikkenGame()

    class MockBeliefNetwork:
        device = "cpu"
        def predict(self, own_hand, played_cards, bid_history, current_trick, void_matrix, my_seat, device="cpu"):
            # Return 3 distributions of 52 cards
            # Give opponent 0 high probability for card 0, opponent 1 for card 1, etc.
            p0 = np.full(52, 0.01, dtype=np.float32)
            p1 = np.full(52, 0.01, dtype=np.float32)
            p2 = np.full(52, 0.01, dtype=np.float32)
            p0[0] = 0.99
            return [p0, p1, p2]

    ismcts = ISMCTSAgent(seat=0, game=game, n_determinizations=3, n_rollouts=6, belief_network=MockBeliefNetwork())
    state = game.reset()
    state.phase = Phase.TRICK_TAKING
    state.contract = Contract.RIK

    # Sample a determinization
    det = ismcts._sample_determinization(state)
    # The determinized state must be valid and filled
    assert det.hands[1].sum() > 0
    assert det.hands[2].sum() > 0
    assert det.hands[3].sum() > 0

    # Also test act() with the BN wired in
    act = ismcts.act(state)
    assert 0 <= act < 52


def test_bvn_dual_loss_counterfactual_and_cql():
    """Verify that BVNDualLoss applies counterfactual Ace loss on Misere and CQL penalty on pass hands."""
    import torch
    from networks.bvn import BVNDualLoss

    loss_fn = BVNDualLoss(lambda_aux=0.5, lambda_cql=0.2)

    # 2 hands: hand 0 has Ace (card 12), hand 1 has no Ace
    hands = torch.zeros((2, 52), dtype=torch.float32)
    hands[0, 12] = 1.0

    bid_taken = torch.tensor([0, 1], dtype=torch.long)  # Hand 0 passed, hand 1 bid Rik
    won = torch.tensor([0.0, 1.0], dtype=torch.float32)
    outcome = torch.tensor([0.0, 0.5], dtype=torch.float32)

    # Simulated predictions where Misere is hallucinated high on hand 0
    win_probs = torch.full((2, 15), 0.3, dtype=torch.float32)
    win_probs[0, int(Contract.MISERE)] = 0.85      # Hallucinated Misere on hand with Ace!
    win_probs[0, int(Contract.ACHT_ALLEEN)] = 0.70  # Hallucinated Acht Alleen on pass hand!

    ev_scores = torch.zeros((2, 15), dtype=torch.float32)
    ev_scores[0, int(Contract.MISERE)] = 0.50

    # Loss without hands (baseline supervised only)
    loss_base, _, _ = loss_fn(win_probs, ev_scores, bid_taken, won, outcome, hands=None)

    # Loss with hands (includes counterfactual Ace grounding + CQL)
    loss_with_cf, _, _ = loss_fn(win_probs, ev_scores, bid_taken, won, outcome, hands=hands)

    assert loss_with_cf > loss_base, "Counterfactual & CQL grounding should increase loss when Ace Misere/Solo is hallucinated"



