"""
agents/random_agent.py — Uniformly random agent for baseline comparisons and unit tests.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from engine.state import RikkenState, Phase
from engine.rules import legal_bids, legal_plays


class RandomAgent:
    """Selects uniformly at random from all legal actions."""

    def __init__(self, seat: int, rng: Optional[np.random.Generator] = None):
        self.seat = seat
        self.rng = rng or np.random.default_rng()

    def act(self, state: RikkenState) -> int:
        assert state.current_player == self.seat
        if state.phase == Phase.BIDDING:
            legal = legal_bids(state)
            return int(self.rng.choice(legal))
        else:
            mask = legal_plays(state)
            legal_cards = np.where(mask)[0]
            return int(self.rng.choice(legal_cards))

    def declare_trump(self, state: RikkenState) -> int:
        return int(self.rng.integers(0, 4))

    def declare_vraagaas(self, state: RikkenState, trump_suit: int) -> int:
        suits = [s for s in range(4) if s != trump_suit]
        return int(self.rng.choice(suits))
