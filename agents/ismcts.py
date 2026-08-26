"""
agents/ismcts.py — Information Set Monte Carlo Tree Search (ISMCTS) agent.

Architecture:
  - Outer loop: determinize the hidden cards using the Belief Network (or uniformly
    at random in Phase 1).
  - Inner loop: standard MCTS on each determinized perfect-information game.
  - Action selection: visit-count aggregation across all determinizations.

Phase 1 (current): uniform random determinization + random rollout policy.
Phase 3 (future): BN-guided determinization + BVN-guided rollout policy.
  To upgrade: replace `_sample_determinization()` and `_rollout_policy()`.

Reference: Cowling, Powley & Whitehouse (2012), "Information Set Monte Carlo
Tree Search", IEEE Transactions on Computational Intelligence and AI in Games.
"""

from __future__ import annotations
import numpy as np
import math
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from collections import defaultdict

from engine.state import RikkenState, Phase
from engine.game import RikkenGame
from engine.rules import legal_plays, legal_bids
import config


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    """
    A node in the MCTS tree.

    Fields:
        action:         The action that led to this node (None for root).
        parent:         Parent node reference.
        children:       Dict from action → child node.
        total_reward:   Accumulated rewards from rollouts through this node.
        visit_count:    Number of times this node was visited.
        availability:   Number of determinizations where this action was legal
                        (needed for ISMCTS's modified UCB formula).
    """
    action: Optional[int]
    parent: Optional['MCTSNode']
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)
    total_reward: float = 0.0
    visit_count: int = 0
    availability: int = 0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    def ucb1(self, exploration_c: float) -> float:
        """
        ISMCTS-UCB1 formula.

        Uses `availability` in the exploration term (instead of parent visit count)
        to correct for the fact that not all actions are legal in every determinization.
        """
        if self.visit_count == 0:
            return float('inf')
        return (self.q_value
                + exploration_c * math.sqrt(math.log(self.availability) / self.visit_count))

    def best_child(self, c: float) -> 'MCTSNode':
        return max(self.children.values(), key=lambda n: n.ucb1(c))

    def most_visited_action(self) -> int:
        return max(self.children, key=lambda a: self.children[a].visit_count)


# ---------------------------------------------------------------------------
# ISMCTS Agent
# ---------------------------------------------------------------------------

class ISMCTSAgent:
    """
    ISMCTS agent for trick-taking phase.

    Args:
        seat:             Player seat index (0-3).
        game:             RikkenGame instance (shared, stateless).
        n_determinizations: Number of worlds to sample per move.
        n_rollouts:       Total MCTS rollouts (split across determinizations).
        exploration_c:    UCB1 exploration constant.
        belief_network:   Optional BN for informed determinization (Phase 3).
                          If None, uniform random sampling is used.
        rng:              NumPy random generator.
    """

    def __init__(
        self,
        seat: int,
        game: RikkenGame,
        n_determinizations: int = config.ISMCTS_DETERMINIZATIONS,
        n_rollouts: int = config.ISMCTS_ROLLOUTS,
        exploration_c: float = config.ISMCTS_C,
        belief_network=None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.seat = seat
        self.game = game
        self.n_det = n_determinizations
        self.n_rollouts = n_rollouts
        self.c = exploration_c
        self.bn = belief_network
        self.rng = rng or np.random.default_rng()

    def act(self, state: RikkenState) -> int:
        """Run ISMCTS and return the best action for state.current_player."""
        assert state.current_player == self.seat
        assert state.phase == Phase.TRICK_TAKING

        # Shared root node across all determinizations
        root = MCTSNode(action=None, parent=None)

        rollouts_per_det = max(1, self.n_rollouts // self.n_det)

        for _ in range(self.n_det):
            # 1. Determinize: sample a plausible complete game state
            det_state = self._sample_determinization(state)

            # 2. Run MCTS on this determinized state
            for _ in range(rollouts_per_det):
                self._run_mcts(root, det_state.copy())

        # 3. Select the most-visited action that is legal in the root state
        mask = legal_plays(state)
        legal_cards = np.where(mask)[0]
        valid_children = {a: child for a, child in root.children.items() if a in legal_cards}
        if not valid_children:
            return int(self.rng.choice(legal_cards))

        return max(valid_children, key=lambda a: valid_children[a].visit_count)

    # -------------------------------------------------------------------------
    # Determinization
    # -------------------------------------------------------------------------
    def _sample_determinization(self, state: RikkenState) -> RikkenState:
        """
        Sample a complete hidden-information state consistent with public knowledge.

        Phase 1: Uniform random — deal remaining cards randomly to opponents,
                 respecting the Void Matrix (no card of a void suit to that player).
        Phase 3: Replace with BN-guided sampling.

        Returns:
            A copy of `state` with all hidden hands filled in.
        """
        s = state.copy()

        # Cards that are "unknown" (not in our hand, not yet played, not in current trick)
        my_hand = s.hands[self.seat].astype(bool)
        played = s.played_cards.astype(bool)
        in_trick = np.zeros(52, dtype=bool)
        for c in s.current_trick:
            if c >= 0:
                in_trick[c] = True

        unknown_cards = np.where(~my_hand & ~played & ~in_trick)[0]

        if len(unknown_cards) == 0:
            return s

        # Count how many cards each opponent needs
        opponents = [p for p in range(4) if p != self.seat]
        opp_counts = [int(s.hands[p].sum()) for p in opponents]

        # Shuffle unknown cards and distribute, respecting void matrix
        sampled_hands = self._distribute_cards(
            unknown_cards, opponents, opp_counts, s.void_matrix, s.trump_suit
        )

        for p, hand_arr in sampled_hands.items():
            s.hands[p] = hand_arr

        return s

    def _distribute_cards(
        self,
        unknown_cards: np.ndarray,
        opponents: List[int],
        opp_counts: List[int],
        void_matrix: np.ndarray,
        trump_suit: int,
    ) -> Dict[int, np.ndarray]:
        """
        Distribute `unknown_cards` to opponents respecting their void constraints.

        Uses rejection sampling with a maximum retry budget.
        Falls back to unconstrained distribution if constraints can't be satisfied.
        """
        from engine.card import suit_of, SUIT_MASKS

        max_retries = 20
        for _ in range(max_retries):
            cards = unknown_cards.copy()
            self.rng.shuffle(cards)

            result = {p: np.zeros(52, dtype=np.int8) for p in opponents}
            valid = True
            idx = 0

            for p, count in zip(opponents, opp_counts):
                assigned = 0
                skipped = []
                while assigned < count and idx < len(cards):
                    c = cards[idx]
                    idx += 1
                    suit = suit_of(c)
                    if void_matrix[p, suit]:
                        # This player is void in this suit → skip
                        skipped.append(c)
                    else:
                        result[p][c] = 1
                        assigned += 1

                # Put skipped cards back (for next player)
                # Replace cards in-place at the end
                cards = np.concatenate([cards[idx:], np.array(skipped, dtype=np.int8)])
                idx = 0

                if assigned < count:
                    # Couldn't fill this player's hand
                    valid = False
                    break

            if valid:
                return result

        # Fallback: ignore void constraints
        cards = unknown_cards.copy()
        self.rng.shuffle(cards)
        result = {p: np.zeros(52, dtype=np.int8) for p in opponents}
        idx = 0
        for p, count in zip(opponents, opp_counts):
            for c in cards[idx:idx + count]:
                result[p][c] = 1
            idx += count
        return result

    # -------------------------------------------------------------------------
    # MCTS
    # -------------------------------------------------------------------------
    def _run_mcts(self, root: MCTSNode, state: RikkenState) -> None:
        """Run one full MCTS iteration: select → expand → rollout → backprop."""
        node = root
        path = [node]
        legal_at_root = self._get_legal_list(state)

        # ---- Selection ----
        while node.children and not self.game.is_terminal(state):
            legal = self._get_legal_list(state)
            # Update availability for all legal actions
            for a in legal:
                if a in node.children:
                    node.children[a].availability += 1

            # Expand if there are untried actions legal in this determinization
            untried = [a for a in legal if a not in node.children]
            if untried:
                break

            # In ISMCTS, only select among children whose actions are legal in this determinization
            legal_children = [child for a, child in node.children.items() if a in legal]
            if not legal_children:
                break

            node = max(legal_children, key=lambda n: n.ucb1(self.c))
            state, reward = self.game.step(state, node.action)
            if reward is not None:
                self._backprop(path + [node], reward, state)
                return
            path.append(node)

        if self.game.is_terminal(state):
            reward = self.game.get_reward(state, self.seat)
            self._backprop(path, reward, state)
            return

        # ---- Expansion ----
        legal = self._get_legal_list(state)
        untried = [a for a in legal if a not in node.children]
        if untried:
            action = self.rng.choice(untried)
            child = MCTSNode(action=int(action), parent=node, availability=1)
            node.children[int(action)] = child
            node = child
            path.append(node)
            state, reward = self.game.step(state, int(action))
            if reward is not None:
                self._backprop(path, reward, state)
                return

        # ---- Rollout ----
        reward = self._rollout(state)

        # ---- Backpropagation ----
        self._backprop(path, reward, state)

    def _rollout(self, state: RikkenState) -> float:
        """
        Random rollout from `state` to terminal.

        Phase 1: uniform random play.
        Phase 3: replace with BVN/BN-guided policy.
        """
        s = state.copy()
        while not self.game.is_terminal(s):
            legal = self._get_legal_list(s)
            action = int(self.rng.choice(legal))
            s, reward = self.game.step(s, action)
            if reward is not None:
                return self.game.get_reward(s, self.seat)
        return self.game.get_reward(s, self.seat)

    def _backprop(
        self, path: List[MCTSNode], reward: float, final_state: RikkenState
    ) -> None:
        """Backpropagate the reward up the tree path."""
        for node in reversed(path):
            node.visit_count += 1
            node.total_reward += reward

    def _get_legal_list(self, state: RikkenState) -> np.ndarray:
        """Return legal actions as a flat array regardless of phase."""
        if state.phase == Phase.BIDDING:
            return np.array(legal_bids(state))
        else:
            mask = legal_plays(state)
            return np.where(mask)[0]
