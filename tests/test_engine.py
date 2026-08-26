"""
tests/test_engine.py — Unit tests for the Rikken game engine.

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from engine.card import (
    card_id, suit_of, rank_of, card_to_str, str_to_card,
    beats, trick_winner, ACE_RANK,
)
from engine.deck import clumping_shuffle, deal
from engine.state import RikkenState, Contract, Phase
from engine.rules import legal_bids, legal_plays, update_void_matrix
from engine.game import RikkenGame
from engine.early_stop import check_basic_early_stop


# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

class TestCardHelpers:
    def test_card_id_roundtrip(self):
        for suit in range(4):
            for rank in range(13):
                c = card_id(suit, rank)
                assert suit_of(c) == suit
                assert rank_of(c) == rank

    def test_card_to_str(self):
        ace_spades = card_id(3, 12)
        assert '♠' in card_to_str(ace_spades)
        assert 'A' in card_to_str(ace_spades)

    def test_str_to_card_roundtrip(self):
        for suit in range(4):
            for rank in range(13):
                c = card_id(suit, rank)
                s = card_to_str(c)
                assert str_to_card(s) == c

    def test_beats_trump_over_off_suit(self):
        trump = 0  # Clubs
        two_of_clubs = card_id(0, 0)
        ace_of_spades = card_id(3, 12)
        led_suit = 3  # Spades
        assert beats(two_of_clubs, ace_of_spades, trump, led_suit)

    def test_beats_higher_rank_same_suit(self):
        led = 2  # Hearts
        king = card_id(2, 11)
        ten  = card_id(2, 8)
        assert beats(king, ten, -1, led)
        assert not beats(ten, king, -1, led)

    def test_trick_winner_led_suit_wins(self):
        # Player 0 leads Ace of Hearts; no trump; others play lower
        trump = -1
        leader = 0
        trick = np.array([
            card_id(2, 12),  # P0: Ace of Hearts (leader)
            card_id(2, 5),   # P1: 7 of Hearts
            card_id(3, 12),  # P2: Ace of Spades (off suit, doesn't win)
            card_id(2, 3),   # P3: 5 of Hearts
        ], dtype=np.int8)
        assert trick_winner(trick, leader, trump) == 0

    def test_trick_winner_trump_wins(self):
        trump = 0  # Clubs
        leader = 1
        trick = np.array([
            card_id(2, 12),  # P0: Ace of Hearts (not leader, off-trump)
            card_id(2, 11),  # P1: King of Hearts (leader)
            card_id(0, 0),   # P2: 2 of Clubs (trump!)
            card_id(2, 10),  # P3: Queen of Hearts
        ], dtype=np.int8)
        assert trick_winner(trick, leader, trump) == 2


# ---------------------------------------------------------------------------
# Deck and dealing
# ---------------------------------------------------------------------------

class TestDeckAndDeal:
    def test_deal_correct_counts(self):
        rng = np.random.default_rng(42)
        deck = clumping_shuffle(None, rng)
        hands = deal(deck)
        # Each player has 13 cards
        assert hands.shape == (4, 52)
        for p in range(4):
            assert hands[p].sum() == 13, f"Player {p} has {hands[p].sum()} cards"

    def test_deal_no_duplicates(self):
        rng = np.random.default_rng(42)
        deck = clumping_shuffle(None, rng)
        hands = deal(deck)
        # Each card assigned exactly once
        total = hands.sum(axis=0)
        assert np.all(total == 1), "Some card appears more than once or not at all"

    def test_deal_round_robin_order(self):
        """Verify the 6-then-7 round-robin deal order."""
        deck = np.arange(52, dtype=np.int8)  # ordered deck for easy verification
        hands = deal(deck)
        # Round 1: P0←[0-5], P1←[6-11], P2←[12-17], P3←[18-23]
        for p in range(4):
            for i in range(6):
                card = p * 6 + i
                assert hands[p, card] == 1, f"P{p} should hold card {card} (round 1)"
        # Round 2: P0←[24-30], P1←[31-37], P2←[38-44], P3←[45-51]
        for p in range(4):
            for i in range(7):
                card = 24 + p * 7 + i
                assert hands[p, card] == 1, f"P{p} should hold card {card} (round 2)"

    def test_clumping_shuffle_all_52_cards(self):
        rng = np.random.default_rng(7)
        deck = clumping_shuffle(None, rng)
        assert len(deck) == 52
        assert sorted(deck) == list(range(52))

    def test_clumping_shuffle_from_prev_tricks(self):
        """Shuffle from a previous game's trick sequence."""
        # Build a fake trick sequence (all 13 tricks with 4 cards each)
        cards = list(range(52))
        trick_seq = [
            (i % 4, [cards[i*4], cards[i*4+1], cards[i*4+2], cards[i*4+3]])
            for i in range(13)
        ]
        rng = np.random.default_rng(99)
        deck = clumping_shuffle(trick_seq, rng)
        assert sorted(deck) == list(range(52))


# ---------------------------------------------------------------------------
# Bidding rules
# ---------------------------------------------------------------------------

class TestBiddingRules:
    def setup_method(self):
        self.rng = np.random.default_rng(0)
        self.game = RikkenGame(rng=self.rng)
        self.state = self.game.reset()

    def test_can_pass_before_bidding(self):
        legal = legal_bids(self.state)
        assert int(Contract.PAS) in legal

    def test_cannot_rebid_after_pass(self):
        s = self.state.copy()
        s.passed[0] = True
        s.current_player = 0
        legal = legal_bids(s)
        assert len(legal) == 0

    def test_cannot_bid_lower_than_current(self):
        s = self.state.copy()
        s.highest_bid = Contract.NEGEN_ALLEEN
        s.current_player = 1
        legal = legal_bids(s)
        for b in legal:
            assert b > int(Contract.NEGEN_ALLEEN) or b == int(Contract.PAS)

    def test_troela_requires_3_aces(self):
        s = self.state.copy()
        # Give player 0 only 2 aces
        s.hands[0] = np.zeros(52, dtype=np.int8)
        s.hands[0][card_id(0, ACE_RANK)] = 1
        s.hands[0][card_id(1, ACE_RANK)] = 1
        # Fill remaining 11 cards with non-aces
        for i in range(11):
            s.hands[0][i] = 1
        s.current_player = 0
        legal = legal_bids(s)
        assert int(Contract.TROELA) not in legal

    def test_troela_available_with_3_aces(self):
        s = self.state.copy()
        # Give player 0 exactly 3 aces
        s.hands[0] = np.zeros(52, dtype=np.int8)
        for suit in range(3):
            s.hands[0][card_id(suit, ACE_RANK)] = 1
        # Fill remaining 10 cards
        for i in range(10):
            s.hands[0][i] = 1
        s.current_player = 0
        legal = legal_bids(s)
        assert int(Contract.TROELA) in legal

    
    def test_rik_beter_ladder(self):
        assert int(Contract.RIK) < int(Contract.RIK_BETER) < int(Contract.ACHT_ALLEEN)

    def test_multi_player_misere_bidding(self):
        s = self.state.copy()
        s.highest_bid = Contract.MISERE
        s.bids[0] = int(Contract.MISERE)
        s.bid_winner = 0
        s.current_player = 1
        legal = legal_bids(s)
        assert int(Contract.MISERE) in legal, "Subsequent player should be allowed to co-bid Misere"

    def test_all_pass_redeal(self):
        """Four consecutive passes should result in a redeal (reward=0)."""
        s = self.state.copy()
        reward = None
        for _ in range(4):
            s, reward = self.game.step(s, int(Contract.PAS))
            if reward is not None:
                break
        assert reward == 0.0
        assert s.phase == Phase.TERMINAL


# ---------------------------------------------------------------------------
# Trick-taking rules
# ---------------------------------------------------------------------------

class TestTrickTakingRules:
    def _make_trick_state(
        self, led_suit: int, trump_suit: int, hand: np.ndarray,
        leader: int = 0, player: int = 1
    ) -> RikkenState:
        """Helper: create a state mid-trick."""
        s = RikkenState.initial()
        s.phase = Phase.TRICK_TAKING
        s.trump_suit = trump_suit
        s.contract = Contract.ACHT_ALLEEN
        s.declarer = 0
        s.trick_leader = leader
        s.current_player = player
        s.hands[player] = hand
        # Leader played a card of led_suit
        s.current_trick[leader] = card_id(led_suit, 5)  # 7 of led_suit
        return s

    def test_must_follow_suit(self):
        """Player must play a card of the led suit if they have one."""
        hand = np.zeros(52, dtype=np.int8)
        hand[card_id(2, 3)] = 1   # 5 of Hearts (led suit)
        hand[card_id(2, 7)] = 1   # 9 of Hearts
        hand[card_id(3, 12)] = 1  # Ace of Spades (off suit)

        s = self._make_trick_state(led_suit=2, trump_suit=0, hand=hand)
        mask = legal_plays(s)

        # Only Hearts are legal
        legal_cards = np.where(mask)[0]
        for c in legal_cards:
            assert suit_of(c) == 2, f"Card {card_to_str(c)} is not Hearts (led suit)"

    def test_must_trump_when_void_in_led_suit(self):
        """If void in led suit and holding trump, must play trump."""
        hand = np.zeros(52, dtype=np.int8)
        hand[card_id(0, 3)] = 1   # 5 of Clubs (trump)
        hand[card_id(3, 5)] = 1   # 7 of Spades (off suit, off trump)
        hand[card_id(3, 8)] = 1   # 10 of Spades

        s = self._make_trick_state(led_suit=2, trump_suit=0, hand=hand)  # led=Hearts, trump=Clubs
        mask = legal_plays(s)

        legal_cards = np.where(mask)[0]
        assert len(legal_cards) == 1
        assert suit_of(legal_cards[0]) == 0  # Must play Clubs (trump)

    def test_free_discard_when_void_in_both(self):
        """Void in led suit AND trump → may play any card."""
        hand = np.zeros(52, dtype=np.int8)
        hand[card_id(1, 3)] = 1   # 5 of Diamonds (not Hearts, not Clubs)
        hand[card_id(3, 8)] = 1   # 10 of Spades (not Hearts, not Clubs)

        s = self._make_trick_state(led_suit=2, trump_suit=0, hand=hand)  # led=Hearts, trump=Clubs
        mask = legal_plays(s)

        # Both cards should be legal
        legal_cards = np.where(mask)[0]
        assert len(legal_cards) == 2

    def test_leader_may_play_any_card(self):
        """Trick leader may play any card in hand."""
        hand = np.zeros(52, dtype=np.int8)
        hand[card_id(0, 0)] = 1
        hand[card_id(1, 5)] = 1
        hand[card_id(2, 10)] = 1

        s = RikkenState.initial()
        s.phase = Phase.TRICK_TAKING
        s.trump_suit = 0
        s.contract = Contract.ACHT_ALLEEN
        s.declarer = 0
        s.trick_leader = 0
        s.current_player = 0
        s.hands[0] = hand
        s.current_trick = np.full(4, -1, dtype=np.int8)

        mask = legal_plays(s)
        assert mask.sum() == 3  # All 3 cards legal


# ---------------------------------------------------------------------------
# Void Matrix
# ---------------------------------------------------------------------------

class TestVoidMatrix:
    def _make_state_for_void(self, trump_suit=0):
        s = RikkenState.initial()
        s.phase = Phase.TRICK_TAKING
        s.trump_suit = trump_suit
        return s

    def test_void_recorded_on_fail_to_follow(self):
        """If a player fails to follow suit, they're marked void in led suit."""
        s = self._make_state_for_void(trump_suit=0)
        # Player 1 discards a Spade when Hearts was led (and player held no Hearts)
        update_void_matrix(s, player=1, led_suit=2, card_played=card_id(3, 5))
        assert s.void_matrix[1, 2], "Player 1 should be void in Hearts (led suit)"

    def test_void_in_trump_when_discarding(self):
        """If a player discards (not led suit, not trump), they're void in both."""
        s = self._make_state_for_void(trump_suit=0)
        # Player 2 plays a Diamond when Hearts was led and Clubs is trump
        update_void_matrix(s, player=2, led_suit=2, card_played=card_id(1, 8))
        assert s.void_matrix[2, 2], "Player 2 should be void in Hearts"
        assert s.void_matrix[2, 0], "Player 2 should be void in Clubs (trump)"

    def test_no_void_when_following_suit(self):
        """Playing the led suit should NOT mark any void."""
        s = self._make_state_for_void(trump_suit=0)
        update_void_matrix(s, player=3, led_suit=2, card_played=card_id(2, 5))
        assert not np.any(s.void_matrix[3]), "No void should be recorded for following suit"

    def test_no_void_when_trumping(self):
        """Playing a trump when void in led suit marks only led-suit void, not trump void."""
        s = self._make_state_for_void(trump_suit=0)
        # Player plays a Club (trump) because they have no Hearts
        update_void_matrix(s, player=1, led_suit=2, card_played=card_id(0, 3))
        assert s.void_matrix[1, 2], "Player 1 should be void in Hearts (led suit)"
        assert not s.void_matrix[1, 0], "Player 1 should NOT be void in Clubs (trump)"


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def _make_state(self, contract, d_tricks, def_tricks, remaining):
        s = RikkenState.initial()
        s.phase = Phase.TRICK_TAKING
        s.contract = contract
        s.declarer = 0
        s.partner = -1
        s.tricks_won[0] = d_tricks
        # distribute def_tricks among opponents
        s.tricks_won[1] = def_tricks
        s.trick_count = d_tricks + def_tricks
        return s

    def test_misere_stops_at_1_trick(self):
        s = self._make_state(Contract.MISERE, d_tricks=1, def_tricks=0, remaining=12)
        result = check_basic_early_stop(s)
        assert result == -1.0, "Misère should terminate immediately when declarer takes 1 trick"

    def test_misere_no_stop_at_0_tricks(self):
        s = self._make_state(Contract.MISERE, d_tricks=0, def_tricks=3, remaining=10)
        result = check_basic_early_stop(s)
        assert result is None, "Misère should continue if declarer has 0 tricks"

    def test_acht_alleen_win_at_8(self):
        s = self._make_state(Contract.ACHT_ALLEEN, d_tricks=8, def_tricks=0, remaining=5)
        result = check_basic_early_stop(s)
        assert result == +1.0

    def test_acht_alleen_loss_at_6_defender(self):
        s = self._make_state(Contract.ACHT_ALLEEN, d_tricks=1, def_tricks=6, remaining=6)
        result = check_basic_early_stop(s)
        assert result == -1.0

    def test_solo_slim_loss_at_1_defender_trick(self):
        s = self._make_state(Contract.SOLO_SLIM, d_tricks=5, def_tricks=1, remaining=7)
        result = check_basic_early_stop(s)
        assert result == -1.0

    def test_piek_loss_in_dead_zone(self):
        """Piek: 2 tricks with remaining=2 (max 4) -> cannot reach 5 tricks -> loss."""
        s = self._make_state(Contract.PIEK, d_tricks=2, def_tricks=9, remaining=2)
        s.trick_count = 11
        result = check_basic_early_stop(s)
        assert result == -1.0, f"Expected -1.0, got {result}"

    def test_piek_win_at_exactly_1(self):
        s = self._make_state(Contract.PIEK, d_tricks=1, def_tricks=12, remaining=0)
        s.trick_count = 13
        result = check_basic_early_stop(s)
        assert result == +1.0

    def test_piek_win_at_exactly_5(self):
        s = self._make_state(Contract.PIEK, d_tricks=5, def_tricks=8, remaining=0)
        s.trick_count = 13
        result = check_basic_early_stop(s)
        assert result == +1.0

    def test_piek_loss_at_6_tricks(self):
        s = self._make_state(Contract.PIEK, d_tricks=6, def_tricks=2, remaining=5)
        s.trick_count = 8
        result = check_basic_early_stop(s)
        assert result == -1.0


# ---------------------------------------------------------------------------
# Full game smoke test
# ---------------------------------------------------------------------------

class TestFullGame:
    def test_heuristic_game_completes(self):
        """A full game with heuristic agents should complete without errors."""
        from agents.heuristic import HeuristicAgent
        from engine.state import Contract, Phase

        rng = np.random.default_rng(123)
        game = RikkenGame(rng=rng)
        agents = [HeuristicAgent(seat=p, rng=rng) for p in range(4)]

        state = game.reset()
        assert state.phase == Phase.BIDDING

        for _ in range(200):  # safety limit
            if state.phase == Phase.TERMINAL:
                break
            elif state.phase == Phase.BIDDING:
                action = agents[state.current_player].act(state)
                state, reward = game.step(state, action)
                if reward is not None:
                    break
                if state.phase == Phase.TRICK_TAKING and state.trick_count == 0:
                    d = state.declarer
                    if Contract.is_trump_contract(state.contract):
                        trump = agents[d].declare_trump(state)
                        vraagaas = -1
                        if state.contract in (Contract.RIK, Contract.RIK_BETER):
                            vraagaas = agents[d].declare_vraagaas(state, trump)
                        state = game.declare(state, trump_suit=trump, vraagaas_suit=vraagaas)
            elif state.phase == Phase.TRICK_TAKING:
                action = agents[state.current_player].act(state)
                state, reward = game.step(state, action)

        assert state.phase == Phase.TERMINAL, "Game did not terminate"
        assert state.reward in (0.0, +1.0, -1.0), f"Invalid reward: {state.reward}"

    def test_1000_games_no_exception(self):
        """Run 1000 games; none should raise an exception."""
        from agents.heuristic import HeuristicAgent

        rng = np.random.default_rng(0)
        game = RikkenGame(rng=rng)
        completed = 0

        for _ in range(1000):
            agents = [HeuristicAgent(seat=p, rng=rng) for p in range(4)]
            state = game.reset()

            for __ in range(300):
                if state.phase == Phase.TERMINAL:
                    completed += 1
                    break
                elif state.phase == Phase.BIDDING:
                    action = agents[state.current_player].act(state)
                    state, reward = game.step(state, action)
                    if reward is not None:
                        completed += 1
                        break
                    if (state.phase == Phase.TRICK_TAKING
                            and state.trick_count == 0
                            and Contract.is_trump_contract(state.contract)):
                        d = state.declarer
                        trump = agents[d].declare_trump(state)
                        vraagaas = -1
                        if state.contract in (Contract.RIK, Contract.RIK_BETER):
                            vraagaas = agents[d].declare_vraagaas(state, trump)
                        state = game.declare(state, trump_suit=trump, vraagaas_suit=vraagaas)
                elif state.phase == Phase.TRICK_TAKING:
                    action = agents[state.current_player].act(state)
                    state, reward = game.step(state, action)
                    if reward is not None:
                        completed += 1
                        break

        assert completed == 1000, f"Only {completed}/1000 games completed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
