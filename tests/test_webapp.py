"""
tests/test_webapp.py — Unit tests for the Rikken interactive web session and endpoints.
"""

import pytest
import numpy as np
from webapp.session import GameSession
from engine.state import Contract, Phase


def test_session_init():
    session = GameSession(human_seat=0, ai_difficulty="heuristic", seed=42)
    assert session.human_seat == 0
    assert session.state.phase in (Phase.BIDDING, Phase.TRICK_TAKING)
    payload = session.get_state_payload()
    assert payload["human_seat"] == 0
    assert len(payload["human_hand"]) == 13
    assert len(payload["hands_count"]) == 4


def test_human_bidding():
    session = GameSession(human_seat=0, ai_difficulty="heuristic", seed=42)
    payload = session.get_state_payload()
    if payload["is_human_turn"] and payload["phase"] == "BIDDING":
        legal_ids = [b["id"] for b in payload["legal_bids"]]
        assert len(legal_ids) > 0
        # Bid PAS
        ok, msg = session.human_bid(int(Contract.PAS))
        assert ok is True
        updated_payload = session.get_state_payload()
        assert len(updated_payload["log"]) > 0


def test_ai_advice_and_beliefs():
    session = GameSession(human_seat=0, ai_difficulty="heuristic", seed=42)
    advice = session.get_ai_advice()
    assert isinstance(advice, dict)

    beliefs = session.get_ai_beliefs()
    assert isinstance(beliefs, dict)
    assert "available" in beliefs


def test_full_trick_taking_interaction():
    session = GameSession(human_seat=0, ai_difficulty="heuristic", seed=42)
    # Fast forward through bidding
    while session.state.phase == Phase.BIDDING:
        if session.state.current_player == session.human_seat:
            legal = session.state.bids
            session.human_bid(int(Contract.PAS))
        else:
            session.advance_ai_turns()

    if session.state.phase == Phase.TRICK_TAKING:
        payload = session.get_state_payload()
        if payload["needs_declaration"]:
            session.human_declare(trump_suit=2, vraagaas_suit=-1)

        # Play legal card
        payload = session.get_state_payload()
        if payload["is_human_turn"] and len(payload["legal_plays"]) > 0:
            card_to_play = payload["legal_plays"][0]
            ok, msg = session.human_play(card_to_play)
            assert ok is True
