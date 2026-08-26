"""
engine/state.py — RikkenState dataclass and Contract enum.

All arrays are NumPy int8 / bool for maximum speed and JAX compatibility.
The state is designed to be cheaply copyable (via .copy()) for ISMCTS.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Contract Enum
# ---------------------------------------------------------------------------
class Contract(IntEnum):
    """Bidding hierarchy in ascending order (higher value = stronger bid)."""
    NO_BID        = -1   # Sentinel: player hasn't bid yet
    PAS           = 0    # Pass (no bid)
    RIK           = 1    # Partner contract: trump in C/D/S, call vraagaas, win 8+ tricks
    RIK_BETER     = 2    # Partner contract: trump fixed to HEARTS, call vraagaas, win 8+ tricks
    ACHT_ALLEEN   = 3    # Solo: choose trump, win 8+ tricks
    PIEK          = 4    # Solo/Multi: no trump, win exactly 1 OR exactly 5 tricks
    NEGEN_ALLEEN  = 5    # Solo: choose trump, win 9+ tricks
    MISERE        = 6    # Solo/Multi: no trump, win exactly 0 tricks
    TIEN_ALLEEN   = 7    # Solo: choose trump, win 10+ tricks
    OPEN_PIEK     = 8    # Like Piek, but cards face-up after trick 1
    OPEN_MISERE   = 9    # Like Misère, but cards face-up
    ELF_ALLEEN    = 10   # Solo: choose trump, win 11+ tricks
    TWAALF_ALLEEN = 11   # Solo: choose trump, win 12+ tricks
    SOLO_SLIM     = 12   # Solo: choose trump, win all 13 tricks
    TROELA        = 13   # Partner (3 Aces): 4th Ace holder is partner, 8+ tricks

    @classmethod
    def is_trump_contract(cls, c: 'Contract') -> bool:
        return c in (cls.RIK, cls.RIK_BETER, cls.ACHT_ALLEEN, cls.NEGEN_ALLEEN,
                     cls.TIEN_ALLEEN, cls.ELF_ALLEEN, cls.TWAALF_ALLEEN,
                     cls.SOLO_SLIM, cls.TROELA)

    @classmethod
    def is_solo(cls, c: 'Contract') -> bool:
        return c in (cls.ACHT_ALLEEN, cls.PIEK, cls.NEGEN_ALLEEN,
                     cls.MISERE, cls.TIEN_ALLEEN, cls.OPEN_PIEK,
                     cls.OPEN_MISERE, cls.ELF_ALLEEN, cls.TWAALF_ALLEEN,
                     cls.SOLO_SLIM)

    @classmethod
    def is_partner_contract(cls, c: 'Contract') -> bool:
        return c in (cls.RIK, cls.RIK_BETER, cls.TROELA)

    @classmethod
    def is_multi_player_allowed(cls, c: 'Contract') -> bool:
        """Whether multiple players can simultaneously bid and play this contract."""
        return c in (cls.PIEK, cls.MISERE, cls.OPEN_PIEK, cls.OPEN_MISERE)

    @classmethod
    def is_open(cls, c: 'Contract') -> bool:
        return c in (cls.OPEN_PIEK, cls.OPEN_MISERE)

    @classmethod
    def target_tricks(cls, c: 'Contract') -> Optional[int]:
        """Minimum tricks the declarer must win (-1 for exact-count contracts)."""
        table = {
            cls.RIK:           8,
            cls.RIK_BETER:     8,
            cls.ACHT_ALLEEN:   8,
            cls.NEGEN_ALLEEN:  9,
            cls.TIEN_ALLEEN:  10,
            cls.ELF_ALLEEN:   11,
            cls.TWAALF_ALLEEN:12,
            cls.SOLO_SLIM:    13,
            cls.TROELA:        8,
            cls.PIEK:         -1,   # Exact: 1 or 5 tricks
            cls.MISERE:        0,   # Exact: 0 tricks
            cls.OPEN_PIEK:    -1,   # Exact: 1 or 5 tricks
            cls.OPEN_MISERE:   0,   # Exact: 0 tricks
        }
        return table.get(c, None)

    @classmethod
    def name(cls, c: int | 'Contract') -> str:
        try:
            return cls(c).name
        except ValueError:
            return f"UNKNOWN({c})"


# ---------------------------------------------------------------------------
# Phase Enum
# ---------------------------------------------------------------------------
class Phase(IntEnum):
    BIDDING      = 0
    TRICK_TAKING = 1
    TERMINAL     = 2


# ---------------------------------------------------------------------------
# RikkenState
# ---------------------------------------------------------------------------
@dataclass
class RikkenState:
    """
    Complete game state for one hand of Rikken.

    Invariants:
      - hands[p] is the private cards currently held by player p.
      - played_cards is public: all cards that have been played in previous tricks.
      - current_trick[p] contains the card played by p in the current trick (-1 if not yet played).
      - void_matrix[p, s] is True iff it is publicly known that player p holds no cards of suit s.
    """

    # -------------------------------------------------------------------------
    # Card state (private + public)
    # -------------------------------------------------------------------------
    hands: np.ndarray           # shape (4, 52), dtype int8 — private (one-hot)
    played_cards: np.ndarray    # shape (52,),   dtype int8 — public (one-hot)
    current_trick: np.ndarray   # shape (4,),    dtype int8 — card per seat (-1=not played)
    cards_face_up: np.ndarray   # shape (4, 52), dtype int8 — face-up for Open contracts

    # -------------------------------------------------------------------------
    # Trick metadata
    # -------------------------------------------------------------------------
    trick_leader: int           # Seat index of the player who led this trick
    current_player: int         # Whose turn to act right now
    trick_count: int            # Number of completed tricks (0..13)
    tricks_won: np.ndarray      # shape (4,), dtype int8 — tricks won per player

    # -------------------------------------------------------------------------
    # Phase
    # -------------------------------------------------------------------------
    phase: Phase                # BIDDING | TRICK_TAKING | TERMINAL

    # -------------------------------------------------------------------------
    # Bidding state
    # -------------------------------------------------------------------------
    bids: np.ndarray            # shape (4,), dtype int8 — Contract value or -1
    highest_bid: Contract       # Current winning bid
    bid_winner: int             # Seat of primary/first contract winner (-1 if not yet determined)
    passed: np.ndarray          # shape (4,), dtype bool — whether player has passed
    pass_count: int             # Consecutive passes in bidding (for redeal detection)

    # -------------------------------------------------------------------------
    # Contract state
    # -------------------------------------------------------------------------
    contract: Contract          # Winning contract type
    trump_suit: int             # Trump suit index (0-3), or -1 if no trump
    declarer: int               # Seat of the primary declaring player (-1 if not set)
    declarer_mask: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=bool)) # True for all active declarers (supports multi-player Misere/Piek)
    partner: int = -1           # Seat of partner (-1 for solo / multi-player contracts)

    # -------------------------------------------------------------------------
    # Rik / Troela specifics
    # -------------------------------------------------------------------------
    vraagaas_suit: int = -1     # Suit of the requested Ace (-1 if N/A)
    aasvragen_triggered: bool = False # True once Declarer has led vraagaas_suit
    partner_revealed: bool = False    # True once partner's identity is publicly known

    # -------------------------------------------------------------------------
    # Inference / Void matrix
    # -------------------------------------------------------------------------
    void_matrix: np.ndarray = field(default_factory=lambda: np.zeros((4, 4), dtype=bool)) # [player, suit]

    # -------------------------------------------------------------------------
    # Deck ordering (for clumping shuffle in next deal)
    # -------------------------------------------------------------------------
    trick_sequence: list = field(default_factory=list) # Ordered list of (winner, [card0,card1,card2,card3])

    # -------------------------------------------------------------------------
    # Terminal payoffs
    # -------------------------------------------------------------------------
    reward: Optional[float] = None # Primary declarer reward (+1.0 / -1.0), None if ongoing
    rewards: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32)) # Individual payoff per seat (-1.0 to +1.0)

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------
    @classmethod
    def initial(cls) -> 'RikkenState':
        """Create a blank state before dealing. Use RikkenGame.reset() in practice."""
        return cls(
            hands=np.zeros((4, 52), dtype=np.int8),
            played_cards=np.zeros(52, dtype=np.int8),
            current_trick=np.full(4, -1, dtype=np.int8),
            cards_face_up=np.zeros((4, 52), dtype=np.int8),
            trick_leader=0,
            current_player=0,
            trick_count=0,
            tricks_won=np.zeros(4, dtype=np.int8),
            phase=Phase.BIDDING,
            bids=np.full(4, -1, dtype=np.int8),
            highest_bid=Contract.PAS,
            bid_winner=-1,
            passed=np.zeros(4, dtype=bool),
            pass_count=0,
            contract=Contract.NO_BID,
            trump_suit=-1,
            declarer=-1,
            declarer_mask=np.zeros(4, dtype=bool),
            partner=-1,
            vraagaas_suit=-1,
            aasvragen_triggered=False,
            partner_revealed=False,
            void_matrix=np.zeros((4, 4), dtype=bool),
            trick_sequence=[],
            reward=None,
            rewards=np.zeros(4, dtype=np.float32),
        )

    def copy(self) -> 'RikkenState':
        """Fast shallow/array copy of state for ISMCTS tree search."""
        s = RikkenState(
            hands=self.hands.copy(),
            played_cards=self.played_cards.copy(),
            current_trick=self.current_trick.copy(),
            cards_face_up=self.cards_face_up.copy(),
            trick_leader=self.trick_leader,
            current_player=self.current_player,
            trick_count=self.trick_count,
            tricks_won=self.tricks_won.copy(),
            phase=self.phase,
            bids=self.bids.copy(),
            highest_bid=self.highest_bid,
            bid_winner=self.bid_winner,
            passed=self.passed.copy(),
            pass_count=self.pass_count,
            contract=self.contract,
            trump_suit=self.trump_suit,
            declarer=self.declarer,
            declarer_mask=self.declarer_mask.copy(),
            partner=self.partner,
            vraagaas_suit=self.vraagaas_suit,
            aasvragen_triggered=self.aasvragen_triggered,
            partner_revealed=self.partner_revealed,
            void_matrix=self.void_matrix.copy(),
            trick_sequence=list(self.trick_sequence),
            reward=self.reward,
            rewards=self.rewards.copy(),
        )
        return s

    @property
    def declarer_tricks(self) -> int:
        """Total tricks won by the declaring side (declarer + partner if applicable)."""
        if self.declarer < 0:
            return 0
        total = int(self.tricks_won[self.declarer])
        if self.partner >= 0 and self.partner != self.declarer:
            total += int(self.tricks_won[self.partner])
        return total

    @property
    def defender_tricks(self) -> int:
        """Total tricks won by the defending side."""
        return int(self.trick_count) - self.declarer_tricks

    @property
    def remaining_tricks(self) -> int:
        """Tricks left to be played in this hand."""
        return 13 - int(self.trick_count)

    @property
    def is_terminal(self) -> bool:
        return self.phase == Phase.TERMINAL
