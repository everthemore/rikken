"""
networks/bn.py — Belief Network (BN).

Architecture: Linear input projection → ResidualBlocks → 3 per-opponent heads.

Purpose: Given the public state (played cards, bidding history, void matrix,
current trick) and the player's own hand, predict the probability distribution
of hidden cards for each of the 3 opponents.

Output:
  - 3 tensors of shape (52,) — one per opponent, each being a probability
    in [0, 1] that opponent holds card c.
  - Hard-masked by the Void Matrix before normalization (zeroing impossibles).
  - Also zeroed for cards already played or in the player's own hand.

Training signal:
  - Binary cross-entropy vs. actual card holdings at each game state step.

XAI note: For the Attention Inspection experiment, the BN can be upgraded to
a Transformer architecture by replacing `ResidualBlock` with `TransformerEncoder`.
This is flagged with a TODO comment in the code.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

import config

NUM_OPPONENTS = 3
HAND_SIZE     = 52
NUM_CONTRACTS = 14
BID_HIST_SIZE = 4 * NUM_CONTRACTS  # 56
INPUT_SIZE    = (
    HAND_SIZE           # own remaining hand (52)
    + HAND_SIZE         # played cards (public) (52)
    + BID_HIST_SIZE     # bid history one-hot (4 players × 14 contracts = 56)
    + HAND_SIZE         # current trick (52)
    + 4 * 4             # void matrix flattened (16)
)  # = 52+52+56+52+16 = 228


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Pre-activation residual block: LayerNorm → Linear → GELU → Dropout → Linear.

    TODO (XAI upgrade): Replace this block with a TransformerEncoderLayer
    to enable attention weight visualization for the Belief Network.
    """

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Belief Network
# ---------------------------------------------------------------------------

class BeliefNetwork(nn.Module):
    """
    Belief Network predicting card distributions for 3 opponents.

    Args:
        hidden:     Hidden dimension for residual blocks.
        num_blocks: Number of residual blocks.
        dropout:    Dropout rate.
    """

    def __init__(
        self,
        hidden: int = config.BN_HIDDEN,
        num_blocks: int = config.BN_NUM_BLOCKS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden = hidden

        # --- Input projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # --- Shared residual trunk ---
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(num_blocks)]
        )

        # --- Per-opponent output heads ---
        # Each head predicts P(opponent p holds card c) for c in 0..51
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, HAND_SIZE),
            )
            for _ in range(NUM_OPPONENTS)
        ])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        own_hand: torch.Tensor,         # (B, 52) float32
        played_cards: torch.Tensor,     # (B, 52) float32
        bid_history: torch.Tensor,      # (B, 52) float32  [4×13 flattened]
        current_trick: torch.Tensor,    # (B, 52) float32
        void_matrix: torch.Tensor,      # (B, 16) float32  [4×4 flattened]
        hard_mask: Optional[torch.Tensor] = None,  # (B, 52) bool — global impossibles
    ) -> List[torch.Tensor]:
        """
        Forward pass.

        Returns:
            List of 3 tensors, each (B, 52) float32 — probability per card per opponent.
            Already masked and normalized (each tensor sums to ~opponent_hand_size).
        """
        B = own_hand.shape[0]

        if bid_history.dim() > 2:
            bid_history = bid_history.flatten(start_dim=1)
        if void_matrix.dim() > 2:
            void_matrix = void_matrix.flatten(start_dim=1)
        if own_hand.dim() > 2:
            own_hand = own_hand.flatten(start_dim=1)
        if played_cards.dim() > 2:
            played_cards = played_cards.flatten(start_dim=1)
        if current_trick.dim() > 2:
            current_trick = current_trick.flatten(start_dim=1)

        # Concatenate all features: (B, 228)
        x = torch.cat([own_hand, played_cards, bid_history, current_trick, void_matrix], dim=-1)

        # Project and run trunk
        x = self.input_proj(x)          # (B, hidden)
        x = self.trunk(x)               # (B, hidden)

        # Per-opponent outputs
        outputs = []
        for head in self.heads:
            logits = head(x)            # (B, 52)
            probs = torch.sigmoid(logits)   # (B, 52) — independent Bernoulli per card

            # Hard mask: zero out globally impossible cards
            if hard_mask is not None:
                probs = probs * hard_mask.float()

            outputs.append(probs)

        return outputs

    def predict(
        self,
        own_hand: np.ndarray,           # int8[52]
        played_cards: np.ndarray,       # int8[52]
        bid_history: np.ndarray,        # int8[4, 13]
        current_trick: np.ndarray,      # int8[52] (one-hot of played cards)
        void_matrix: np.ndarray,        # bool[4, 4]
        my_seat: int,                   # which seat is "me" (0-3)
        device: str = config.DEVICE,
    ) -> List[np.ndarray]:
        """
        Inference convenience method.

        Returns:
            List of 3 float32[52] arrays — one per opponent (in seat order,
            excluding own seat).
        """
        self.eval()
        if bid_history.ndim == 1 and len(bid_history) == 4:
            bids_oh = np.zeros((4, 14), dtype=np.float32)
            for p_idx, b in enumerate(bid_history):
                if 0 <= b < 14:
                    bids_oh[p_idx, b] = 1.0
            bid_history = bids_oh
        elif bid_history.size == 52:
            # Pad 52 to 56 for 14 contracts
            pad = np.zeros(4, dtype=np.float32)
            bid_history = np.concatenate([bid_history.flatten(), pad])

        with torch.no_grad():
            # Compute hard mask: cards that cannot be in any opponent's hand
            impossible = own_hand.astype(bool) | played_cards.astype(bool)
            # Also zero out cards in the current trick
            for c in np.where(current_trick)[0]:
                impossible[c] = True
            hard_mask = torch.tensor(~impossible, dtype=torch.float32, device=device).unsqueeze(0)

            def t(arr, dtype=torch.float32):
                return torch.tensor(arr, dtype=dtype, device=device).unsqueeze(0)

            outs = self.forward(
                own_hand=t(own_hand.astype(np.float32)),
                played_cards=t(played_cards.astype(np.float32)),
                bid_history=t(bid_history.flatten().astype(np.float32)),
                current_trick=t(current_trick.astype(np.float32)),
                void_matrix=t(void_matrix.flatten().astype(np.float32)),
                hard_mask=hard_mask,
            )
            return [o.squeeze(0).cpu().numpy() for o in outs]

    def sample_opponent_hand(
        self,
        probs: np.ndarray,      # float32[52] — marginal probabilities
        n_cards: int,           # how many cards opponent should hold
        void_mask: np.ndarray,  # bool[4] — void suits for this opponent
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample a consistent hand for one opponent from the BN's probability output.

        Ensures:
          - Exactly `n_cards` cards are sampled.
          - No card from a void suit is included.

        Uses sequential sampling with renormalization (Plackett-Luce style).
        """
        from engine.card import suit_of, SUIT_MASKS

        p = probs.copy()

        # Zero out void suits and already-known zeros
        for suit in range(4):
            if void_mask[suit]:
                p[SUIT_MASKS[suit]] = 0.0

        hand = np.zeros(52, dtype=np.int8)
        remaining = n_cards
        available = np.where(p > 0)[0]

        for _ in range(n_cards):
            if len(available) == 0:
                break
            # Normalize and sample
            p_sub = p[available]
            total = p_sub.sum()
            if total <= 0:
                break
            p_norm = p_sub / total
            chosen_idx = rng.choice(len(available), p=p_norm)
            card = available[chosen_idx]
            hand[card] = 1
            p[card] = 0.0
            available = np.where(p > 0)[0]

        return hand


class BNLoss(nn.Module):
    """
    Training loss for BN: Binary cross-entropy per card per opponent.

    For each training step, we have the true one-hot hands of all 3 opponents.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predictions: List[torch.Tensor],    # List of 3 × (B, 52)
        targets: torch.Tensor,              # (B, 3, 52) float32 — true hands
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=targets.device)
        for i, pred in enumerate(predictions):
            target = targets[:, i, :]
            total_loss = total_loss + F.binary_cross_entropy(pred, target)
        return total_loss / NUM_OPPONENTS
