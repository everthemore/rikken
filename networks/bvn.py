"""
networks/bvn.py — Bidding Value Network (BVN).

Architecture: Transformer encoder -> MLP head.
Updated to 14 contracts: PAS(0) through TROELA(13).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

import config

NUM_CONTRACTS = 14      # PAS(0) through TROELA(13)
NUM_PLAYERS   = 4
HAND_SIZE     = 52
BID_HIST_SIZE = NUM_PLAYERS * NUM_CONTRACTS   # 4 * 14 = 56
INPUT_SIZE    = HAND_SIZE + BID_HIST_SIZE     # 108


class BVN(nn.Module):
    """
    Bidding Value Network.
    """

    def __init__(
        self,
        d_model: int = config.BVN_D_MODEL,
        nhead: int = config.BVN_NHEAD,
        num_layers: int = config.BVN_NUM_LAYERS,
        mlp_hidden: int = config.BVN_MLP_HIDDEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(INPUT_SIZE, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, NUM_CONTRACTS),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        hands: torch.Tensor,
        bid_hist: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        batch_size = hands.shape[0]
        x = torch.cat([hands, bid_hist], dim=-1)
        h = self.input_proj(x).unsqueeze(1)
        h = self.transformer(h).squeeze(1)
        logits = self.head(h)
        return logits, None

    def predict_ev(
        self,
        hand: np.ndarray,
        bids: np.ndarray,
        device: str = config.DEVICE,
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            hand_t = torch.tensor(hand, dtype=torch.float32, device=device).unsqueeze(0)
            bid_hist_1hot = np.zeros((NUM_PLAYERS, NUM_CONTRACTS), dtype=np.float32)
            for p in range(NUM_PLAYERS):
                b = bids[p]
                if 0 <= b < NUM_CONTRACTS:
                    bid_hist_1hot[p, b] = 1.0
            bid_hist_t = torch.tensor(bid_hist_1hot.flatten(), dtype=torch.float32, device=device).unsqueeze(0)

            logits, _ = self.forward(hand_t, bid_hist_t)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            return probs


class BVNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        bid_taken: torch.Tensor,
        outcome: torch.Tensor,
    ) -> torch.Tensor:
        taken_logits = logits.gather(1, bid_taken.unsqueeze(1)).squeeze(1)
        return self.bce(taken_logits, outcome.float())
