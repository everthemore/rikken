"""
networks/bvn.py — Bidding Value Network (BVN) Dual-Head Architecture.

Dual-Head Design:
  1. Win-Probability Head: Sigmoid activation in [0.0, 1.0] representing
     P(Win | hand, contract) for direct, human-interpretable win chances.
  2. Expected-Value Head: Tanh activation in [-1.0, +1.0] representing
     Expected Game Points for strategic and risk assessment.

Supports dynamic contract sizing (14 or 15) and legacy checkpoint compatibility.
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

import config

NUM_CONTRACTS = 15      # PAS(0) through MOELA(14)
NUM_PLAYERS   = 4
HAND_SIZE     = 52
BID_HIST_SIZE = NUM_PLAYERS * NUM_CONTRACTS   # 4 * 15 = 60
INPUT_SIZE    = HAND_SIZE + BID_HIST_SIZE     # 112


class BVN(nn.Module):
    """
    Dual-Head Bidding Action-Value Network:
      - Head 1: P(Win | s, a) in [0.0, 1.0] (Win Probability)
      - Head 2: Q(s, a) in [-1.0, +1.0] (Expected Game Points)
    """

    def __init__(
        self,
        num_contracts: int = NUM_CONTRACTS,
        d_model: int = config.BVN_D_MODEL,
        nhead: int = config.BVN_NHEAD,
        num_layers: int = config.BVN_NUM_LAYERS,
        mlp_hidden: int = config.BVN_MLP_HIDDEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_contracts = num_contracts
        self.input_size = HAND_SIZE + NUM_PLAYERS * num_contracts
        self.d_model = d_model

        self.input_proj = nn.Linear(self.input_size, d_model)

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

        # Head 1: Win Probability P(Win | hand, c) in [0.0, 1.0]
        self.win_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_contracts),
            nn.Sigmoid(),
        )

        # Head 2: Expected Points EV in [-1.0, +1.0]
        self.ev_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_contracts),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @classmethod
    def from_checkpoint(cls, ckpt_path_or_dict, device="cpu") -> 'BVN':
        if isinstance(ckpt_path_or_dict, str):
            ckpt = torch.load(ckpt_path_or_dict, map_location=device, weights_only=False)
        else:
            ckpt = ckpt_path_or_dict
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

        # Determine contract size
        if 'win_head.4.weight' in sd:
            n_c = sd['win_head.4.weight'].shape[0]
        elif 'ev_head.4.weight' in sd:
            n_c = sd['ev_head.4.weight'].shape[0]
        elif 'head.4.weight' in sd:
            n_c = sd['head.4.weight'].shape[0]
        else:
            n_c = NUM_CONTRACTS

        model = cls(num_contracts=n_c).to(device)

        # Handle legacy single-head checkpoints gracefully
        if 'head.4.weight' in sd and 'win_head.4.weight' not in sd:
            legacy_sd = {}
            for k, v in sd.items():
                if k.startswith('head.'):
                    legacy_sd[k.replace('head.', 'ev_head.')] = v
                else:
                    legacy_sd[k] = v
            model.load_state_dict(legacy_sd, strict=False)
            # Initialize win_head to produce ~50% prior probabilities
            with torch.no_grad():
                model.win_head[4].weight.zero_()
                model.win_head[4].bias.zero_()
        else:
            model.load_state_dict(sd)

        model.eval()
        return model

    def forward(
        self,
        hands: torch.Tensor,
        bid_hist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple of:
              - win_probs: (B, num_contracts) float32 in [0.0, 1.0]
              - ev_scores: (B, num_contracts) float32 in [-1.0, +1.0]
        """
        if bid_hist.dim() > 2:
            bid_hist = bid_hist.flatten(start_dim=1)
        if hands.dim() > 2:
            hands = hands.flatten(start_dim=1)

        x = torch.cat([hands, bid_hist], dim=-1)
        h = self.input_proj(x).unsqueeze(1)
        h = self.transformer(h).squeeze(1)

        win_probs = self.win_head(h)
        ev_scores = self.ev_head(h)
        return win_probs, ev_scores

    def predict(
        self,
        hand: np.ndarray,
        bids: np.ndarray,
        device: str = config.DEVICE,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inference helper.

        Returns:
            win_probs: (NUM_CONTRACTS,) numpy array in [0.0, 1.0]
            ev_scores: (NUM_CONTRACTS,) numpy array in [-1.0, +1.0]
        """
        self.eval()
        with torch.no_grad():
            hand_t = torch.tensor(hand, dtype=torch.float32, device=device).unsqueeze(0)
            bid_hist_1hot = np.zeros((NUM_PLAYERS, self.num_contracts), dtype=np.float32)
            for p in range(NUM_PLAYERS):
                b = bids[p]
                if 0 <= b < self.num_contracts:
                    bid_hist_1hot[p, b] = 1.0
            bid_hist_t = torch.tensor(bid_hist_1hot.flatten(), dtype=torch.float32, device=device).unsqueeze(0)

            win_p, ev = self.forward(hand_t, bid_hist_t)
            win_res = win_p.squeeze(0).cpu().numpy()
            ev_res  = ev.squeeze(0).cpu().numpy()

            if self.num_contracts < NUM_CONTRACTS:
                full_win = np.full(NUM_CONTRACTS, 0.0, dtype=np.float32)
                full_ev  = np.full(NUM_CONTRACTS, -1.0, dtype=np.float32)
                full_win[:self.num_contracts] = win_res
                full_ev[:self.num_contracts]  = ev_res
                return full_win, full_ev

            return win_res, ev_res

    def predict_ev(
        self,
        hand: np.ndarray,
        bids: np.ndarray,
        device: str = config.DEVICE,
    ) -> np.ndarray:
        """Backward compatibility for legacy code querying EV only."""
        _, ev = self.predict(hand, bids, device=device)
        return ev

    def predict_win_prob(
        self,
        hand: np.ndarray,
        bids: np.ndarray,
        device: str = config.DEVICE,
    ) -> np.ndarray:
        """Helper querying Win Probabilities only."""
        win_p, _ = self.predict(hand, bids, device=device)
        return win_p


class BVNDualLoss(nn.Module):
    """
    Joint loss function for Dual-Head BVN:
      - Binary Cross-Entropy (BCE) on Win Probability head
      - Smooth L1 (Huber) regression on Expected Points head
    """

    def __init__(self, beta: float = 0.1, alpha_ev: float = 0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.huber = nn.SmoothL1Loss(beta=beta)
        self.alpha_ev = alpha_ev

    def forward(
        self,
        win_probs: torch.Tensor,
        ev_scores: torch.Tensor,
        bid_taken: torch.Tensor,
        won: torch.Tensor,
        outcome: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        taken_win = win_probs.gather(1, bid_taken.unsqueeze(1)).squeeze(1)
        taken_ev  = ev_scores.gather(1, bid_taken.unsqueeze(1)).squeeze(1)

        loss_win = self.bce(taken_win, won.float())
        loss_ev  = self.huber(taken_ev, outcome.float())
        total_loss = loss_win + self.alpha_ev * loss_ev
        return total_loss, loss_win, loss_ev


# Alias for compatibility
BVNLoss = BVNDualLoss
