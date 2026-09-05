"""
training/train_imitation.py — Phase 1 Behavioral Cloning & Warm-Start from Heuristic Expert.

Generates natural Rikken deals, extracts heuristic bidding decisions, and pre-trains
the BVN with Cross-Entropy Loss to establish a human-like contract baseline:
  - RIK: ~35% on standard hands
  - PAS: ~50% on unsuited hands
  - Solo contracts: proportionally scaled to hand strength
  - OPEN_PIEK / OPEN_MISERE: ~0.0% on normal hands

Also generates 2,500 natural heuristic games to warm-start the Belief Network (BN)
on realistic card-play trajectories.

Usage:
    python -m training.train_imitation [--deals N] [--epochs N] [--batch-size N] [--device cuda|mps|cpu]
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from networks.bvn import BVN, NUM_CONTRACTS
from networks.bn import BeliefNetwork
from engine.state import RikkenState, Contract, Phase
from engine.game import RikkenGame
from agents.heuristic import HeuristicAgent
from training.data_gen import run_one_game, _pack_shard
from training.train_bn import train as train_bn
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def generate_heuristic_bidding_data(num_deals: int = 25000) -> tuple[np.ndarray, np.ndarray]:
    """Generates natural deals and collects (hand, heuristic_bid) pairs."""
    log.info(f"Generating {num_deals:,} deals ({num_deals * 4:,} hands) from Heuristic Expert...")
    t0 = time.time()
    game = RikkenGame()
    h = HeuristicAgent(0)

    hands = []
    bids = []

    for _ in range(num_deals):
        state = game.reset()
        for p in range(4):
            h.set_seat(p)
            sp = state.copy()
            sp.current_player = p
            bid = h.act(sp)
            hands.append(state.hands[p].astype(np.float32))
            bids.append(bid)

    dt = time.time() - t0
    log.info(f"Collected {len(hands):,} bidding samples in {dt:.1f}s ({len(hands)/max(dt, 1e-4):.0f} hands/s)")
    return np.array(hands, dtype=np.float32), np.array(bids, dtype=np.int64)


def train_bvn_imitation(
    hands: np.ndarray,
    bids: np.ndarray,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = config.DEVICE,
    model_path: str = "checkpoints",
) -> BVN:
    """Pre-trains BVN using Cross-Entropy Loss to imitate Heuristic bids."""
    os.makedirs(model_path, exist_ok=True)
    if device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    log.info(f"Training BVN Imitation on {device} ({len(hands):,} samples, {epochs} epochs)...")

    hands_t = torch.tensor(hands, dtype=torch.float32)
    bids_t = torch.tensor(bids, dtype=torch.long)
    bid_hist_t = torch.zeros((len(hands), 4, NUM_CONTRACTS), dtype=torch.float32)

    dataset = TensorDataset(hands_t, bids_t, bid_hist_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BVN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for b_hands, b_bids, b_bh in loader:
            b_hands, b_bids, b_bh = b_hands.to(device), b_bids.to(device), b_bh.to(device)
            optimizer.zero_grad()

            h_repr = model.input_proj(torch.cat([b_hands, b_bh.flatten(1)], dim=-1)).unsqueeze(1)
            h_repr = model.transformer(h_repr).squeeze(1)

            # Pass through win_head layers up to the final linear layer (pre-sigmoid logits)
            x = h_repr
            for layer in list(model.win_head.children())[:-1]:
                x = layer(x)
            logits = x

            loss = criterion(logits, b_bids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = len(b_hands)
            total_loss += loss.item() * bs
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == b_bids).sum().item()
            total += bs

        scheduler.step()
        avg_loss = total_loss / total
        acc = (correct / total) * 100
        log.info(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Imitation Accuracy: {acc:5.2f}%")

    # Save warm-start checkpoints
    ckpt_gen_0 = os.path.join(model_path, "bvn_gen_0.pt")
    ckpt_best  = os.path.join(model_path, "bvn_best.pt")
    torch.save({"model_state_dict": model.state_dict()}, ckpt_gen_0)
    torch.save({"model_state_dict": model.state_dict()}, ckpt_best)
    log.info(f"Saved warm-started BVN to {ckpt_gen_0} and {ckpt_best}")
    return model


def generate_heuristic_cardplay_data(
    num_games: int = 2500,
    output_dir: str = "data/imitation",
) -> str:
    """Generates natural heuristic games for Belief Network pre-training."""
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Generating {num_games:,} natural heuristic card-play games for Belief Network...")
    t0 = time.time()

    all_records = []
    for seed in range(num_games):
        res = run_one_game(seed)
        for rec in res.get("play_records", []):
            rec["type"] = "play"
            all_records.append(rec)

    shard_path = os.path.join(output_dir, "heuristic_play_shard.npz")
    _pack_shard(all_records, shard_path)
    dt = time.time() - t0
    log.info(f"Saved {len(all_records):,} card-play records to {shard_path} in {dt:.1f}s")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Heuristic Imitation Learning Warm-Start")
    parser.add_argument("--deals", type=int, default=25000, help="Number of deals to evaluate with Heuristic (default: 25000 = 100k hands)")
    parser.add_argument("--play-games", type=int, default=2500, help="Number of natural games for Belief Network (default: 2500)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument("--device", type=str, default=config.DEVICE)
    parser.add_argument("--model-path", type=str, default="checkpoints")
    args = parser.parse_args()

    print("=================================================================")
    print("  PHASE 1: HEURISTIC IMITATION LEARNING & WARM-START")
    print("=================================================================")

    # 1. Warm-start BVN Bidding Network
    hands, bids = generate_heuristic_bidding_data(num_deals=args.deals)
    train_bvn_imitation(
        hands=hands,
        bids=bids,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        model_path=args.model_path,
    )

    # 2. Warm-start Belief Network on natural play
    play_data_dir = generate_heuristic_cardplay_data(num_games=args.play_games)
    log.info("Pre-training Belief Network on natural play data...")
    bn_model = train_bn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_path=play_data_dir,
        model_path=args.model_path,
        device=args.device,
    )
    bn_gen_0 = os.path.join(args.model_path, "bn_gen_0.pt")
    bn_best  = os.path.join(args.model_path, "bn_best.pt")
    torch.save({"model_state_dict": bn_model.state_dict()}, bn_gen_0)
    torch.save({"model_state_dict": bn_model.state_dict()}, bn_best)
    log.info(f"Saved warm-started Belief Network to {bn_gen_0} and {bn_best}")

    print("\n=================================================================")
    print("  PHASE 1 IMITATION PRE-TRAINING COMPLETED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    main()
