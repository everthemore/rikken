"""
training/train_bn.py — Memory-efficient Supervised training loop for the Belief Network.

Features:
  - Streaming Shard Dataset (constant ~20 MB RAM footprint).
  - Complete checkpointing (`bn_checkpoint.pt`, `bn_best.pt`, `bn_final.pt`).
  - Supports Apple Silicon MPS, CUDA, and CPU.
"""

from __future__ import annotations
import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
import time

from networks.bn import BeliefNetwork, BNLoss
from training.dataset import build_dataset
import config

log = logging.getLogger(__name__)


def train(
    epochs: int = config.NUM_EPOCHS_BN,
    lr: float = config.LEARNING_RATE,
    batch_size: int = config.BATCH_SIZE,
    data_path: str = config.DATA_PATH,
    model_path: str = config.MODEL_PATH,
    device: str = config.DEVICE,
    resume: str = '',
    resume_latest: bool = False,
) -> BeliefNetwork:
    """Memory-efficient supervised training loop for the Belief Network."""

    os.makedirs(model_path, exist_ok=True)

    if device == 'cpu' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        log.info("Detected Apple MPS (Metal Performance Shaders) — using 'mps' device.")

    log.info(f"Using device: {device}")

    # --- Streaming Datasets ---
    train_ds = build_dataset(data_path, mode='bn', split='train')
    val_ds   = build_dataset(data_path, mode='bn', split='val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, pin_memory=(device != 'cpu'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=1, pin_memory=(device != 'cpu'))

    total_train_samples = len(train_ds)
    total_val_samples   = len(val_ds)
    log.info(f"Dataset: {total_train_samples:,} train samples, {total_val_samples:,} val samples")

    # --- Model, Optimizer, Scheduler ---
    model = BeliefNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = BNLoss()

    start_epoch = 1
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    latest_ckpt_path = os.path.join(model_path, 'bn_checkpoint.pt')
    if resume_latest and os.path.exists(latest_ckpt_path):
        resume = latest_ckpt_path

    if resume and os.path.exists(resume):
        checkpoint = torch.load(resume, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_val_loss = float('inf')  # Fresh best tracking for this generation
            log.info(f"Resumed model weights and optimizer state from {resume}")
        else:
            model.load_state_dict(checkpoint)
            log.info(f"Loaded model weights only from {resume}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        t0 = time.time()

        for own, played, bid_hist, trick, void_mat, opp_hands in train_loader:
            own, played, bid_hist, trick, void_mat, opp_hands = (
                own.to(device), played.to(device), bid_hist.to(device),
                trick.to(device), void_mat.to(device), opp_hands.to(device)
            )

            hard_mask = (1 - own) * (1 - played)

            optimizer.zero_grad()
            preds = model(
                own_hand=own,
                played_cards=played,
                bid_history=bid_hist.view(bid_hist.shape[0], -1),
                current_trick=trick,
                void_matrix=void_mat.view(void_mat.shape[0], -1),
                hard_mask=hard_mask,
            )
            loss = criterion(preds, opp_hands.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = len(own)
            train_loss += loss.item() * bs
            train_count += bs

        train_loss /= max(train_count, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for own, played, bid_hist, trick, void_mat, opp_hands in val_loader:
                own, played, bid_hist, trick, void_mat, opp_hands = (
                    own.to(device), played.to(device), bid_hist.to(device),
                    trick.to(device), void_mat.to(device), opp_hands.to(device)
                )
                hard_mask = (1 - own) * (1 - played)
                preds = model(
                    own_hand=own,
                    played_cards=played,
                    bid_history=bid_hist.view(bid_hist.shape[0], -1),
                    current_trick=trick,
                    void_matrix=void_mat.view(void_mat.shape[0], -1),
                    hard_mask=hard_mask,
                )
                loss = criterion(preds, opp_hands.float())
                bs = len(own)
                val_loss += loss.item() * bs
                val_count += bs

        val_loss /= max(val_count, 1)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train {train_loss:.4f} | "
            f"val {val_loss:.4f} | "
            f"{elapsed:.1f}s"
        )

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'history': history,
        }
        torch.save(checkpoint_data, latest_ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_data['best_val_loss'] = best_val_loss
            ckpt = os.path.join(model_path, 'bn_best.pt')
            torch.save(checkpoint_data, ckpt)
            log.info(f"  → Saved best checkpoint: {ckpt}")

    torch.save(model.state_dict(), os.path.join(model_path, 'bn_final.pt'))
    np.save(os.path.join(model_path, 'bn_history.npy'), history)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Belief Network')
    parser.add_argument('--epochs',        type=int,   default=config.NUM_EPOCHS_BN)
    parser.add_argument('--lr',            type=float, default=config.LEARNING_RATE)
    parser.add_argument('--batch-size',    type=int,   default=config.BATCH_SIZE)
    parser.add_argument('--data-path',     type=str,   default=config.DATA_PATH)
    parser.add_argument('--model-path',    type=str,   default=config.MODEL_PATH)
    parser.add_argument('--resume',        type=str,   default='')
    parser.add_argument('--resume-latest', action='store_true', help='Resume automatically from bn_checkpoint.pt')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(config.TORCH_SEED)
    train(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        data_path=args.data_path, model_path=args.model_path,
        resume=args.resume, resume_latest=args.resume_latest,
    )
