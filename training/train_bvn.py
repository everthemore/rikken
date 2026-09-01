"""
training/train_bvn.py — Supervised training loop for the Bidding Value Network with full checkpointing.

Features:
  - Saves best checkpoint (`bvn_best.pt`), latest checkpoint (`bvn_checkpoint.pt`), and final (`bvn_final.pt`).
  - Checkpoints store model weights, optimizer state, scheduler state, current epoch, best validation loss, and loss history.
  - Resume easily from any checkpoint using `--resume <path>` or automatically with `--resume-latest`.
  - Supports Apple Silicon MPS, CUDA, and CPU automatically.

Usage:
    python -m training.train_bvn [--epochs N] [--lr LR] [--resume PATH] [--resume-latest]
"""

from __future__ import annotations
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import logging
import time

from networks.bvn import BVN, BVNLoss
from training.dataset import build_dataset
import config

log = logging.getLogger(__name__)


def train(
    epochs: int = config.NUM_EPOCHS_BVN,
    lr: float = config.LEARNING_RATE,
    batch_size: int = config.BATCH_SIZE,
    data_path: str = config.DATA_PATH,
    model_path: str = config.MODEL_PATH,
    device: str = config.DEVICE,
    resume: str = '',
    resume_latest: bool = False,
) -> BVN:
    """Full supervised training loop for the BVN with complete state checkpointing."""

    os.makedirs(model_path, exist_ok=True)

    # Detect MPS device on Apple Silicon if device is not explicitly set to cuda
    if device == 'cpu' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        log.info("Detected Apple MPS (Metal Performance Shaders) — using 'mps' device.")

    log.info(f"Using device: {device}")

    # --- Data ---
    train_ds = build_dataset(data_path, mode='bvn', split='train')
    val_ds   = build_dataset(data_path, mode='bvn', split='val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, pin_memory=(device != 'cpu'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=1, pin_memory=(device != 'cpu'))

    # --- Model, Optimizer, Scheduler ---
    model = BVN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = BVNLoss()

    start_epoch = 1
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    latest_ckpt_path = os.path.join(model_path, 'bvn_checkpoint.pt')
    if resume_latest and os.path.exists(latest_ckpt_path):
        resume = latest_ckpt_path

    if resume and os.path.exists(resume):
        try:
            checkpoint = torch.load(resume, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                history = checkpoint.get('history', history)
                log.info(f"Resumed complete state from {resume} (starting at epoch {start_epoch})")
            else:
                model.load_state_dict(checkpoint)
                log.info(f"Loaded model weights only from {resume}")
        except Exception as e:
            log.warning(f"Could not load checkpoint {resume} ({e}) — starting fresh model weights.")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        t0 = time.time()

        for hands, bid_hist, bid_taken, outcome in train_loader:
            hands, bid_hist, bid_taken, outcome = (
                hands.to(device), bid_hist.to(device),
                bid_taken.to(device), outcome.to(device)
            )

            optimizer.zero_grad()
            logits, _ = model(hands, bid_hist)
            loss = criterion(logits, bid_taken, outcome)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = len(hands)
            train_loss += loss.item() * bs
            train_count += bs

        train_loss /= max(train_count, 1)

        # --- Validate ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for hands, bid_hist, bid_taken, outcome in val_loader:
                hands, bid_hist, bid_taken, outcome = (
                    hands.to(device), bid_hist.to(device),
                    bid_taken.to(device), outcome.to(device)
                )
                logits, _ = model(hands, bid_hist)
                loss = criterion(logits, bid_taken, outcome)
                val_loss += loss.item() * len(hands)

                taken_logit = logits.gather(1, bid_taken.unsqueeze(1)).squeeze(1)
                preds = torch.sign(taken_logit)
                target_sign = torch.sign(outcome)
                correct += (preds == target_sign).sum().item()
                total += len(outcome)

        val_loss /= max(total, 1)
        val_acc   = correct / max(total, 1)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train {train_loss:.4f} | "
            f"val {val_loss:.4f} | "
            f"acc {val_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        # Save latest rolling checkpoint
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
            ckpt = os.path.join(model_path, 'bvn_best.pt')
            torch.save(checkpoint_data, ckpt)
            log.info(f"  → Saved best checkpoint: {ckpt}")

    # Save final
    torch.save(model.state_dict(), os.path.join(model_path, 'bvn_final.pt'))
    np.save(os.path.join(model_path, 'bvn_history.npy'), history)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Bidding Value Network')
    parser.add_argument('--epochs',        type=int,   default=config.NUM_EPOCHS_BVN)
    parser.add_argument('--lr',            type=float, default=config.LEARNING_RATE)
    parser.add_argument('--batch-size',    type=int,   default=config.BATCH_SIZE)
    parser.add_argument('--data-path',     type=str,   default=config.DATA_PATH)
    parser.add_argument('--model-path',    type=str,   default=config.MODEL_PATH)
    parser.add_argument('--resume',        type=str,   default='')
    parser.add_argument('--resume-latest', action='store_true', help='Resume automatically from bvn_checkpoint.pt')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(config.TORCH_SEED)
    train(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        data_path=args.data_path, model_path=args.model_path,
        resume=args.resume, resume_latest=args.resume_latest,
    )
