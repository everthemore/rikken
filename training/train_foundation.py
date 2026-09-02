"""
training/train_foundation.py — Supervised Training on Stratified Counterfactual Dataset.

Trains BVN and Belief Network on balanced stratified contract data to establish
an accurate initial value function across all 15 Rikken contracts before
self-play policy iteration begins.
"""

from __future__ import annotations
import os
import shutil
import argparse
import torch
import logging

from networks.bvn import BVN, BVNLoss
from networks.bn import BeliefNetwork
from training.train_bvn import train as train_bvn
from training.train_bn import train as train_bn
import config

log = logging.getLogger(__name__)


def train_foundation_models(
    stratified_data_path: str = 'data/stratified',
    model_path: str = 'checkpoints',
    bvn_epochs: int = 15,
    bn_epochs: int = 15,
    batch_size: int = config.BATCH_SIZE,
    device: str = config.DEVICE,
) -> None:
    """Pre-trains BVN and BN on stratified data and saves foundation checkpoints."""
    os.makedirs(model_path, exist_ok=True)

    print("=================================================================")
    print("  PHASE 1: FOUNDATION PRE-TRAINING ON STRATIFIED CONTRACT DATA")
    print("=================================================================")
    print(f"Data source: {stratified_data_path}")
    print(f"Target model directory: {model_path}")

    # 1. Train BVN on balanced contract data
    print("\n--- 1. Training Foundational BVN Q-Network ---")
    bvn_model = train_bvn(
        epochs=bvn_epochs,
        batch_size=batch_size,
        data_path=stratified_data_path,
        model_path=model_path,
        device=device,
    )

    # Save as foundation checkpoint
    foundation_bvn_path = os.path.join(model_path, 'bvn_foundation.pt')
    torch.save({'model_state_dict': bvn_model.state_dict()}, foundation_bvn_path)
    shutil.copyfile(foundation_bvn_path, os.path.join(model_path, 'bvn_best.pt'))
    print(f"Saved foundational BVN to {foundation_bvn_path} and set as bvn_best.pt")

    # 2. Train Belief Network
    print("\n--- 2. Training Foundational Belief Network ---")
    bn_model = train_bn(
        epochs=bn_epochs,
        batch_size=batch_size,
        data_path=stratified_data_path,
        model_path=model_path,
        device=device,
    )

    foundation_bn_path = os.path.join(model_path, 'bn_foundation.pt')
    torch.save({'model_state_dict': bn_model.state_dict()}, foundation_bn_path)
    shutil.copyfile(foundation_bn_path, os.path.join(model_path, 'bn_best.pt'))
    print(f"Saved foundational BN to {foundation_bn_path} and set as bn_best.pt")

    print("\n=================================================================")
    print("  PHASE 1 FOUNDATION PRE-TRAINING SUCCESSFULLY COMPLETED!")
    print("=================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Foundational Stratified Pre-training")
    parser.add_argument('--data-path', type=str, default='data/stratified')
    parser.add_argument('--model-path', type=str, default='checkpoints')
    parser.add_argument('--bvn-epochs', type=int, default=15)
    parser.add_argument('--bn-epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--device', type=str, default=config.DEVICE)
    args = parser.parse_args()

    train_foundation_models(
        stratified_data_path=args.data_path,
        model_path=args.model_path,
        bvn_epochs=args.bvn_epochs,
        bn_epochs=args.bn_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
