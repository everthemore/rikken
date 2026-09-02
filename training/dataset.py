"""
training/dataset.py — High-performance Streaming Shard Dataset for BVN and BN.

Memory-efficient design:
  - Streams shards one at a time (peak RAM: ~5 MB per worker instead of 40 GB).
  - Keeps arrays in compact int8 format until batched.
  - Automatically handles multi-worker DataLoader partitioning and epoch shuffling.
"""
from __future__ import annotations
from networks.bvn import NUM_CONTRACTS


import os
import glob
import math
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Iterator, Tuple, List, Optional
import config


class StreamingShardDataset(IterableDataset):
    """
    Memory-efficient streaming dataset that loads .npz shards on-demand.

    Args:
        data_path:   Directory containing shard_*.npz files.
        mode:        'bvn' or 'bn'.
        split:       'train' (first 90% of shards) or 'val' (last 10%).
        train_frac:  Fraction of shards for training.
        shuffle_shards: Whether to shuffle shard order every epoch.
        seed:        Random seed for shard shuffling.
    """

    def __init__(
        self,
        data_path: str = config.DATA_PATH,
        mode: str = 'bn',
        split: str = 'train',
        train_frac: float = config.TRAIN_VAL_SPLIT,
        shuffle_shards: bool = True,
        seed: int = config.NUMPY_SEED,
    ):
        super().__init__()
        assert mode in ('bvn', 'bn')
        assert split in ('train', 'val')

        self.mode = mode
        self.split = split
        self.shuffle_shards = shuffle_shards
        self.seed = seed

        # Find all .npz shards across single or multiple directories (Replay Buffer)
        shard_files = []
        if isinstance(data_path, (list, tuple)):
            paths = data_path
        elif ',' in str(data_path):
            paths = [p.strip() for p in data_path.split(',') if p.strip()]
        else:
            paths = [data_path]

        for p in paths:
            if os.path.isfile(p) and p.endswith('.npz'):
                shard_files.append(p)
            elif os.path.isdir(p):
                shard_files.extend(glob.glob(os.path.join(p, '*.npz')))
                shard_files.extend(glob.glob(os.path.join(p, '**', '*.npz'), recursive=True))

        shard_files = sorted(list(set(shard_files)))
        if not shard_files:
            raise FileNotFoundError(f"No .npz shards found in {data_path}")

        if len(shard_files) == 1:
            self.shards = shard_files
        else:
            split_idx = max(1, int(len(shard_files) * train_frac))
            if split == 'train':
                self.shards = shard_files[:split_idx]
            else:
                self.shards = shard_files[split_idx:] if split_idx < len(shard_files) else shard_files[-1:]

        # Precompute total sample count from first shard
        if self.shards:
            sample_data = np.load(self.shards[0])
            if mode == 'bvn':
                samples_per_shard = len(sample_data['bvn_hands'])
            else:
                samples_per_shard = len(sample_data['bn_own'])
            self.total_samples = samples_per_shard * len(self.shards)
        else:
            self.total_samples = 0

    def __len__(self) -> int:
        return self.total_samples

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        # Worker partitioning for PyTorch DataLoader
        worker_info = torch.utils.data.get_worker_info()
        shards_to_process = list(self.shards)

        if self.shuffle_shards:
            rng = np.random.default_rng(self.seed + (0 if worker_info is None else worker_info.seed % 10000))
            rng.shuffle(shards_to_process)

        if worker_info is not None:
            # Partition shards among parallel DataLoader workers
            per_worker = int(math.ceil(len(shards_to_process) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(shards_to_process))
            shards_to_process = shards_to_process[iter_start:iter_end]

        for shard_path in shards_to_process:
            try:
                data = np.load(shard_path)
            except Exception:
                continue

            if self.mode == 'bvn':
                if 'bvn_hands' not in data:
                    continue
                hands = data['bvn_hands']
                bid_hist = data['bvn_bid_hist']
                bid_taken = data['bvn_bid_taken']
                outcome = data['bvn_outcome']

                n = len(hands)
                indices = np.arange(n)
                if self.shuffle_shards:
                    np.random.shuffle(indices)

                for idx in indices:
                    bh = bid_hist[idx].astype(np.float32)
                    if bh.size == 4 * 14:
                        bh_full = np.zeros((4, NUM_CONTRACTS), dtype=np.float32)
                        bh_full[:, :14] = bh.reshape(4, 14)
                        bh_t = torch.from_numpy(bh_full)
                    else:
                        bh_t = torch.from_numpy(bh).view(4, NUM_CONTRACTS)

                    yield (
                        torch.from_numpy(hands[idx]).float(),
                        bh_t,
                        torch.tensor(bid_taken[idx], dtype=torch.long),
                        torch.tensor(outcome[idx], dtype=torch.float32),
                    )

            else:  # 'bn'
                if 'bn_own' not in data:
                    continue
                own_hand = data['bn_own']
                played = data['bn_played']
                bid_hist = data['bn_bid_hist']
                trick = data['bn_trick']
                void_mat = data['bn_void']
                opp_hands = data['bn_opp']

                n = len(own_hand)
                indices = np.arange(n)
                if self.shuffle_shards:
                    np.random.shuffle(indices)

                for idx in indices:
                    bh = bid_hist[idx].astype(np.float32)
                    if bh.size == 4 * 14:
                        bh_full = np.zeros((4, NUM_CONTRACTS), dtype=np.float32)
                        bh_full[:, :14] = bh.reshape(4, 14)
                        bh_t = torch.from_numpy(bh_full)
                    else:
                        bh_t = torch.from_numpy(bh).view(4, NUM_CONTRACTS)

                    yield (
                        torch.from_numpy(own_hand[idx]).float(),
                        torch.from_numpy(played[idx]).float(),
                        bh_t,
                        torch.from_numpy(trick[idx]).float(),
                        torch.from_numpy(void_mat[idx]).float().view(4, 4),
                        torch.from_numpy(opp_hands[idx]).float().view(3, 52),
                    )


def build_dataset(
    data_path: str = config.DATA_PATH,
    mode: str = 'bn',
    split: str = 'train',
    train_frac: float = config.TRAIN_VAL_SPLIT,
) -> StreamingShardDataset:
    """Build a memory-efficient StreamingShardDataset."""
    return StreamingShardDataset(
        data_path=data_path,
        mode=mode,
        split=split,
        train_frac=train_frac,
        shuffle_shards=(split == 'train'),
    )
