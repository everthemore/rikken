"""
training/data_gen.py — Phase 1 dataset generation using the Heuristic Agent.

Performance notes (measured on Apple M-series, 8 logical cores, 8.6 GB RAM):
  - Single-thread throughput:  ~600 games/s
  - 4-worker spawn pool:       ~1050 games/s (1.77x speedup with 2000-game batches)
  - Recommended invocation:    --workers 4 --games-per-batch 2000
  - 1M games expected time:    ~16 min (4 workers, 2000-game batches)
  - Disk usage:                ~470 MB for 1M games (100 shards × 10k games)

Resumability:
  - Shards that already exist on disk are SKIPPED automatically.
  - Run the same command again after an interruption to fill missing shards.
  - Each shard is written atomically (temp file → rename) so partial writes
    never corrupt the dataset.

Usage:
    python -m training.data_gen --games 1000000 --workers 4
    python -m training.data_gen --games 1000000 --workers 1   # safer on some systems
"""

from __future__ import annotations
import os
import sys
import numpy as np
import multiprocessing as mp
from typing import Optional, Tuple, List
import argparse
import time
import logging
import tempfile

from engine.state import RikkenState, Contract, Phase
from engine.game import RikkenGame
from agents.heuristic import HeuristicAgent
import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single game runner
# ---------------------------------------------------------------------------

def run_one_game(seed: int) -> dict:
    """
    Run one complete game with 4 heuristic agents and return the collected records.

    Returns:
        dict with keys 'bid_records' and 'play_records'.
        Returns empty lists for redeals (all-pass games).
    """
    rng = np.random.default_rng(seed)
    game = RikkenGame(rng=rng)
    agents = [HeuristicAgent(seat=p, rng=rng) for p in range(4)]

    state = game.reset()
    bid_records = []
    play_records = []
    prev_bid_history = np.zeros((4, 14), dtype=np.int8)

    # ---- Bidding phase ----
    while state.phase == Phase.BIDDING:
        p = state.current_player
        agent = agents[p]
        action = agent.act(state)

        bid_records.append({
            'player':       p,
            'hand':         state.hands[p].copy(),
            'bid_history':  prev_bid_history.copy(),
            'bid_taken':    action,
        })

        if action != int(Contract.PAS):
            prev_bid_history[p, action] = 1

        state, reward = game.step(state, action)
        if reward is not None and state.phase == Phase.TERMINAL:
            return {'bid_records': [], 'play_records': []}

    # ---- Declaration ----
    if state.phase == Phase.TRICK_TAKING and state.trick_count == 0:
        declarer_agent = agents[state.declarer]
        if Contract.is_trump_contract(state.contract):
            trump = declarer_agent.declare_trump(state)
            vraagaas = -1
            if state.contract in (Contract.RIK, Contract.RIK_BETER):
                vraagaas = declarer_agent.declare_vraagaas(state, trump)
            state = game.declare(state, trump_suit=trump, vraagaas_suit=vraagaas)

    # ---- Trick-taking phase ----
    while state.phase == Phase.TRICK_TAKING:
        p = state.current_player
        opponents = [q for q in range(4) if q != p]
        opp_hands = np.stack([state.hands[q] for q in opponents], axis=0)

        trick_onehot = np.zeros(52, dtype=np.int8)
        for card in state.current_trick:
            if card >= 0:
                trick_onehot[card] = 1

        play_records.append({
            'player':       p,
            'own_hand':     state.hands[p].copy(),
            'played':       state.played_cards.copy(),
            'bid_history':  prev_bid_history.copy(),
            'trick':        trick_onehot.copy(),
            'void_matrix':  state.void_matrix.copy(),
            'opp_hands':    opp_hands.copy(),
            'trick_count':  state.trick_count,
        })

        action = agents[p].act(state)
        state, reward = game.step(state, action)

    for rec in bid_records:
        p = rec['player']
        rec['outcome'] = float(game.get_reward(state, p) > 0)

    return {
        'bid_records':  bid_records,
        'play_records': play_records,
        'contract':     int(state.contract) if state.contract else -1,
        'declarer':     state.declarer,
    }


# ---------------------------------------------------------------------------
# Shard writer (atomic write via temp file)
# ---------------------------------------------------------------------------

def _pack_shard(records: List[dict], shard_path: str) -> None:
    """
    Pack records into a compressed .npz shard.

    Written atomically: data goes to a temp file first, then is renamed
    to the final path. This guarantees no partial/corrupt shards on disk.
    """
    bvn_hands, bvn_bid_hist, bvn_bid_taken, bvn_outcome = [], [], [], []
    bn_own, bn_played, bn_bid_hist, bn_trick, bn_void, bn_opp = [], [], [], [], [], []

    for rec in records:
        if rec.get('type') == 'bid':
            bvn_hands.append(rec['hand'])
            bvn_bid_hist.append(rec['bid_history'].flatten())
            bvn_bid_taken.append(rec['bid_taken'])
            bvn_outcome.append(rec['outcome'])
        elif rec.get('type') == 'play':
            bn_own.append(rec['own_hand'])
            bn_played.append(rec['played'])
            bn_bid_hist.append(rec['bid_history'].flatten())
            bn_trick.append(rec['trick'])
            bn_void.append(rec['void_matrix'].flatten())
            bn_opp.append(rec['opp_hands'].flatten())

    save_dict = {}
    if bvn_hands:
        save_dict.update({
            'bvn_hands':     np.array(bvn_hands,     dtype=np.int8),
            'bvn_bid_hist':  np.array(bvn_bid_hist,  dtype=np.int8),
            'bvn_bid_taken': np.array(bvn_bid_taken, dtype=np.int8),
            'bvn_outcome':   np.array(bvn_outcome,   dtype=np.float32),
        })
    if bn_own:
        save_dict.update({
            'bn_own':        np.array(bn_own,      dtype=np.int8),
            'bn_played':     np.array(bn_played,   dtype=np.int8),
            'bn_bid_hist':   np.array(bn_bid_hist, dtype=np.int8),
            'bn_trick':      np.array(bn_trick,    dtype=np.int8),
            'bn_void':       np.array(bn_void,     dtype=np.int8),
            'bn_opp':        np.array(bn_opp,      dtype=np.int8),
        })

    # Atomic write: temp file → rename
    dir_ = os.path.dirname(shard_path) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix='.tmp.npz')
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **save_dict)
        # np.savez_compressed will not append .npz since it already ends with .npz
        os.replace(tmp_path, shard_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _worker(args: Tuple[int, int, int]) -> List[dict]:
    """Worker: run `n_games` games starting at `seed_offset` and return flat records."""
    n_games, seed_offset, shard_idx = args
    flat_records = []

    for i in range(n_games):
        result = run_one_game(seed=seed_offset + i)
        for rec in result['bid_records']:
            rec['type'] = 'bid'
            flat_records.append(rec)
        for rec in result['play_records']:
            rec['type'] = 'play'
            flat_records.append(rec)

    return flat_records


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dataset(
    n_games: int = config.GAMES_PER_SHARD * config.DATA_SHARDS,
    n_workers: int = config.NUM_DATA_WORKERS,
    data_path: str = config.DATA_PATH,
    games_per_shard: int = config.GAMES_PER_SHARD,
    games_per_batch: int = 2000,
    resume: bool = True,
) -> None:
    """
    Generate the full Phase 1 dataset with automatic resume support.

    Args:
        n_games:        Total number of games to simulate.
        n_workers:      Number of parallel worker processes (use 1 on some macOS
                        setups; 4 is optimal for 8-core Apple Silicon with
                        games_per_batch=2000).
        data_path:      Directory to write shards into.
        games_per_shard: Games per .npz file (default 10,000).
        games_per_batch: Games sent to each worker per task. Larger = better
                        efficiency but more memory per worker (2000 recommended).
        resume:         If True, skip shards that already exist on disk.
                        Set to False to regenerate everything from scratch.
    """
    os.makedirs(data_path, exist_ok=True)
    n_shards = (n_games + games_per_shard - 1) // games_per_shard

    # --- Identify which shards need to be generated ---
    pending_shards = []
    skipped = 0
    for shard in range(n_shards):
        shard_path = os.path.join(data_path, f"shard_{shard:04d}.npz")
        if resume and os.path.exists(shard_path):
            skipped += 1
        else:
            pending_shards.append(shard)

    if skipped:
        print(f"Resuming: {skipped}/{n_shards} shards already exist — skipping them.")
    if not pending_shards:
        print("All shards already generated. Nothing to do.")
        return

    total_pending_games = len(pending_shards) * games_per_shard
    print(f"Generating {len(pending_shards)} shards "
          f"({total_pending_games:,} games) → '{data_path}'")
    print(f"Workers: {n_workers}, games/batch: {games_per_batch}, "
          f"games/shard: {games_per_shard}")
    if n_workers > 1:
        print(f"Note: using spawn multiprocessing context (macOS-safe)")

    # --- Build task list: one task per worker batch within each shard ---
    # Each shard is split into ceil(games_per_shard/games_per_batch) worker tasks.
    # Results are accumulated in-process, then the whole shard is written at once.
    start = time.time()
    games_done = skipped * games_per_shard
    total_games = n_shards * games_per_shard

    ctx = mp.get_context('spawn')

    for shard_num, shard in enumerate(pending_shards):
        seed_base = shard * games_per_shard
        shard_path = os.path.join(data_path, f"shard_{shard:04d}.npz")

        # Split shard into worker batches
        batches = []
        for offset in range(0, games_per_shard, games_per_batch):
            batch_size = min(games_per_batch, games_per_shard - offset)
            batches.append((batch_size, seed_base + offset, shard))

        all_records: List[dict] = []

        if n_workers <= 1:
            for task in batches:
                all_records.extend(_worker(task))
        else:
            with ctx.Pool(processes=n_workers) as pool:
                for result in pool.imap_unordered(_worker, batches):
                    all_records.extend(result)

        _pack_shard(all_records, shard_path)
        games_done += games_per_shard

        elapsed = time.time() - start
        rate = (shard_num + 1) * games_per_shard / elapsed if elapsed > 0 else 0
        eta   = (total_pending_games - (shard_num + 1) * games_per_shard) / rate if rate > 0 else 0
        print(
            f"  Shard {shard+1:4d}/{n_shards} "
            f"({shard_num+1}/{len(pending_shards)} pending) | "
            f"{games_done:,}/{total_games:,} games | "
            f"{rate:.0f} g/s | "
            f"ETA {eta/60:.1f} min"
        )

    total_time = time.time() - start
    if total_time > 0:
        print(f"\nDone in {total_time:.1f}s "
              f"({total_pending_games / total_time:.0f} games/s avg)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Rikken training dataset')
    parser.add_argument('--games',          type=int,  default=config.GAMES_PER_SHARD * config.DATA_SHARDS)
    parser.add_argument('--workers',        type=int,  default=4,
                        help='Parallel workers (4 recommended on 8-core Apple Silicon)')
    parser.add_argument('--path',           type=str,  default=config.DATA_PATH)
    parser.add_argument('--games-per-shard',type=int,  default=config.GAMES_PER_SHARD)
    parser.add_argument('--games-per-batch',type=int,  default=2000,
                        help='Games per worker task (larger = more efficient, default 2000)')
    parser.add_argument('--no-resume',      action='store_true',
                        help='Regenerate all shards even if they already exist')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    generate_dataset(
        n_games=args.games,
        n_workers=args.workers,
        data_path=args.path,
        games_per_shard=args.games_per_shard,
        games_per_batch=args.games_per_batch,
        resume=not args.no_resume,
    )
