# 06. Resource Estimation, Timing & Checkpointing Guide

Empirical performance measurements and resource requirements based on testing on **Apple Silicon (8 CPU cores, 8.6 GB RAM, macOS)**.

---

## 1. Resource & Timing Estimates Summary

| Step | Command | Throughput | Estimated Time | RAM Usage | Disk Space |
|:---|:---|:---:|:---:|:---:|:---:|
| **1. Data Gen** (1M games) | `python main.py generate --games 1000000 --workers 4` | ~1,050 games/s | **~16 min** | ~1.2 GB | ~470 MB (100 shards) |
| **2. Train BVN** (30 epochs) | `python main.py train-bvn --epochs 30` | ~29 steps/s (MPS) | **~34 min** | ~1.5 GB | ~25 MB (checkpoints) |
| **3. Train BN** (30 epochs) | `python main.py train-bn --epochs 30` | ~25 steps/s (MPS) | **~39 min** | ~1.8 GB | ~35 MB (checkpoints) |
| **4. Benchmark** (5k games) | `python main.py benchmark --games 5000` | ~600 games/s | **~8 sec** | ~150 MB | Negligible |

---

## 2. Checkpointing & Resumability Guide

All components are designed with atomic, fault-tolerant checkpointing so you can stop and resume anytime.

### 2.1 Dataset Generation Resumability
The data generator writes each 10,000-game shard atomically (`shard_XXXX.npz.tmp` $	o$ `shard_XXXX.npz`).
- **Auto-Resume**: If data generation is interrupted at shard 45, running the command again will automatically detect the 45 existing shards and resume from shard 46.
```bash
# Resume generation seamlessly:
python main.py generate --games 1000000 --workers 4

# To force complete regeneration from scratch:
python main.py generate --games 1000000 --workers 4 --no-resume
```

### 2.2 Neural Network Training Checkpointing
Both `train-bvn` and `train-bn` save complete training state dictionaries:
- `checkpoints/bvn_checkpoint.pt` / `bn_checkpoint.pt`: Saved every epoch (contains `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_val_loss`, `history`).
- `checkpoints/bvn_best.pt` / `bn_best.pt`: Saved whenever validation loss improves.
- `checkpoints/bvn_final.pt` / `bn_final.pt`: Saved upon epoch completion.

#### How to Resume Training:
```bash
# Resume BVN training from the latest checkpoint:
python main.py train-bvn --resume-latest

# Or resume from a specific checkpoint file:
python main.py train-bvn --resume checkpoints/bvn_checkpoint.pt

# Resume BN training from the latest checkpoint:
python main.py train-bn --resume-latest
```

---

## 3. Recommended Execution Workflow

```bash
# Step 1: Generate Phase 1 Dataset (1M games, ~16 min)
python main.py generate --games 1000000 --workers 4

# Step 2: Train Bidding Value Network (~34 min)
python main.py train-bvn --epochs 30

# Step 3: Train Belief Network (~39 min)
python main.py train-bn --epochs 30

# Step 4: Run ISMCTS / Heuristic Benchmark
python main.py benchmark --games 5000
```
