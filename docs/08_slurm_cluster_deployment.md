# 08. Leiden ALICE HPC Cluster Deployment Guide

This guide details how to deploy, schedule, and scale the **Rikken AI Self-Play Reinforcement Learning Pipeline** on the **Leiden University ALICE HPC Cluster**.

---

## 1. ALICE HPC Architecture & Partition Mapping

The ALICE HPC cluster utilizes specific partitions based on hardware types and job durations:

| Workload | Recommended ALICE Partition | Resources Requested | Max Duration |
|:---|:---|:---|:---|
| **Parallel Self-Play (Array)** | `cpu-short,short` | 4 cores, 4 GB RAM per task | Up to 2 hours |
| **Neural Retraining (GPU)** | `gpu-short,gpu-l4-24g,gpu-2080ti-11g,gpu-a100-80g` | 1 GPU (`--gres=gpu:1`), 8 cores, 16 GB RAM | Up to 2 hours |
| **Interactive Debugging** | `gpu-i` or `short` | 1 GPU or 4 CPUs | Up to 1 hour |

---

## 2. Quick Start: 3-Step ALICE Deployment

### Step 1: Clone Repository & Run One-Time Environment Setup
Log into ALICE via SSH:
```bash
ssh <username>@login.alice.universiteitleiden.nl
cd /home/<username>/
git clone <your-repo-url> Rikken
cd Rikken

# Run automated environment setup (loads modules, creates conda env, installs PyTorch + CUDA)
bash cluster/env_setup.sh
```

### Step 2: Transfer Initial Model Checkpoints
Copy your pre-trained Phase 2 checkpoints (`bvn_best.pt` and `bn_best.pt`) from your local machine to ALICE:
```bash
# On your local laptop:
scp checkpoints/bv* checkpoints/bn* <username>@login.alice.universiteitleiden.nl:/home/<username>/Rikken/checkpoints/
```

### Step 3: Launch the Master Pipeline
To run 5 iterations of 25,000 self-play games each (125,000 total games) with a 5-generation rolling replay buffer:
```bash
bash cluster/run_pipeline.sh --iterations 5 --workers 50 --games-per-worker 500 --buffer-window 5
```

---


---

## 3. 15-Minute ALICE Smoke Test (Recommended First Step)

Before submitting a multi-thousand-game production job, run the automated **15-minute smoke test** to verify that ALICE CPU array tasks, SLURM job dependencies, and GPU allocations work seamlessly:

```bash
bash cluster/test_smoke.sh
```

### What the Smoke Test Validates:
1. **CPU Array Scheduling**: Dispatches 4 parallel tasks generating 250 games each (1,000 games total at 100 rollouts) on ALICE CPU nodes (~3 minutes).
2. **SLURM Job Chaining**: Tests `--dependency=afterok:$JOB_ID` to confirm that the GPU job sleeps until the CPU jobs complete.
3. **GPU Allocation & Retraining**: Requests 1 NVIDIA GPU (`--gres=gpu:1`), trains BVN + BN for 3 epochs, and verifies checkpoint updates (~2 minutes).
4. **Total Runtime**: **~5 to 8 minutes** end-to-end.

## 4. SLURM Scripts Overview

### `cluster/submit_self_play.slurm` (CPU Array)
- Spawns an array of 50 independent CPU tasks (`--array=0-49`).
- Each task runs MCTS self-play on 4 CPU cores using vectorised numpy states.
- Saves compressed `.npz` shards to `data/self_play/iter_N/self_play_shard_XXXX.npz`.

### `cluster/submit_retrain.slurm` (GPU Retraining)
- Allocates an NVIDIA GPU via `#SBATCH --gres=gpu:1`.
- Aggregates shards across the **Rolling Replay Buffer** (`iter_N`, `iter_N-1`, ..., `iter_N-W+1`).
- Fine-tunes BVN (Transformer) and Belief Network (ResNet).
- Automatically saves updated checkpoints to `checkpoints/bvn_best.pt` and `checkpoints/bn_best.pt`.

### `cluster/run_pipeline.sh` (Master Pipeline)
- Coordinates self-play and retraining via **SLURM Job Dependencies** (`--dependency=afterok:$SELF_PLAY_JOB`).
- Ensures retraining begins automatically the moment all 50 self-play array tasks complete.

---

## 5. Monitoring & Job Management on ALICE

```bash
# View active and queued jobs:
squeue -u $USER

# Monitor live output logs:
tail -f cluster/slurm_logs/self_play_*.out
tail -f cluster/slurm_logs/retrain_*.out

# Cancel all jobs if needed:
scancel -u $USER
```

---

## 6. Storage Best Practices on ALICE

- **Code & Checkpoints**: Keep in `/home/<username>/Rikken` (backed up, low-latency).
- **High-Volume Shards**: If scaling beyond 500,000 games, use your project data scratch directory (`/data/volume_2/<username>` or `$TMPDIR`).
