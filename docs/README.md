# Rikken AI — Documentation Index

This folder contains the complete technical and game-design documentation for the
**Rikken AI** project: a state-of-the-art hybrid neural network + ISMCTS agent
for the Dutch trick-taking card game *Rikken*, built for academic publication.

---

## Documents

| File | Contents |
|---|---|
| [01_game_rules.md](01_game_rules.md) | Complete game rules, bidding hierarchy, trick-taking mechanics, custom house rules |
| [02_human_heuristics.md](02_human_heuristics.md) | Documented human strategies, bidding guidelines, play tactics per contract type |
| [03_implementation.md](03_implementation.md) | Full technical implementation: engine modules, data structures, algorithms |
| [04_architecture.md](04_architecture.md) | AI architecture: BVN, Belief Network, ISMCTS, training pipeline |
| [05_tests_and_validation.md](05_tests_and_validation.md) | Test suite design, all 36 test cases, benchmark results, validation data |
| [06_resource_and_checkpoint_guide.md](06_resource_and_checkpoint_guide.md) | Hardware profiles, timing benchmarks, step-by-step checkpointing instructions |
| [07_dataset_analysis.md](07_dataset_analysis.md) | Analysis of 1M-game dataset: heuristic player strategies, win-rates, and bidding distributions |
| [08_slurm_cluster_deployment.md](08_slurm_cluster_deployment.md) | HPC cluster deployment, SLURM array jobs for self-play, GPU retraining, automated pipeline |
| [09_self_play_reinforcement_learning.md](09_self_play_reinforcement_learning.md) | 47-generation empirical trajectory, 5-window replay buffer analysis, variance damping |
| [10_xai_and_rule_extraction.md](10_xai_and_rule_extraction.md) | Extracted bidding thresholds, decision tree rules, trick-taking tactics, and decision matrix |

---

## Quick-start

```bash
# Run a single game with verbose output
python main.py play --verbose --seed 42

# Run a 5000-game benchmark
python main.py benchmark --games 5000

# Generate the Phase 1 training dataset (1M games)
python main.py generate --games 1000000 --workers 8

# Run all 36 unit tests
python -m pytest tests/ -v
```

---

## Project structure

```
rikken/
├── docs/                   ← You are here
├── engine/
│   ├── card.py             Card constants, encoding, trick resolution
│   ├── deck.py             Clumping shuffle, round-robin deal
│   ├── state.py            RikkenState dataclass, Contract enum
│   ├── rules.py            Legal actions, void matrix, Aasvragen
│   ├── early_stop.py       Basic + advanced early stopping
│   └── game.py             RikkenGame orchestrator
├── agents/
│   ├── heuristic.py        Rule-based heuristic agent (Phase 1)
│   ├── ismcts.py           ISMCTS agent skeleton (Phase 3)
│   └── random_agent.py     Uniform random baseline
├── networks/
│   ├── bvn.py              Bidding Value Network (Transformer)
│   └── bn.py               Belief Network (ResNet)
├── training/
│   ├── data_gen.py         Multiprocess heuristic self-play data generation
│   ├── dataset.py          PyTorch Dataset wrappers for sharded .npz files
│   ├── train_bvn.py        BVN supervised training loop
│   └── train_bn.py         BN supervised training loop
├── xai/                    Explainable AI stubs (Phase 4)
├── tests/
│   └── test_engine.py      36 unit tests (all passing)
├── config.py               All hyperparameters in one place
└── main.py                 CLI entry point
```

---

## Technology stack

| Layer | Technology | Reason |
|---|---|---|
| Engine / data | Python + NumPy (`int8`) | Vectorised ops, JAX-compatible tensors |
| Neural networks | PyTorch | Flexible research framework, GPU support |
| Search | Python ISMCTS | Pure Python for clarity; hotpath is the engine |
| Data format | `.npz` shards (compressed) | Memory-mapped, random-access, portable |
| Testing | `pytest` | Standard, composable, good fixtures |

---

*Generated: 2026-08-24. See individual files for detailed information.*
