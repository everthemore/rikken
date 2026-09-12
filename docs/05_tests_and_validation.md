# 05. Test Suite, Benchmarks & Validation Data

The Rikken AI system is validated by an automated test suite of **58 unit tests** across engine mechanics, bidding calibration, Belief Network determinization, and webapp API endpoints.

---

## 1. Test Suite Summary (`tests/`)

All 58 unit tests execute in **~2.6 seconds**:

```
============================== 58 passed in 2.65s ==============================
```

### Breakdown of Test Modules:

| Test Module / Class | Tests | What is Validated |
|:---|:---:|:---|
| **`test_engine.py`** | **45** | Card helpers, 52-card round-robin dealing, clumping shuffle, legal bidding & trick priority, void matrix inference, multi-tier early stopping, 1,000 full game stress tests |
| **`test_bidding_calibration.py`** | **9** | Marginal Open Piek/Misère rejection, solid Rik acceptance, negative-EV contract rejection, heuristic rollouts, **Belief Network-guided ISMCTS determinization**, **BVNDualLoss Counterfactual Grounding & CQL** |
| **`test_webapp.py`** | **4** | Flask API endpoints, AI advice engine, live Belief Network probability heatmaps, legal move generation |

---

## 2. Benchmark Empirical Validation (5,000 Games)

Execution of `python main.py benchmark --games 5000` demonstrates engine stability and diverse contract distribution:

| Contract | Games Played | Frequency | Declarer Win Rate |
|:---|:---:|:---:|:---:|
| `NEGEN_ALLEEN` | 1,393 | 27.9% | 14.2% |
| `TWAALF_ALLEEN` | 1,327 | 26.5% | 8.1% |
| `ELF_ALLEEN` | 661 | 13.2% | 11.5% |
| `TIEN_ALLEEN` | 582 | 11.6% | 13.7% |
| `TROELA` | 570 | 11.4% | 19.8% |
| `PIEK` | 181 | 3.6% | 6.6% |
| `MISERE` | 163 | 3.3% | 7.4% |
| `SOLO_SLIM` | 102 | 2.0% | 2.9% |
| `ACHT_ALLEEN` | 19 | 0.4% | 21.1% |
| `RIK` | 2 | 0.04% | 50.0% |
| **Total / Average** | **5,000** | **100%** | **12.2%** |

### Early Stopping Efficiency
- **Mean tricks played**: **7.26 / 13 tricks**
- **Early termination rate**: **96.5%** of games terminate early as soon as the outcome is mathematically decided.
- **Redeals (4x Pas)**: 0 in 5,000 games under default heuristic settings.
