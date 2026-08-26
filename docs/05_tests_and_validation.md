# 05. Test Suite, Benchmarks & Validation Data

The Rikken engine is validated by an automated test suite of 36 unit tests covering all rules, edge cases, and 1,000 full game simulations.

---

## 1. Test Suite Summary (`tests/test_engine.py`)

All 36 unit tests execute in **~2.5 seconds**:

```
============================== 36 passed in 2.56s ==============================
```

### Breakdown of Test Modules:

| Test Class | Tests | What is Validated |
|:---|:---:|:---|
| `TestCardHelpers` | 7 | Card ID $\leftrightarrow$ string roundtrips, rank/suit lookups, `beats()` logic, trick winners |
| `TestDeckAndDeal` | 5 | 52-card distribution, no duplicates, round-robin 6-then-7 order, clumping riffle shuffle |
| `TestBiddingRules` | 6 | Pass rules, non-rebidding, ascending hierarchy, Troela 3-Ace requirement, 4-pass redeal |
| `TestTrickTakingRules` | 4 | Follow-suit obligation, must-trump logic, free discards on dual void, trick-lead freedom |
| `TestVoidMatrix` | 4 | Void inference on fail-to-follow, dual void inference on discard, clean tracking |
| `TestEarlyStopping` | 8 | Misère 1-trick loss, Alleen win/loss thresholds, Solo Slim 1-trick loss, Piek dead zones & wins |
| `TestFullGame` | 2 | End-to-end game execution, 1,000 continuous games without exceptions or illegal moves |

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
