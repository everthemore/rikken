# 09. Self-Play Reinforcement Learning & Convergence Methodology

This document outlines the self-play reinforcement learning pipeline, the mathematical convergence framework, and the rolling replay buffer architecture for the constant-free Action-Value Q-Network ($Q \in [-1.0, +1.0]$) and Belief-Guided ISMCTS.

---

## 1. Core Reinforcement Learning Architecture

The current system operates under an **AlphaZero-style Policy Iteration Loop** adapted for imperfect-information card games:

```
  ┌─────────────────────────────────────────────────────────────┐
  │                 PARALLEL SELF-PLAY WORKERS                  │
  │  - 100 SLURM CPU Workers simulating matches                 │
  │  - Neural agents play with current BVN Q-Network + ISMCTS   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Generates new shard data
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │             ROLLING SLIDING REPLAY BUFFER                   │
  │  - Retains latest W generations (e.g. W = 3 to 5)          │
  │  - Stores state s, action a, and signed terminal reward R   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Batched PyTorch Streaming
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 GPU RETRAINING ON ALICE                     │
  │  - BVN Q-Network: SmoothL1 (Huber) regression to R in [-1,1]│
  │  - Belief Network: BCE multi-head loss on hidden cards      │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Deploys updated weights
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 EVALUATION TOURNAMENT                       │
  │  - Head-to-Head benchmark vs Baseline across 300+ deals     │
  │  - Evaluates Offense, Defense, and Overall Win Rates        │
  └─────────────────────────────────────────────────────────────┘
```

---

## 2. Dynamic Pass Valuation: Overcoming Fixed Thresholds

### 2.1 The Constant-Free Policy
Previous iterations used independent binary classification heads with an artificial safety threshold (`P(win) >= 0.50`). The current architecture replaces this with a **Pure Action-Value Network**:

$$\pi(s) = rg\max_{a \in 	ext{Legal}(s)} Q_	heta(s, a)$$

- **`PAS` has an actively learned value**:
  - Against an overbidding opponent (e.g. suicidal solo bid): $Q(s, 	ext{PAS}) pprox \mathbf{+0.85}$ (high defensive payoff).
  - On a neutral deal: $Q(s, 	ext{PAS}) pprox \mathbf{0.0}$.
- **Decision Rule**: The agent bids if and only if declaring has a strictly higher expected payoff than defending after passing:
  $$	ext{Bid } a \iff Q(s, a) > Q(s, 	ext{PAS})$$

---

## 3. Mathematical Stability & Non-Divergence

Unlike standard Deep Q-Networks (DQN) which suffer from the "Deadly Triad" (temporal difference bootstrapping on non-stationary targets), our BVN is trained via **Monte Carlo Policy Evaluation on True Terminal Rewards**:

$$\mathcal{L}(	heta) = rac{1}{B} \sum_{i=1}^B 	ext{Huber}\left( Q_	heta(s_i, a_i) - R_i ight) \quad 	ext{where } R_i \in \{-1.0, +1.0\}$$

Because the target $R_i$ is a fixed, ground-truth game result (not a moving bootstrap estimate $\gamma \max Q$), training is **mathematically convex and strictly immune to Q-learning divergence**.

---

## 4. Sliding Replay Buffer Architecture

To prevent catastrophic forgetting of rare contract modalities (*Solo Slim*, *Troela*, *Open Misère*) and stabilize inter-generational learning, the pipeline integrates a **Sliding Replay Buffer** across iterations:

$$\mathcal{D}_{	ext{train}}^{(N)} = igcup_{k=\max(1, N-W+1)}^{N} \mathcal{D}_{	ext{self-play}}^{(k)}$$

- **Default Window Size**: $W = 3$ to $5$ generations.
- **Continuous Retention**: Ensures the neural networks generalize across multiple past policy generations rather than overfitting to the latest generation's specific meta.

---

## 5. Active Cluster Runs (ALICE HPC)

*A clean multi-generation training run on the fresh signed Q-value architecture is currently active on the ALICE HPC cluster.*

Empirical convergence logs, multi-generation tournament tables, and loss trajectories will be compiled and documented here upon completion of the cluster pipeline.
