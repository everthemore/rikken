# 04. Neural Architectures & Training Pipeline

The Rikken AI combines Neural Networks for intuition with Information Set Monte Carlo Tree Search (ISMCTS) for policy execution in an end-to-end AlphaZero-inspired architecture.

```
       +-------------------------------------------------------+
       |                  Bidding Phase                        |
       |  Input: Private Hand (52) + Bidding History (56)      |
       |  Network: Action-Value Q-Network (BVN - Transformer)  |
       |  Output: Expected Net Score Q(s, a) in [-1.0, +1.0]   |
       |  Policy: Pure Argmax Q(s, a) over legal contract mask |
       +-------------------------------------------------------+
                                  |
                                  v
       +-------------------------------------------------------+
       |               Trick-Taking Phase                      |
       |  Public State + Own Hand + Void Matrix                |
       |                          |                            |
       |                          v                            |
       |  Belief Network (BN - ResNet + Hard Void Masking)     |
       |  Output: P(opponent p holds card c)                   |
       |                          |                            |
       |                          v                            |
       |  ISMCTS Policy (Belief-Guided Determinization)        |
       |  Output: Argmax visit-count move                      |
       +-------------------------------------------------------+
```

---

## 1. Bidding Action-Value Q-Network (BVN) (`networks/bvn.py`)

- **Input Dimension**: 108 features
  - Own 13-card hand: `int8[52]` one-hot
  - Bidding history: `float32[4, 14]` one-hot flattened (56 features for all 14 contracts: `PAS` through `TROELA`)
- **Architecture**:
  - Linear Projection: $108 	o 128$
  - Transformer Encoder: 3 layers, 4 attention heads, `d_model=128`, Pre-LN, GELU activations
  - Value Head: $128 	o 64 	o 14$ with `nn.Tanh()` output activation
  - Outputs continuous Expected Returns $Q_	heta(s, a) \in [-1.0, +1.0]$ across all 14 legal actions.
- **Constant-Free Policy Execution**:
  $$\pi(s) = rg\max_{a \in 	ext{Legal}(s)} Q_	heta(s, a)$$
  Zero manual thresholds or heuristic parameters. The agent dynamically compares the expected value of declaring against the expected value of defending after passing.
- **Loss Function**: Smooth L1 (Huber) regression ($eta = 0.1$) on signed terminal match outcomes $R \in \{-1.0, +1.0\}$:
  $$\mathcal{L}(	heta) = rac{1}{B} \sum_{i=1}^B 	ext{Huber}\left( Q_	heta(s_i, a_i) - R_i ight)$$

---

## 2. Belief Network (BN) (`networks/bn.py`)

- **Input Dimension**: 228 features
  - Own remaining hand: `52`
  - Publicly played cards: `52`
  - Bidding history: `56` ($4 	imes 14$)
  - Current trick cards: `52` (one-hot)
  - Flattened Void Matrix: `16` ($4 	imes 4$)
- **Architecture**:
  - Input Projection: $228 	o 256$ with LayerNorm & GELU
  - Residual Trunk: 4 `ResidualBlocks` (256 hidden dimensions)
  - 3 Output Heads (one per opponent seat): $256 	o 128 	o 52$
  - Sigmoid $	o$ Hard Masking against Void Matrix & known cards $	o$ Normalization
- **Loss**: Average Binary Cross-Entropy loss across all 3 opponent marginal distributions.

---

## 3. ISMCTS Agent (`agents/ismcts.py`)

1. **Informed Determinization**: For each of $K$ determinizations, sample hidden hands from the Belief Network's marginal probabilities while strictly respecting known cards and the logical Void Matrix.
2. **Tree Search**: Execute MCTS rollouts using the ISMCTS-UCB1 formula:
   $$	ext{UCB1}(a) = rac{Q(a)}{N(a)} + c \sqrt{rac{\ln(	ext{Availability}(a))}{N(a)}}$$
3. **Action Selection**: Select the card with the highest cumulative visit count across all sampled worlds.

---

## 4. Multi-Generation Reinforcement Learning Pipeline

1. **Data Generation & Self-Play**:
   - $N$ parallel workers simulate matches between neural agents.
   - Bidding states store the signed terminal return $R \in \{-1.0, +1.0\}$ from each player's perspective.
   - Trick-taking transitions store ground-truth hidden opponent hands for the Belief Network.
2. **GPU Retraining (ALICE HPC)**:
   - Supervised fine-tuning of BVN via SmoothL1 loss on the rolling multi-generation replay buffer.
   - Supervised fine-tuning of BN via multi-target BCE loss.
3. **Benchmark Tournament**:
   - Rotates seating across hundreds of matches against standard baselines to track convergence and win rates.
