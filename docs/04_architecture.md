# 04. Neural Architectures & Training Pipeline

The Rikken AI combines Neural Networks for intuition with Information Set Monte Carlo Tree Search (ISMCTS) for policy execution.

```
       +-------------------------------------------------------+
       |                  Bidding Phase                        |
       |  Input: Private Hand (52) + Bidding History (52)      |
       |  Network: Bidding Value Network (BVN - Transformer)   |
       |  Output: Expected Value (EV) per legal contract       |
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

## 1. Bidding Value Network (BVN) (`networks/bvn.py`)

- **Input Dimension**: 104 features
  - Own 13-card hand: `int8[52]` one-hot
  - Bidding history: `int8[4, 13]` flattened (52)
- **Architecture**:
  - Linear Projection: $104 	o 128$
  - Transformer Encoder: 3 layers, 4 attention heads, `d_model=128`, Pre-LN
  - MLP Head: $128 	o 64 	o 13$ (GELU activations, Dropout 0.1)
  - Softmax over legal contract mask
- **XAI Capability**: Attention weights per head/layer are preserved during evaluation to inspect which cards/bids drive aggressive bids.
- **Loss**: Binary Cross-Entropy on game win/loss outcome for the bid selected.

---

## 2. Belief Network (BN) (`networks/bn.py`)

- **Input Dimension**: 224 features
  - Own remaining hand: `52`
  - Public played cards: `52`
  - Bidding history: `52`
  - Current trick cards: `52`
  - Flattened Void Matrix: `16` ($4 	imes 4$)
- **Architecture**:
  - Input Projection: $224 	o 256$ with LayerNorm & GELU
  - Residual Trunk: 4 `ResidualBlocks` (256 hidden)
  - 3 Output Heads (one per opponent): $256 	o 128 	o 52$
  - Sigmoid $	o$ Hard Masking against Void Matrix & known cards $	o$ Normalization
- **Loss**: Average BCE loss across all 3 opponent card distributions.

---

## 3. ISMCTS Agent (`agents/ismcts.py`)

1. **Determinization**: For each of $K$ determinizations, sample hidden hands from the BN marginal probabilities while strictly respecting the Void Matrix.
2. **Tree Search**: Execute MCTS rollouts using the ISMCTS-UCB1 formula:
   $$	ext{UCB1}(a) = rac{Q(a)}{N(a)} + c \sqrt{rac{\ln(	ext{Availability}(a))}{N(a)}}$$
3. **Action Selection**: Select the card with the highest cumulative visit count across all sampled worlds.

---

## 4. Training Pipeline (Phases 1, 2 & 3)

1. **Phase 1 (Bootstrapping)**: Generate 1,000,000 games using heuristic self-play, stored as compressed `.npz` shards.
2. **Phase 2 (Supervised Learning)**:
   - Train BVN on bid-outcome pairs ($104 	o 13$).
   - Train BN on step-by-step opponent card distribution targets ($224 	o 3 	imes 52$).
3. **Phase 3 (Self-Play RL)**:
   - 4 ISMCTS agents driven by the trained BVN/BN play iterative tournaments.
   - Retrain networks on self-play game logs to reach superhuman performance.
