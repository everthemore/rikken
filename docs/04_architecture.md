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

## 1. Dual-Head Bidding Value Network (BVN) (`networks/bvn.py`)

- **Input Dimension**: 112 features
  - Own 13-card hand: `int8[52]` one-hot
  - Bidding history: `float32[4, 15]` one-hot flattened (60 features for all 15 contracts: `PAS` through `MOELA`)
- **Architecture**:
  - Linear Projection: $112 \to 128$
  - Transformer Encoder: 3 layers, 4 attention heads, `d_model=128`, Pre-LN, GELU activations
  - **Dual-Head Output**:
    1. **Win-Probability Head**: $128 \to 64 \to 15$ with `nn.Sigmoid()` output activation:
       $$P_\theta(\text{Win} \mid s, a) \in [0.0, 1.0]$$
       Represents the probability of winning the contract given the player's hand and current auction state.
    2. **Expected-Value (Points) Head**: $128 \to 64 \to 15$ with `nn.Tanh()` output activation:
       $$\mathbb{E}_\theta[\text{Points} \mid s, a] \in [-1.0, +1.0]$$
       Represents the normalized expected game score used for UI analytics and strategic point evaluation.
- **Why Dual-Head Architecture Was Essential**:
  Under pure Expected Value maximization, because solo contracts (like Piek/Misère) have $\pm 6$ to $\pm 9$ point stakes compared to $\pm 1$ for RIK, an agent with only a 55% win chance on OPEN PIEK achieves $+0.90$ expected points, dominating a safe 80% lock on RIK ($+0.60$ expected points). The agent was mathematically incentivized to gamble on big contracts.
- **Argmax Win Probability Policy Execution**:
  $$\pi(s) = \arg\max_{a \in \text{Legal}(s)} P_\theta(\text{Win} \mid s, a)$$
  The agent bids the contract with the **highest probability of winning**, naturally favoring ~75–85% RIK bids over ~50–55% PIEK gambles on ordinary hands!
- **Joint Loss Function**:
  $$\mathcal{L}(\theta) = \mathcal{L}_{\text{BCE}}(P_\theta(s, a), y_{\text{won}}) + \mathcal{L}_{\text{Huber}}(\mathbb{E}_\theta(s, a), R)$$
  where $y_{\text{won}} \in \{0.0, 1.0\}$ and $R \in [-1.0, +1.0]$ is the normalized zero-sum match payoff.

<details>
<summary><b>🔍 View PyTorch Implementation: Dual-Head BVN Architecture & Forward Pass (networks/bvn.py)</b></summary>

```python
class BVN(nn.Module):
    def __init__(self, num_contracts=15, d_model=128, nhead=4, num_layers=3, mlp_hidden=64):
        super().__init__()
        self.input_size = 52 + 4 * num_contracts  # 112 features
        self.input_proj = nn.Linear(self.input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Head 1: P(Win) in [0.0, 1.0] (Sigmoid)
        self.win_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(mlp_hidden, num_contracts), nn.Sigmoid()
        )

        # Head 2: Expected Points in [-1.0, +1.0] (Tanh)
        self.ev_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(mlp_hidden, num_contracts), nn.Tanh()
        )

    def forward(self, hands: torch.Tensor, bid_hist: torch.Tensor):
        x = torch.cat([hands, bid_hist], dim=-1)
        h = self.input_proj(x).unsqueeze(1)
        h = self.transformer(h).squeeze(1)
        return self.win_head(h), self.ev_head(h)
```
</details>

<details>
<summary><b>🔍 View Policy Implementation: Argmax Win Probability Bidding Action (agents/neural_agent.py)</b></summary>

```python
def _act_bid(self, state: RikkenState) -> int:
    legal = legal_bids(state)
    if not legal or legal == [int(Contract.PAS)]:
        return int(Contract.PAS)

    # Predict win probabilities across all contracts
    win_probs, ev_scores = self.bvn.predict(hand=state.hands[self.seat], bids=state.bids, device=self.device)

    # Mask illegal bids with -inf
    masked_win = np.full(len(win_probs), -np.inf)
    for b in legal:
        masked_win[b] = win_probs[b]

    # Select legal contract with highest Win Probability:
    best_bid = int(np.argmax(masked_win))
    return best_bid
```
</details>

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
