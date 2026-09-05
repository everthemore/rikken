# Future Idea: Stratified Counterfactual Pre-training

## 1. Initial Motivation
In card games with large bidding ladders (like Rikken with 15 contract tiers), natural self-play often suffers from contract starvation: rare high-stakes contracts (such as *Piek*, *Solo Slim*, or *Open Misère*) occur in fewer than 1% of deals.

The initial idea of **Stratified Counterfactual Pre-training** was to artificially force every contract to be played equally often across random deals (e.g. 50 games of *Open Piek*, 50 games of *Rik*, 50 games of *Misère*) to ensure that the value network $Q(s, a)$ has non-zero training signal for all actions before self-play begins.

## 2. Pathologies Encountered in Practice
When implemented directly on random deals, several unexpected failure modes emerged:
1. **The Inverted Contract Illusion**:
   - For contracts like `PIEK` and `OPEN_PIEK`, success requires winning *exactly 1 trick* (or 5 tricks).
   - When random deals were forced into `OPEN_PIEK`, players occasionally stumbled into 1 or 5 tricks ~50% of the time.
   - The value network learned that *ordinary hands with multiple Aces* have a high win chance (~55%) on `OPEN_PIEK`.
2. **Distortion of Prior Probabilities**:
   - In natural Rikken, `RIK` occurs in ~35% of deals, while `OPEN_PIEK` occurs in ~0.0% of deals.
   - Forcing equal representation destroyed the network's understanding of hand rarity, causing `OPEN_PIEK` to dominate `RIK` across early generations.

## 3. Potential Future Directions
If revisiting stratified counterfactual generation in the future, consider:
- **Strict Heuristic Pre-Filtering**: Only force a contract on deals that satisfy expert eligibility rules (e.g., only force *Piek* on hands with exactly 1 honor and minimal middle cards).
- **Importance Sampling / Likelihood Weighting**: Reweight counterfactual samples by their actual deal probability $P(\text{hand} \mid \text{contract})$ to prevent rare freak contracts from dominating the prior value landscape.
