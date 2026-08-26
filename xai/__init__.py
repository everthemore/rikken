"""
xai/ — Explainable AI module stubs.

Planned experiments (Phase 4):
  1. Divergence Filtering:
       Compare ISMCTS policy with Heuristic Agent on same state.
       Flag states where ISMCTS diverges radically but wins more.

  2. Unsupervised State Clustering:
       Collect vectorized state tensors from divergent plays.
       Apply t-SNE/UMAP for dimensionality reduction.
       K-Means clustering to group similar game situations.

  3. Symbolic Rule Extraction:
       Fit sklearn DecisionTree per cluster to produce human-readable IF-THEN rules.

  4. Attention Inspection:
       Use BVN's saved attention weights to visualize card/bid influence.
       Requires return_attention=True in BVN.forward().
"""

# TODO (Phase 4): Implement divergence filtering
def collect_divergent_states(ismcts_agent, heuristic_agent, game, n_games=10_000):
    """
    Play n_games and record states where ISMCTS and Heuristic disagree.

    Returns:
        List of (state_tensor, ismcts_action, heuristic_action, ismcts_win_rate)
    """
    raise NotImplementedError("XAI Phase 4: not yet implemented")


# TODO (Phase 4): Implement UMAP/t-SNE clustering
def cluster_divergent_states(state_tensors, n_clusters=10):
    """
    Cluster state tensors using UMAP + K-Means.

    Returns:
        labels (np.ndarray), embeddings_2d (np.ndarray)
    """
    raise NotImplementedError("XAI Phase 4: not yet implemented")


# TODO (Phase 4): Implement decision tree extraction
def extract_rules(state_tensors, labels, feature_names=None):
    """
    Fit a DecisionTree per cluster and return human-readable rules.

    Returns:
        List of rule strings per cluster.
    """
    raise NotImplementedError("XAI Phase 4: not yet implemented")


# TODO (Phase 4): Attention visualization
def inspect_attention(bvn, hand, bid_history):
    """
    Run BVN with attention weights and return a structured report.

    Returns:
        Dict mapping card → attention weight
    """
    raise NotImplementedError("XAI Phase 4: not yet implemented")
