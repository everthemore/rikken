"""
config.py — Central hyperparameter and path configuration for the Rikken AI project.
"""

# ---------------------------------------------------------------------------
# Game Constants
# ---------------------------------------------------------------------------
NUM_PLAYERS: int = 4
NUM_CARDS: int = 52
NUM_SUITS: int = 4
NUM_RANKS: int = 13
NUM_TRICKS: int = 13
CARDS_ROUND1: int = 6
CARDS_ROUND2: int = 7

# ---------------------------------------------------------------------------
# Shuffle / Deal
# ---------------------------------------------------------------------------
SHUFFLE_INTENSITY: int = 3  # Riffle passes per deal (3 ≈ casual human table-shuffle).

# ---------------------------------------------------------------------------
# Heuristic Agent
# ---------------------------------------------------------------------------
TROELA_CALL_RATE: float = 0.70  # Set to 1.0 to always call Troela when eligible.

# ---------------------------------------------------------------------------
# ISMCTS
# ---------------------------------------------------------------------------
ISMCTS_ROLLOUTS: int = 800
ISMCTS_C: float = 1.41
ISMCTS_DETERMINIZATIONS: int = 20

# ---------------------------------------------------------------------------
# Neural Network — Bidding Value Network (BVN)
# ---------------------------------------------------------------------------
BVN_D_MODEL: int = 128
BVN_NHEAD: int = 4
BVN_NUM_LAYERS: int = 3
BVN_MLP_HIDDEN: int = 64

# ---------------------------------------------------------------------------
# Neural Network — Belief Network (BN)
# ---------------------------------------------------------------------------
BN_HIDDEN: int = 256
BN_NUM_BLOCKS: int = 4

# ---------------------------------------------------------------------------
# Training & Self-Play RL
# ---------------------------------------------------------------------------
LEARNING_RATE: float = 3e-4
BATCH_SIZE: int = 512
NUM_EPOCHS_BVN: int = 30
NUM_EPOCHS_BN: int = 30
DATA_SHARDS: int = 100
GAMES_PER_SHARD: int = 10_000
NUM_DATA_WORKERS: int = 8
TRAIN_VAL_SPLIT: float = 0.9
REPLAY_BUFFER_WINDOW: int = 3  # Keep data from the latest N self-play iterations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH: str = "data/"
MODEL_PATH: str = "checkpoints/"
LOG_PATH: str = "logs/"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
try:
    import torch
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE: str = "cpu"

# ---------------------------------------------------------------------------
# Random Seeds
# ---------------------------------------------------------------------------
NUMPY_SEED: int = 42
TORCH_SEED: int = 42
