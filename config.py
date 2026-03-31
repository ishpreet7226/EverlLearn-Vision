"""
EverLearn Vision - Dataset Configuration
Centralise all dataset and training settings here.
"""

# ── Dataset ──────────────────────────────────────────────────────────────────
DATA_DIR = "data"          # Root data folder
IMAGE_SIZE = (224, 224)    # Resize all images to this (H, W)
NUM_WORKERS = 4            # DataLoader workers (set 0 on Windows if issues)

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = "cuda"            # "cuda" | "mps" | "cpu"

# ── Checkpoints ───────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_NAME = "best_model.pth"
