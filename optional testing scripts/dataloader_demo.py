"""
EverLearn Vision – DataLoader Demo
====================================
Run this script to test your entire data pipeline end-to-end.
It prints class names, dataset sizes, and the shape of one batch.

Usage:
    python dataloader_demo.py
"""

import sys
from pathlib import Path
import torch

# Make sure we can import from src/
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import get_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load data using ImageFolder-powered DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  EverLearn Vision — DataLoader Demo")
print("=" * 55)

train_loader, val_loader, class_names = get_dataloaders(
    data_dir="data",        # looks for data/train/ and data/val/
    image_size=(224, 224),  # every image resized to 224×224
    batch_size=32,          # 32 images per batch
    num_workers=0,          # set to 0 for demo (avoids multiprocessing issues)
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Print class info
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n📌  Classes detected  : {class_names}")
print(f"    Number of classes : {len(class_names)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Dataset sizes
# ─────────────────────────────────────────────────────────────────────────────
print(f"📊  Train dataset size : {len(train_loader.dataset)} images")
print(f"    Val   dataset size : {len(val_loader.dataset)} images")
print(f"    Train batches      : {len(train_loader)}")
print(f"    Val   batches      : {len(val_loader)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Grab one batch and inspect its shape
# ─────────────────────────────────────────────────────────────────────────────
images, labels = next(iter(train_loader))

# images shape: (batch_size, channels, height, width)
#   batch_size = 32     → number of images in this batch
#   channels   = 3      → RGB (Red, Green, Blue)
#   height     = 224    → pixels
#   width      = 224    → pixels
print(f"🖼️   Batch image tensor shape : {list(images.shape)}")
print(f"    [batch_size, channels, height, width]")
print(f"     = [{images.shape[0]}, {images.shape[1]}, {images.shape[2]}, {images.shape[3]}]")
print()

# labels shape: (batch_size,) — integer index for each image's class
print(f"🏷️   Batch labels shape        : {list(labels.shape)}")
print(f"    Label values (first 8)    : {labels[:8].tolist()}")
print(f"    Mapped class names        : {[class_names[i] for i in labels[:8].tolist()]}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Pixel value range check (after normalisation)
# ─────────────────────────────────────────────────────────────────────────────
print(f"📐  Pixel value range (after normalisation):")
print(f"    Min : {images.min():.4f}   Max : {images.max():.4f}")
print(f"    (Values are NOT in [0,1] because ImageNet normalisation was applied)")

print("\n✅  Pipeline working correctly! You can now run train.py\n")
print("=" * 55 + "\n")
