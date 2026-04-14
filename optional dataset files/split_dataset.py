"""
EverLearn Vision – Train/Val Split Script
==========================================

DO YOU NEED TO RUN THIS?
  → YES, only if your dataset is NOT already split into train/ and val/ folders.
  → NO,  if you already have both data/train/ and data/val/ ready.

WHAT THIS SCRIPT DOES (plain English):
  When you download a dataset from Kaggle or elsewhere, it usually comes as:
      data/train/cats/...  (all images in one place)
  But PyTorch needs a separate val/ folder to test accuracy during training.
  This script MOVES a random percentage of images from train/ → val/
  without copying or duplicating — it physically relocates files.

HOW IT WORKS — step by step:
  1. It reads --val_split (default 0.20 = 20%) from the command line.
  2. It scans every class folder inside data/train/  (e.g. cats/, dogs/).
  3. For each class, it shuffles the image list randomly (seeded for reproducibility).
  4. It takes the first 20% of that shuffled list as the validation images.
  5. It creates data/val/<class>/ if it doesn't exist.
  6. It MOVES (not copies) each chosen image there using shutil.move().
  Result: data/train/ now has 80% of images, data/val/ has 20%.

WHY 80/20 IS THE STANDARD SPLIT:
  You want most data for training (so the model learns well).
  You need SOME held-out data for validation (to measure real accuracy).
  80/20 is the industry-standard compromise, attributed to the Pareto Principle.

KEY LIBRARY — shutil.move(src, dst):
  From Python's standard 'shutil' module (shell utilities).
  Moves a file from src to dst path. If on the same filesystem, it's an
  instant rename (no file data copied). Cross-filesystem = copy + delete.
  We use this (not shutil.copy) because we DON'T want dataset duplication.

KEY LIBRARY — random.seed(42):
  Setting a seed means the 'randomness' is REPRODUCIBLE. If two people
  run this script with seed=42, they get the exact same train/val split.
  This is critical for reproducible experiments in academic settings.

Usage:
    python split_dataset.py                   # 80/20 split (default)
    python split_dataset.py --val_split 0.15  # 85/15 split
    python split_dataset.py --seed 99         # different random shuffle
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(data_dir: str, val_split: float, seed: int) -> None:
    random.seed(seed)
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir   = data_path / "val"

    if not train_dir.exists():
        print(f"❌  '{train_dir}' does not exist. Nothing to split.")
        return

    class_dirs = [d for d in sorted(train_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        print("❌  No class sub-folders found inside data/train/")
        return

    print(f"\n📂  Splitting  →  train: {(1-val_split)*100:.0f}%  |  val: {val_split*100:.0f}%\n")

    SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    for cls_dir in class_dirs:
        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in SUPPORTED]
        if not images:
            print(f"  ⚠️   {cls_dir.name}: no images found, skipping.")
            continue

        random.shuffle(images)
        n_val = max(1, int(len(images) * val_split))
        val_images = images[:n_val]

        dest_cls = val_dir / cls_dir.name
        dest_cls.mkdir(parents=True, exist_ok=True)

        for img in val_images:
            shutil.move(str(img), str(dest_cls / img.name))

        print(f"  ✅  {cls_dir.name:<20}  moved {n_val:>4} → val/   "
              f"(train: {len(images)-n_val}, val: {n_val})")

    print("\n✅  Split complete! Run `python verify_dataset.py` to confirm.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split training data into train/val sets.")
    parser.add_argument("--data_dir",   default="data",  help="Root data directory (default: data/)")
    parser.add_argument("--val_split",  type=float, default=0.2, help="Fraction for val set (default: 0.2)")
    parser.add_argument("--seed",       type=int,   default=42,  help="Random seed for reproducibility")
    args = parser.parse_args()

    if not 0 < args.val_split < 1:
        print("❌  --val_split must be between 0 and 1 (e.g. 0.2 for 20%)")
    else:
        split_dataset(args.data_dir, args.val_split, args.seed)
