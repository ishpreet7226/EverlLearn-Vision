"""
EverLearn Vision – Train/Val Split Script
==========================================
Moves a percentage of images from data/train/<class>
into data/val/<class> randomly.

Usage:
    python split_dataset.py                  # 80/20 split (default)
    python split_dataset.py --val_split 0.15  # 85/15 split
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
