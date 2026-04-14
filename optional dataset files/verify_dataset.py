"""
EverLearn Vision – Dataset Verification Script
===============================================

DO YOU NEED TO RUN THIS?
  → YES. Always run this BEFORE python train.py to avoid surprises mid-training.
  → It is read-only — it NEVER modifies, moves, or deletes any files. Safe to run anytime.

WHAT THIS SCRIPT DOES (plain English):
  Imagine you set up your data/ folder and start training, only to find out after 2 hours
  that one of your val/ class folders was empty, or was named 'Cat' instead of 'cat'.
  This script acts as a pre-flight checklist — it prints a table of every class and
  image count per folder, then cross-checks that the SAME classes exist in train AND val.

HOW IT WORKS — step by step:
  1. It scans top-level sub-folders inside data/ (e.g. train/, val/).
     These are called 'splits'.
  2. For each split folder, it calls count_images() which:
     - Opens every sub-folder (each = one class, e.g. cats/, dogs/)
     - Counts how many valid image files are inside (by file extension)
     - Returns a dict like {'cats': 800, 'dogs': 750}
  3. It prints a neat table: split name, class name, image count.
     Empty classes (count=0) get a ⚠️  warning — these would cause training errors.
  4. Cross-consistency check: it compares class names between train/ and val/.
     - If val/ has 'dog' but train/ has 'dogs' — MISMATCH warning (⚠️)
     - If all classes match across splits — GREEN ✅ shown

WHY IS THIS IMPORTANT? (Viva point)
  PyTorch's ImageFolder discovers classes FROM FOLDER NAMES at runtime.
  If train/ has folders ['cats', 'dogs'] but val/ has ['cat', 'Dog'],
  the class indices will be misaligned and accuracy metrics will be meaningless.
  This script catches such mismatches before wasting compute on training.

KEY LIBRARY — collections.defaultdict:
  A dictionary that automatically creates a default value (here: an empty set)
  for any new key you access. We use it to collect class names per split
  without having to check 'if key exists' every time.

KEY CONCEPT — set difference for cross-split check:
  ref_classes - all_classes[s] = classes in train but NOT in val  (missing)
  all_classes[s] - ref_classes = classes in val but NOT in train  (extra)
  Python sets make this a single, readable line of code.

Usage:
    python verify_dataset.py
    python verify_dataset.py --data_dir /path/to/your/data
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def count_images(folder: Path) -> dict:
    """Count images per class inside a split folder (e.g., train/val/test)."""
    class_counts = {}
    for class_dir in sorted(folder.iterdir()):
        if class_dir.is_dir():
            images = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            class_counts[class_dir.name] = len(images)
    return class_counts


def verify_dataset(data_dir: str) -> None:
    data_path = Path(data_dir)

    print("=" * 55)
    print("  EverLearn Vision — Dataset Verification")
    print("=" * 55)

    if not data_path.exists():
        print(f"\n❌  ERROR: '{data_dir}' does not exist.")
        print("   Create it and add your split folders (train/val/test).\n")
        return

    # Discover split folders (train, val, test, etc.)
    splits = [d for d in sorted(data_path.iterdir()) if d.is_dir()]

    if not splits:
        print(f"\n⚠️   No sub-folders found inside '{data_dir}'.")
        print("   Expected: data/train/class1, data/val/class1, …\n")
        return

    all_classes: dict[str, set] = defaultdict(set)
    total_images_overall = 0

    for split in splits:
        class_counts = count_images(split)
        if not class_counts:
            print(f"\n⚠️   '{split.name}' is empty or has no class sub-folders.")
            continue

        split_total = sum(class_counts.values())
        total_images_overall += split_total

        print(f"\n📂  {split.name}/  ({split_total} images)")
        print(f"    {'Class':<25} {'Images':>8}")
        print(f"    {'-'*25} {'-'*8}")
        for cls, count in class_counts.items():
            flag = " ⚠️  (empty!)" if count == 0 else ""
            print(f"    {cls:<25} {count:>8}{flag}")
            all_classes[split.name].add(cls)

    # Cross-split class consistency check
    if len(all_classes) > 1:
        split_names = list(all_classes.keys())
        ref_classes = all_classes[split_names[0]]
        print("\n🔍  Class consistency across splits:")
        for s in split_names[1:]:
            missing = ref_classes - all_classes[s]
            extra   = all_classes[s] - ref_classes
            if missing:
                print(f"    ⚠️   '{s}' is missing classes: {missing}")
            if extra:
                print(f"    ⚠️   '{s}' has extra classes:   {extra}")
            if not missing and not extra:
                print(f"    ✅  '{s}' matches '{split_names[0]}' classes.")

    print(f"\n{'='*55}")
    print(f"  Total images found: {total_images_overall}")
    print(f"{'='*55}\n")

    if total_images_overall == 0:
        print("❌  No images found. Check paths and file extensions.")
    else:
        print("✅  Dataset looks good! Ready to train.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify EverLearn Vision dataset structure.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to the root data directory (default: data/)",
    )
    args = parser.parse_args()
    verify_dataset(args.data_dir)
