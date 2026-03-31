"""
EverLearn Vision – Dataset Verification Script
===============================================
Run this BEFORE training to validate your dataset structure.

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
