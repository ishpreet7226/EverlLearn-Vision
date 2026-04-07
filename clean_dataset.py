"""
EverLearn Vision – Dataset Cleaner
=====================================
Scans all images in data/ and removes any that PIL cannot open.
Run this ONCE before training if you see UnidentifiedImageError.

Usage:
    python clean_dataset.py              # dry run (shows bad files only)
    python clean_dataset.py --delete     # actually delete bad files
"""

import argparse
from pathlib import Path
from PIL import Image

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def scan_and_clean(data_dir: str, delete: bool) -> None:
    data_path = Path(data_dir)
    all_images = [
        f for f in data_path.rglob("*")
        if f.suffix.lower() in SUPPORTED
    ]

    print(f"\n🔍  Scanning {len(all_images)} images in '{data_dir}'...\n")
    bad_files = []

    for img_path in all_images:
        try:
            with Image.open(img_path) as img:
                img.verify()   # verify checks the file without fully decoding it
        except Exception:
            bad_files.append(img_path)

    if not bad_files:
        print("✅  All images are valid! No corrupt files found.\n")
        return

    print(f"⚠️   Found {len(bad_files)} corrupt/unreadable image(s):\n")
    for f in bad_files:
        print(f"    ❌  {f}")

    if delete:
        print(f"\n🗑️   Deleting {len(bad_files)} bad file(s)...")
        for f in bad_files:
            f.unlink()
            print(f"    Deleted: {f}")
        print(f"\n✅  Cleanup complete. Re-run verify_dataset.py to confirm.\n")
    else:
        print("\n💡  Run with --delete to remove these files:\n")
        print("    python3 clean_dataset.py --delete\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Root data directory")
    parser.add_argument("--delete", action="store_true",
                        help="Delete corrupt files (default: dry run only)")
    args = parser.parse_args()
    scan_and_clean(args.data_dir, args.delete)
