"""
EverLearn Vision – Dataset Cleaner
=====================================

DO YOU NEED TO RUN THIS?
  → YES, if you downloaded images from the internet and aren't sure they're all valid.
  → YES, if training crashes mid-epoch with an 'UnidentifiedImageError'.
  → NO,  if your dataset was already cleaned or came from a trusted source.

WHAT THIS SCRIPT DOES (plain English):
  Not every .jpg file on the internet is a real JPEG. Some are:
    - Truncated (download was interrupted mid-file)
    - Mislabeled (a .txt file renamed to .jpg)
    - Zero-byte empty files
  If even ONE such file enters the DataLoader, training will CRASH completely.
  This script scans every image file under data/, tries to open each one
  with PIL, and reports (or optionally deletes) the broken ones.

HOW IT WORKS — step by step:
  1. Path.rglob('*') walks EVERY file recursively under data/ (train, val, any depth).
  2. We filter to only known image extensions (.jpg, .jpeg, .png, .bmp, .webp, .tiff).
  3. For each image, we open it with PIL and call img.verify().
     - verify() checks the file header/structure WITHOUT fully decoding pixel data.
     - This makes it fast — it doesn't load the full image into memory.
  4. If verify() raises ANY Exception (broken file), we add it to bad_files list.
  5. Default is DRY RUN — it only prints the bad files, does NOT delete them.
  6. Add --delete flag to actually call Path.unlink() which permanently removes the file.

WHY TWO MODES (dry run vs delete)?
  Deleting files is IRREVERSIBLE. A dry run first lets you inspect the list
  and make sure the 'bad' files are actually garbage (not just files you need).
  ALWAYS run without --delete first, then run with it if the list looks right.

KEY LIBRARY — PIL Image.open() + img.verify():
  PIL = Python Imaging Library (maintained as Pillow).
  img.verify() does a lightweight structural integrity check.
  It does NOT decode pixels — just checks headers and file structure.
  Calling open() + verify() together is the standard Python idiom for
  detecting corrupted image files without loading them fully into RAM.

KEY LIBRARY — Path.unlink():
  pathlib's way to delete a file. Equivalent to os.remove().
  After unlink(), the file is permanently gone from the filesystem.

Usage:
    python clean_dataset.py              # dry run (shows bad files only)
    python clean_dataset.py --delete     # actually delete bad files
    python clean_dataset.py --data_dir /path/to/data --delete
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
