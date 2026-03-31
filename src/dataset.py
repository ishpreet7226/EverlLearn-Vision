"""
EverLearn Vision – DataLoader Utilities
========================================
Loads any folder-structured dataset automatically using torchvision.

VIVA NOTE — Why ImageFolder?
  torchvision.datasets.ImageFolder scans a root folder and:
  - Treats every sub-folder as a class label
  - Assigns integer indices automatically (sorted alphabetically)
  - Returns (image_tensor, class_index) pairs
  So you never need to write a custom Dataset — just keep your files organised!

Usage:
    from src.dataset import get_dataloaders
    train_loader, val_loader, class_names = get_dataloaders("data")
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# pin_memory speeds up CPU→GPU transfers, but ONLY works with CUDA.
# On MPS (Apple Silicon) or CPU it causes a warning and is ignored anyway.
_PIN_MEMORY = torch.cuda.is_available()


# ── Transforms ────────────────────────────────────────────────────────────────
# VIVA NOTE — What is a transform pipeline?
#   A pipeline of operations applied to every image before it reaches the model.
#   transforms.Compose chains them in order — each output feeds into the next.

def get_transforms(split: str, image_size: tuple[int, int]) -> transforms.Compose:
    """
    Return the correct augmentation pipeline for 'train' or 'val'.

    VIVA NOTE — Why different transforms for train vs val?
      During TRAINING we add random flips, crops, and colour jitter to
      artificially diversify the data and prevent overfitting (data augmentation).
      During VALIDATION we want consistent, deterministic results so we only
      resize + centre-crop — never randomise.
    """
    if split == "train":
        return transforms.Compose([
            # Randomly crop and scale the image to image_size.
            # Forces the model to learn from different regions, not just the centre.
            transforms.RandomResizedCrop(image_size),

            # Randomly flip 50% of images horizontally.
            # Cats/dogs look the same mirrored — free extra training data.
            transforms.RandomHorizontalFlip(),

            # Slightly vary brightness, contrast, saturation.
            # Makes the model robust to different lighting conditions.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

            # Convert PIL Image (H, W, C) uint8 → FloatTensor (C, H, W) in [0, 1].
            # PyTorch models expect tensors, not PIL images.
            transforms.ToTensor(),

            # Normalise each channel using ImageNet statistics.
            # VIVA NOTE: mean & std come from the ImageNet dataset (1.2M images).
            # Since our backbone was pretrained on ImageNet, using the same
            # normalisation puts our pixel values in the same range the model
            # was originally trained on — improving transfer learning quality.
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   # R, G, B means
                                 std=[0.229, 0.224, 0.225]),    # R, G, B stds
        ])
    else:
        # Val/Test: deterministic pipeline — no randomness
        return transforms.Compose([
            transforms.Resize(image_size),       # Scale shorter edge to 224
            transforms.CenterCrop(image_size),   # Crop exactly 224×224 from centre
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ── DataLoader Builder ─────────────────────────────────────────────────────────
def get_dataloaders(
    data_dir: str,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Build train and val DataLoaders from folder structure:
        data_dir/train/<class_name>/image.jpg
        data_dir/val/<class_name>/image.jpg

    Returns:
        train_loader  – shuffled batches for training
        val_loader    – ordered batches for evaluation
        class_names   – list of class labels, e.g. ['cat', 'dog']

    VIVA NOTE — What is a DataLoader?
      DataLoader wraps a Dataset and handles:
        - Batching:     Groups individual samples into batches of size `batch_size`
        - Shuffling:    Randomises order each epoch (train only) to reduce bias
        - Parallelism:  Loads data in `num_workers` background processes so
                        the GPU never waits for data
        - pin_memory:   Locks loaded tensors in CPU RAM for faster GPU transfer
    """
    data_path = Path(data_dir)

    # VIVA NOTE — ImageFolder auto-discovers class names from sub-folder names.
    # class_to_idx maps {"cat": 0, "dog": 1} — sorted alphabetically.
    train_dataset = datasets.ImageFolder(
        root=str(data_path / "train"),
        transform=get_transforms("train", image_size),
    )
    val_dataset = datasets.ImageFolder(
        root=str(data_path / "val"),
        transform=get_transforms("val", image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Randomise order every epoch → reduces overfitting
        num_workers=num_workers,
        pin_memory=_PIN_MEMORY,  # Only True on CUDA; False on MPS/CPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # Keep order consistent for reproducible metrics
        num_workers=num_workers,
        pin_memory=_PIN_MEMORY,
    )

    return train_loader, val_loader, train_dataset.classes
