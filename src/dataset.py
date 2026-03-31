"""
EverLearn Vision – DataLoader Utilities
========================================
Loads any folder-structured dataset automatically.

Usage:
    from src.dataset import get_dataloaders
    train_loader, val_loader, class_names = get_dataloaders("data")
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(split: str, image_size: tuple[int, int]) -> transforms.Compose:
    """Return augmentation pipeline for train vs val/test."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_dataloaders(
    data_dir: str,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Build train and val DataLoaders from:
        data_dir/train/<class_name>/...
        data_dir/val/<class_name>/...

    Returns:
        train_loader, val_loader, class_names
    """
    data_path = Path(data_dir)
    train_dataset = datasets.ImageFolder(
        root=str(data_path / "train"),
        transform=get_transforms("train", image_size),
    )
    val_dataset = datasets.ImageFolder(
        root=str(data_path / "val"),
        transform=get_transforms("val", image_size),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes
