"""
EverLearn Vision – Main Training Script
=========================================
Entry point for training a classifier on any folder-structured dataset.

Usage:
    python train.py
    python train.py --backbone resnet50 --epochs 20 --lr 0.0005
"""

import argparse
import os
import torch
import torch.nn as nn

import config
from src.dataset import get_dataloaders
from src.model import build_model, SUPPORTED_BACKBONES
from src.trainer import train_one_epoch, evaluate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"\n🚀  EverLearn Vision")
    print(f"    Device   : {device}")
    print(f"    Backbone : {args.backbone}")
    print(f"    Epochs   : {args.epochs}")
    print(f"    Batch    : {args.batch_size}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=config.DATA_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
    )
    num_classes = len(class_names)
    print(f"    Classes  : {class_names}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ── Training Loop ─────────────────────────────────────────────────────────
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = 0.0
    best_ckpt = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_NAME)

    for epoch in range(1, args.epochs + 1):
        print(f"  Epoch [{epoch}/{args.epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"    Train  → loss: {train_loss:.4f}  acc: {train_acc*100:.2f}%\n"
            f"    Val    → loss:  {val_loss:.4f}  acc:  {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            print(f"    ✅  Best model saved → {best_ckpt}")
        print()

    print(f"🎉  Training complete. Best val accuracy: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EverLearn Vision – Train a classifier")
    parser.add_argument("--backbone", default="resnet18",
                        choices=SUPPORTED_BACKBONES,
                        help="Pretrained backbone architecture")
    parser.add_argument("--epochs",     type=int,   default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()
    main(args)
