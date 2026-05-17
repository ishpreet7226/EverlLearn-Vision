"""
EverLearn Vision – Training with Status JSON
===============================================
Wrapper around the core training pipeline that writes epoch-by-epoch
progress to a JSON file so the frontend can poll it.

Called by the /teach/train endpoint as a subprocess:
    python -m app.train_with_status --backbone resnet18 --epochs 10 ...
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

import config
from src.dataset import get_dataloaders
from src.model import build_model, get_device, SUPPORTED_BACKBONES
from src.trainer import train_one_epoch, evaluate


def write_status(path: str, data: dict) -> None:
    """Atomically write status JSON (write to tmp then rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def main(args: argparse.Namespace) -> None:
    device = get_device()
    status_file = args.status_file

    write_status(status_file, {
        "status": "loading_data",
        "epoch": 0,
        "total_epochs": args.epochs,
        "train_loss": None, "train_acc": None,
        "val_loss": None, "val_acc": None,
        "best_val_acc": None,
    })

    # ── Data ──────────────────────────────────────────────────────────────────
    try:
        train_loader, val_loader, class_names = get_dataloaders(
            data_dir=config.DATA_DIR,
            image_size=config.IMAGE_SIZE,
            batch_size=args.batch_size,
            num_workers=config.NUM_WORKERS,
        )
    except Exception as e:
        write_status(status_file, {"status": "error", "error": f"Failed to load data: {e}"})
        return

    num_classes = len(class_names)

    # ── Model ─────────────────────────────────────────────────────────────────
    write_status(status_file, {
        "status": "building_model",
        "epoch": 0,
        "total_epochs": args.epochs,
        "classes": class_names,
        "train_loss": None, "train_acc": None,
        "val_loss": None, "val_acc": None,
        "best_val_acc": None,
    })

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
    save_path = os.path.join(config.CHECKPOINT_DIR, "model.pth")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        write_status(status_file, {
            "status": "training",
            "epoch": epoch,
            "total_epochs": args.epochs,
            "classes": class_names,
            "train_loss": None, "train_acc": None,
            "val_loss": None, "val_acc": None,
            "best_val_acc": round(best_val_acc, 4),
        })

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        epoch_time = time.time() - epoch_start

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "backbone": args.backbone,
                "num_classes": num_classes,
                "class_names": class_names,
                "val_acc": val_acc,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, save_path)

        write_status(status_file, {
            "status": "training",
            "epoch": epoch,
            "total_epochs": args.epochs,
            "classes": class_names,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "best_val_acc": round(best_val_acc, 4),
            "epoch_time": round(epoch_time, 1),
        })

    # ── Done ──────────────────────────────────────────────────────────────────
    write_status(status_file, {
        "status": "complete",
        "epoch": args.epochs,
        "total_epochs": args.epochs,
        "classes": class_names,
        "train_loss": round(train_loss, 4),
        "train_acc": round(train_acc, 4),
        "val_loss": round(val_loss, 4),
        "val_acc": round(val_acc, 4),
        "best_val_acc": round(best_val_acc, 4),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with status JSON output")
    parser.add_argument("--backbone", default="resnet18", choices=SUPPORTED_BACKBONES)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--status-file", default="logs/training_status.json")
    args = parser.parse_args()
    main(args)
