"""
EverLearn Vision – Main Training Script
=========================================
Full training + validation pipeline with epoch logging and checkpointing.

Usage:
    python train.py
    python train.py --backbone resnet50 --epochs 20 --lr 0.0005
"""

import argparse
import logging
import os
import time

import torch
import torch.nn as nn

import config
from src.dataset import get_dataloaders
from src.model import build_model, get_device, SUPPORTED_BACKBONES
from src.trainer import train_one_epoch, evaluate


# ── Logging setup ──────────────────────────────────────────────────────────────
# Logs go to both the terminal AND a file (logs/training.log)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),                          # print to terminal
        logging.FileHandler("logs/training.log", mode="w"),  # save to file
    ],
)
log = logging.getLogger(__name__)


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    device = get_device()

    log.info("=" * 55)
    log.info("  EverLearn Vision — Training")
    log.info("=" * 55)
    log.info(f"  Device   : {device}")
    log.info(f"  Backbone : {args.backbone}")
    log.info(f"  Epochs   : {args.epochs}")
    log.info(f"  Batch    : {args.batch_size}")
    log.info(f"  LR       : {args.lr}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=config.DATA_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
    )
    num_classes = len(class_names)
    log.info(f"  Classes  : {class_names}")
    log.info(f"  Train    : {len(train_loader.dataset)} images  |  "
             f"Val: {len(val_loader.dataset)} images")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=True,
    ).to(device)

    # Loss function: CrossEntropyLoss — measures prediction error per batch
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Adam — adjusts weights using gradient + adaptive learning rates
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # LR Scheduler: halve the learning rate every `step_size` epochs
    # VIVA NOTE: As training progresses, smaller steps help fine-tune weights
    # without overshooting the optimal solution.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ── Training Loop ─────────────────────────────────────────────────────────
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = 0.0
    # Save as model.pth — will include weights + class labels
    save_path = os.path.join(config.CHECKPOINT_DIR, "model.pth")

    log.info("\n" + "-" * 55)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ── Train ──────────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # ── Validate ───────────────────────────────────────────────────────────
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        # Decay learning rate according to schedule
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        # ── Epoch summary ──────────────────────────────────────────────────────
        log.info(
            f"Epoch [{epoch:>2}/{args.epochs}]  "
            f"Train loss: {train_loss:.4f}  acc: {train_acc*100:5.2f}%  │  "
            f"Val loss: {val_loss:.4f}  acc: {val_acc*100:5.2f}%  │  "
            f"LR: {current_lr:.2e}  │  {epoch_time:.1f}s"
        )

        # ── Save best checkpoint ───────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Save weights AND class names together so we can reload and predict
            # without needing to know the folder structure at inference time.
            torch.save(
                {
                    "epoch":        epoch,
                    "backbone":     args.backbone,
                    "num_classes":  num_classes,
                    "class_names":  class_names,      # ← labels baked in!
                    "val_acc":      val_acc,
                    "model_state":  model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                save_path,
            )
            log.info(f"  ✅  New best  ({val_acc*100:.2f}%)  → saved to {save_path}")

    log.info("-" * 55)
    log.info(f"🎉  Training complete. Best val accuracy: {best_val_acc*100:.2f}%")
    log.info(f"    Checkpoint : {save_path}")
    log.info(f"    Log file   : logs/training.log")
    log.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EverLearn Vision – Train a classifier")
    parser.add_argument("--backbone",    default="resnet18", choices=SUPPORTED_BACKBONES,
                        help="Pretrained backbone architecture")
    parser.add_argument("--epochs",      type=int,   default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()
    main(args)
