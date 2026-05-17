"""
EverLearn Vision – Retraining Pipeline
========================================
Self-improving ML loop: reads user corrections from the database,
augments the original dataset, fine-tunes the current model,
and promotes the new version only if accuracy improves.

Workflow:
    1. Query feedback corrections from the database (SQLite or PostgreSQL)
    2. Find matching images in data/ and symlink to correct class folders
    3. Load current best model weights (fine-tuning, not from scratch)
    4. Train for N epochs on the augmented dataset
    5. Compare val_acc: promote if improved, keep old if not
    6. Save versioned checkpoint (model_vN.pth) regardless
    7. Log everything to MLflow

Usage:
    python retrain.py
    python retrain.py --epochs 5 --lr 0.0001

After successful promotion:
    curl -X POST http://localhost:8000/reload-model
"""

import argparse
import glob
import logging
import os
import shutil
import time

import mlflow
import requests
import torch
import torch.nn as nn

import config
from app.database import SessionLocal
from app.models import Feedback
from src.dataset import get_dataloaders
from src.model import build_model, get_device, SUPPORTED_BACKBONES
from src.trainer import train_one_epoch, evaluate

# ── Logging setup ──────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/retrain.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_next_version(checkpoint_dir: str) -> int:
    """Scan checkpoint_dir for model_v*.pth and return the next version number."""
    existing = glob.glob(os.path.join(checkpoint_dir, "model_v*.pth"))
    if not existing:
        return 1
    versions = []
    for path in existing:
        basename = os.path.basename(path)  # model_v3.pth
        try:
            v = int(basename.replace("model_v", "").replace(".pth", ""))
            versions.append(v)
        except ValueError:
            continue
    return max(versions) + 1 if versions else 1


def get_current_val_acc(checkpoint_path: str) -> float:
    """Read the val_acc from the current best checkpoint."""
    if not os.path.exists(checkpoint_path):
        return 0.0
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return ckpt.get("val_acc", 0.0)
    except Exception:
        return 0.0


def fetch_feedback_corrections() -> list[dict]:
    """
    Query PostgreSQL for feedback entries where the user corrected the prediction.
    Returns list of dicts: {image_name, predicted_label, actual_label, confidence}
    """
    db = SessionLocal()
    try:
        entries = (
            db.query(Feedback)
            .filter(Feedback.predicted_label != Feedback.actual_label)
            .all()
        )
        corrections = [
            {
                "image_name": e.image_name,
                "predicted_label": e.predicted_label,
                "actual_label": e.actual_label,
                "confidence": e.confidence,
            }
            for e in entries
        ]
        return corrections
    finally:
        db.close()


def find_image_in_dataset(image_name: str, data_dir: str) -> str | None:
    """
    Search the entire data/ directory tree for an image matching the filename.
    Returns the full path if found, None otherwise.
    """
    for root, _, files in os.walk(data_dir):
        if image_name in files:
            return os.path.join(root, image_name)
    return None


def augment_dataset_with_feedback(
    corrections: list[dict],
    data_dir: str,
    augmented_dir: str,
) -> int:
    """
    Create an augmented dataset by:
    1. Copying the original data/ directory structure
    2. Symlinking corrected images to their actual_label class folder

    Returns the number of corrections successfully applied.
    """
    # Copy original dataset structure using symlinks (fast, no disk duplication)
    if os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)

    for split in ["train", "val"]:
        src_split = os.path.join(data_dir, split)
        dst_split = os.path.join(augmented_dir, split)
        if not os.path.exists(src_split):
            continue

        for class_name in os.listdir(src_split):
            src_class = os.path.join(src_split, class_name)
            dst_class = os.path.join(dst_split, class_name)
            if not os.path.isdir(src_class):
                continue

            os.makedirs(dst_class, exist_ok=True)
            for img_file in os.listdir(src_class):
                src_img = os.path.join(src_class, img_file)
                dst_img = os.path.join(dst_class, img_file)
                if os.path.isfile(src_img) and not os.path.exists(dst_img):
                    os.symlink(os.path.abspath(src_img), dst_img)

    # Apply corrections: symlink each corrected image to its actual_label folder
    applied = 0
    for corr in corrections:
        # Find the original image
        img_path = find_image_in_dataset(corr["image_name"], data_dir)
        if img_path is None:
            log.warning(f"  ⚠️  Image not found in dataset: {corr['image_name']}")
            continue

        # Symlink to the CORRECT class folder in the training set
        actual_class_dir = os.path.join(augmented_dir, "train", corr["actual_label"])
        os.makedirs(actual_class_dir, exist_ok=True)

        # Use a unique name to avoid collisions
        dst_name = f"feedback_{corr['actual_label']}_{corr['image_name']}"
        dst_path = os.path.join(actual_class_dir, dst_name)

        if not os.path.exists(dst_path):
            os.symlink(os.path.abspath(img_path), dst_path)
            applied += 1

    return applied


def try_reload_api_model():
    """Call POST /reload-model on the running FastAPI server (if available)."""
    try:
        resp = requests.post("http://localhost:8000/reload-model", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            log.info(f"  🔄  API model reloaded: v{data.get('version', '?')}")
        else:
            log.warning(f"  ⚠️  API reload returned {resp.status_code}")
    except requests.ConnectionError:
        log.info("  ℹ️  FastAPI server not running — model will reload on next startup")
    except Exception as e:
        log.warning(f"  ⚠️  API reload failed: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    device = get_device()
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "model.pth")

    log.info("=" * 60)
    log.info("  EverLearn Vision — Retraining Pipeline")
    log.info("=" * 60)

    # ── Step 1: Fetch feedback corrections ────────────────────────────────────
    log.info("\n📋  Step 1: Fetching feedback corrections from database...")
    corrections = fetch_feedback_corrections()
    log.info(f"    Found {len(corrections)} corrections")

    if not corrections:
        log.info("    No corrections to apply. Nothing to retrain.")
        log.info("    Submit feedback via POST /feedback or the frontend first.")
        return

    for c in corrections:
        log.info(f"    • {c['image_name']}: {c['predicted_label']} → {c['actual_label']} "
                 f"(conf: {c['confidence']:.2f})")

    # ── Step 2: Get current model accuracy ────────────────────────────────────
    current_acc = get_current_val_acc(checkpoint_path)
    log.info(f"\n📊  Step 2: Current model val_acc: {current_acc*100:.2f}%")

    # ── Step 3: Determine version number ──────────────────────────────────────
    next_version = get_next_version(config.CHECKPOINT_DIR)
    log.info(f"    Next version: v{next_version}")

    # ── Step 4: Augment dataset with feedback ─────────────────────────────────
    augmented_dir = os.path.join(config.DATA_DIR, ".augmented")
    log.info(f"\n🔧  Step 3: Augmenting dataset with corrections...")
    applied = augment_dataset_with_feedback(corrections, config.DATA_DIR, augmented_dir)
    log.info(f"    Applied {applied} corrections to training set")

    if applied == 0:
        log.info("    No images could be matched. Check that image names in feedback")
        log.info("    correspond to files in the data/ directory.")
        return

    # ── Step 5: Load data from augmented dataset ──────────────────────────────
    log.info(f"\n📦  Step 4: Loading augmented dataset...")
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=augmented_dir,
        image_size=config.IMAGE_SIZE,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
    )
    num_classes = len(class_names)
    log.info(f"    Classes: {class_names}")
    log.info(f"    Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # ── Step 6: Load current model for fine-tuning ────────────────────────────
    log.info(f"\n🧠  Step 5: Loading current model for fine-tuning...")
    model = build_model(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=False,
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        log.info(f"    Loaded weights from {checkpoint_path}")
    else:
        log.info("    No existing checkpoint — training from pretrained backbone")
        model = build_model(
            num_classes=num_classes,
            backbone=args.backbone,
            pretrained=True,
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    # Use a lower learning rate for fine-tuning (gentler weight updates)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # ── Step 7: Train ─────────────────────────────────────────────────────────
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = 0.0
    versioned_path = os.path.join(config.CHECKPOINT_DIR, f"model_v{next_version}.pth")

    log.info(f"\n🏋️  Step 6: Fine-tuning for {args.epochs} epochs (lr={args.lr})...")
    log.info("-" * 60)

    # ── MLflow tracking ───────────────────────────────────────────────────────
    mlflow.set_experiment("EverLearn-Vision-Retrain")
    run_name = f"retrain_v{next_version}_{args.backbone}_ep{args.epochs}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "backbone": args.backbone,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "version": next_version,
            "feedback_corrections": len(corrections),
            "corrections_applied": applied,
            "current_val_acc": current_acc,
            "mode": "fine-tuning",
        })

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            epoch_time = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr,
            }, step=epoch)

            log.info(
                f"Epoch [{epoch:>2}/{args.epochs}]  "
                f"Train: {train_loss:.4f} / {train_acc*100:5.2f}%  │  "
                f"Val: {val_loss:.4f} / {val_acc*100:5.2f}%  │  "
                f"LR: {current_lr:.2e}  │  {epoch_time:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        log.info("-" * 60)

        # ── Step 8: Save versioned checkpoint ─────────────────────────────────
        checkpoint_data = {
            "epoch": args.epochs,
            "backbone": args.backbone,
            "num_classes": num_classes,
            "class_names": class_names,
            "val_acc": best_val_acc,
            "version": next_version,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }

        torch.save(checkpoint_data, versioned_path)
        log.info(f"\n💾  Saved: {versioned_path} (val_acc: {best_val_acc*100:.2f}%)")

        # ── Step 9: Accuracy gate — promote or discard ────────────────────────
        log.info(f"\n🔍  Step 7: Accuracy comparison")
        log.info(f"    Current model : {current_acc*100:.2f}%")
        log.info(f"    Retrained     : {best_val_acc*100:.2f}%")

        promoted = False
        if best_val_acc > current_acc:
            # Promote: copy to model.pth
            shutil.copy2(versioned_path, checkpoint_path)
            log.info(f"    ✅  PROMOTED — v{next_version} is the new production model!")
            promoted = True

            # Try to hot-reload the API server
            try_reload_api_model()
        else:
            log.info(f"    ❌  NOT PROMOTED — current model is still better")
            log.info(f"    The v{next_version} checkpoint is saved for reference")

        mlflow.log_metrics({
            "best_val_acc": best_val_acc,
            "current_model_acc": current_acc,
            "improvement": best_val_acc - current_acc,
        })
        mlflow.log_param("promoted", promoted)
        mlflow.log_artifact(versioned_path)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)
        log.info("\n🧹  Cleaned up augmented dataset")

    log.info("=" * 60)
    log.info(f"  Retraining complete. Version: v{next_version}")
    log.info(f"  Result: {'PROMOTED ✅' if promoted else 'NOT PROMOTED ❌'}")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EverLearn Vision – Retrain with feedback corrections"
    )
    parser.add_argument("--backbone", default="resnet18", choices=SUPPORTED_BACKBONES,
                        help="Backbone architecture (must match current model)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of fine-tuning epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for fine-tuning (default: 0.0001)")
    args = parser.parse_args()
    main(args)
