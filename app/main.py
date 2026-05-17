"""
EverLearn Vision – FastAPI Backend
====================================
Production-ready REST API for image classification with feedback storage,
self-improving model management, and Teachable Machine workflow.

Endpoints:
    GET  /                        → Health check + model metadata
    POST /predict                 → Upload an image, receive predicted label + confidence
    POST /feedback                → Store user correction of a prediction
    GET  /feedback                → Retrieve all stored feedback (paginated)
    POST /reload-model            → Hot-reload the latest model without restarting
    POST /retrain                 → Trigger retraining pipeline in the background
    POST /teach/upload-dataset    → Upload images organised by class
    POST /teach/train             → Launch training as a background process
    GET  /teach/training-status   → Poll training progress

Run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

from contextlib import asynccontextmanager
from io import BytesIO
import json
import os
import random
import shutil
import subprocess
import sys
from typing import List

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.model_loader import ModelBundle, load_model
from app.inference import run_prediction
from app.database import Base, engine, get_db
from app.schemas import FeedbackCreate, FeedbackListResponse, FeedbackResponse
from app import crud

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/model.pth"
DATA_DIR = "data"
TRAINING_STATUS_FILE = "logs/training_status.json"
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
    "image/tiff",
}

# ── Global model reference ────────────────────────────────────────────────────
model_bundle: ModelBundle | None = None


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and create DB tables at startup."""
    global model_bundle
    print("🚀  Starting EverLearn Vision API...")

    # Create database tables (if they don't already exist)
    try:
        Base.metadata.create_all(bind=engine)
        print("✅  Database tables ready")
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        print("    Feedback endpoints will not work.")

    # Load ML model
    try:
        model_bundle = load_model(CHECKPOINT_PATH)
        print("✅  Model ready — accepting requests")
    except FileNotFoundError as e:
        print(f"❌  {e}")
        print("    The server will start, but /predict will return 503.")
    yield
    # Cleanup (if needed in the future)
    print("👋  Shutting down EverLearn Vision API")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="EverLearn Vision API",
    description="Image classification API powered by a fine-tuned ResNet model.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow the Next.js frontend to call this API ────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response schemas ──────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: int | None = None
    backbone: str | None = None
    classes: list[str] | None = None
    device: str | None = None


class PredictResponse(BaseModel):
    label: str
    confidence: float
    all_probabilities: dict[str, float]


class ErrorResponse(BaseModel):
    detail: str


# ── GET / — Health check ──────────────────────────────────────────────────────
@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns server status and loaded model metadata.",
)
async def health_check():
    if model_bundle is None:
        return HealthResponse(status="degraded", model_loaded=False)

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=model_bundle.version,
        backbone=model_bundle.backbone,
        classes=model_bundle.class_names,
        device=str(model_bundle.device),
    )


# ── POST /predict — Image classification ─────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or corrupt image"},
        500: {"model": ErrorResponse, "description": "Inference failure"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary="Classify an uploaded image",
    description=(
        "Upload an image file (JPEG, PNG, BMP, WEBP) and receive the predicted "
        "class label, confidence score, and per-class probabilities."
    ),
)
async def predict(file: UploadFile = File(..., description="Image file to classify")):
    # ── Guard: model must be loaded ───────────────────────────────────────────
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Ensure checkpoints/model.pth exists and "
                "restart the server."
            ),
        )

    # ── Validate content type ─────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type: '{file.content_type}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # ── Read and decode the image ─────────────────────────────────────────────
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file could not be decoded as an image. It may be corrupt.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded image: {e}",
        )

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        result = run_prediction(image, model_bundle)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {e}",
        )

    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        all_probabilities=result.all_probabilities,
    )


# ── POST /feedback — Store user correction ────────────────────────────────────
@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=201,
    summary="Submit prediction feedback",
    description=(
        "Store a user's correction when the model's prediction was wrong. "
        "This data can later be used for model retraining."
    ),
)
def submit_feedback(
    feedback_data: FeedbackCreate,
    db: Session = Depends(get_db),
):
    try:
        row = crud.create_feedback(db, feedback_data)
        return row
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save feedback: {e}",
        )


# ── GET /feedback — Retrieve stored feedback ──────────────────────────────────
@app.get(
    "/feedback",
    response_model=FeedbackListResponse,
    summary="List all feedback",
    description="Retrieve all stored feedback entries, newest first, with pagination.",
)
def list_feedback(
    skip: int = Query(0, ge=0, description="Number of entries to skip"),
    limit: int = Query(50, ge=1, le=200, description="Max entries to return"),
    db: Session = Depends(get_db),
):
    try:
        entries = crud.get_all_feedback(db, skip=skip, limit=limit)
        count = crud.get_feedback_count(db)
        return FeedbackListResponse(count=count, feedback=entries)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve feedback: {e}",
        )


# ── POST /reload-model — Hot-reload latest model ──────────────────────────────
@app.post(
    "/reload-model",
    summary="Hot-reload the model",
    description=(
        "Reload checkpoints/model.pth into memory without restarting the server. "
        "Called automatically after successful retraining."
    ),
)
def reload_model():
    global model_bundle
    try:
        model_bundle = load_model(CHECKPOINT_PATH)
        return {
            "status": "reloaded",
            "version": model_bundle.version,
            "backbone": model_bundle.backbone,
            "classes": model_bundle.class_names,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {e}",
        )


# ── POST /retrain — Trigger retraining pipeline ─────────────────────────────
@app.post(
    "/retrain",
    summary="Trigger model retraining",
    description=(
        "Launch the retraining pipeline as a background process. "
        "The pipeline reads feedback corrections from the database, "
        "fine-tunes the model, and promotes it if accuracy improves."
    ),
)
def trigger_retrain():
    try:
        # Launch retrain.py as a detached background process
        process = subprocess.Popen(
            [sys.executable, "retrain.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "status": "started",
            "pid": process.pid,
            "message": "Retraining pipeline launched in background. "
                       "Check logs/retrain.log for progress.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start retraining: {e}",
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TEACHABLE MACHINE ENDPOINTS                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── POST /teach/upload-dataset ────────────────────────────────────────────────
@app.post(
    "/teach/upload-dataset",
    summary="Upload dataset for Teachable Machine training",
    description=(
        "Upload images grouped by class. Each file's class membership is "
        "specified via the `class_names` form field (parallel arrays). "
        "Images are saved to data/train and data/val with an 80/20 split."
    ),
)
async def upload_dataset(
    files: List[UploadFile] = File(..., description="Image files to upload"),
    class_names: List[str] = Form(..., description="Class name for each file (same order)"),
):
    """
    Accept a batch of images with class labels and build the dataset.

    The frontend sends one `files[]` entry per image and a matching
    `class_names[]` entry with the class name that image belongs to.
    """
    if len(files) != len(class_names):
        raise HTTPException(
            status_code=400,
            detail=f"Mismatch: {len(files)} files but {len(class_names)} class names.",
        )

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # ── Clear existing dataset ────────────────────────────────────────────────
    for split in ["train", "val"]:
        split_dir = os.path.join(DATA_DIR, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

    # ── Group files by class ──────────────────────────────────────────────────
    class_files: dict[str, list[tuple[str, bytes]]] = {}
    for file, cls_name in zip(files, class_names):
        cls_name = cls_name.strip()
        if not cls_name:
            continue
        contents = await file.read()
        if cls_name not in class_files:
            class_files[cls_name] = []
        class_files[cls_name].append((file.filename, contents))

    if len(class_files) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 classes are required for training.",
        )

    # ── Write to disk with 80/20 train/val split ─────────────────────────────
    stats = {}
    for cls_name, imgs in class_files.items():
        random.shuffle(imgs)
        split_idx = max(1, int(len(imgs) * 0.8))  # At least 1 in train
        train_imgs = imgs[:split_idx]
        val_imgs = imgs[split_idx:] if len(imgs) > 1 else imgs[:1]  # Copy 1 to val if tiny

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
            split_class_dir = os.path.join(DATA_DIR, split_name, cls_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for filename, data in split_imgs:
                safe_name = filename.replace("/", "_").replace("\\", "_")
                filepath = os.path.join(split_class_dir, safe_name)
                with open(filepath, "wb") as f:
                    f.write(data)

        stats[cls_name] = {"train": len(train_imgs), "val": len(val_imgs)}

    return {
        "status": "dataset_ready",
        "classes": list(class_files.keys()),
        "stats": stats,
        "total_images": len(files),
    }


# ── POST /teach/train ────────────────────────────────────────────────────────
@app.post(
    "/teach/train",
    summary="Start model training",
    description=(
        "Launch the training pipeline as a background process. "
        "The frontend can poll /teach/training-status for progress."
    ),
)
async def teach_train(
    backbone: str = Form("resnet18"),
    epochs: int = Form(10),
    lr: float = Form(0.001),
    batch_size: int = Form(32),
):
    # Validate backbone
    supported = ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_small"]
    if backbone not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported backbone '{backbone}'. Choose from: {supported}",
        )

    # Check dataset exists
    train_dir = os.path.join(DATA_DIR, "train")
    if not os.path.exists(train_dir) or not any(os.scandir(train_dir)):
        raise HTTPException(
            status_code=400,
            detail="No training data found. Upload a dataset first via /teach/upload-dataset.",
        )

    # Remove old checkpoint to force fresh training
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    # Write initial status
    os.makedirs("logs", exist_ok=True)
    with open(TRAINING_STATUS_FILE, "w") as f:
        json.dump({
            "status": "starting",
            "epoch": 0,
            "total_epochs": epochs,
            "train_loss": None,
            "train_acc": None,
            "val_loss": None,
            "val_acc": None,
            "best_val_acc": None,
        }, f)

    # Launch the training wrapper (writes status JSON each epoch)
    process = subprocess.Popen(
        [
            sys.executable, "-m", "app.train_with_status",
            "--backbone", backbone,
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--batch_size", str(batch_size),
            "--status-file", TRAINING_STATUS_FILE,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return {
        "status": "started",
        "pid": process.pid,
        "backbone": backbone,
        "epochs": epochs,
    }


# ── GET /teach/training-status ────────────────────────────────────────────────
@app.get(
    "/teach/training-status",
    summary="Get training progress",
    description="Returns the current training status including epoch, accuracy, and loss.",
)
async def training_status():
    if not os.path.exists(TRAINING_STATUS_FILE):
        return {"status": "idle"}

    try:
        with open(TRAINING_STATUS_FILE, "r") as f:
            data = json.load(f)

        # If training completed, auto-reload the model
        if data.get("status") == "complete" and os.path.exists(CHECKPOINT_PATH):
            global model_bundle
            try:
                model_bundle = load_model(CHECKPOINT_PATH)
                data["model_reloaded"] = True
            except Exception:
                data["model_reloaded"] = False

        return data
    except (json.JSONDecodeError, IOError):
        return {"status": "unknown", "error": "Could not read status file."}


# ── POST /teach/upload-folder — Folder-based dataset upload ───────────────────
@app.post(
    "/teach/upload-folder",
    summary="Upload a folder-structured dataset",
    description=(
        "Upload an entire dataset folder. Subfolder names become class labels. "
        "Images are auto-split into train (80%) and val (20%)."
    ),
)
async def upload_folder(
    files: List[UploadFile] = File(..., description="All image files from the folder"),
    paths: List[str] = Form(..., description="Relative paths for each file (preserves folder structure)"),
):
    """
    Accept files uploaded via webkitdirectory input. Each file comes with its
    relative path like 'dataset/cats/img1.jpg'. We parse the subfolder name
    as the class label.
    """
    if len(files) != len(paths):
        raise HTTPException(
            status_code=400,
            detail=f"Mismatch: {len(files)} files but {len(paths)} paths.",
        )

    # ── Parse class names from folder paths ───────────────────────────────────
    # Expected path format: <root_folder>/<class_name>/<image_file>
    # or just <class_name>/<image_file>
    class_files: dict[str, list[tuple[str, bytes]]] = {}

    for file, rel_path in zip(files, paths):
        # Skip non-image files
        if not file.content_type or not file.content_type.startswith("image/"):
            continue

        parts = rel_path.replace("\\", "/").strip("/").split("/")

        # We need at least 2 parts: class_name/image_file
        # If 3+ parts: root_folder/class_name/image_file → use parts[-2]
        if len(parts) < 2:
            continue  # skip files in root (no class folder)

        class_name = parts[-2]  # The parent folder = class name

        # Skip hidden folders
        if class_name.startswith(".") or class_name.startswith("_"):
            continue

        contents = await file.read()
        if class_name not in class_files:
            class_files[class_name] = []
        class_files[class_name].append((file.filename or parts[-1], contents))

    if len(class_files) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 class folders, found: {list(class_files.keys()) or 'none'}. "
                   "Make sure your folder has subfolders like: dataset/cats/*.jpg, dataset/dogs/*.jpg",
        )

    # ── Clear existing dataset ────────────────────────────────────────────────
    for split in ["train", "val"]:
        split_dir = os.path.join(DATA_DIR, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

    # ── Write to disk with 80/20 train/val split ─────────────────────────────
    stats = {}
    for cls_name, imgs in class_files.items():
        random.shuffle(imgs)
        split_idx = max(1, int(len(imgs) * 0.8))
        train_imgs = imgs[:split_idx]
        val_imgs = imgs[split_idx:] if len(imgs) > 1 else imgs[:1]

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
            split_class_dir = os.path.join(DATA_DIR, split_name, cls_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for filename, data in split_imgs:
                safe_name = filename.replace("/", "_").replace("\\", "_")
                filepath = os.path.join(split_class_dir, safe_name)
                with open(filepath, "wb") as f:
                    f.write(data)

        stats[cls_name] = {"train": len(train_imgs), "val": len(val_imgs)}

    return {
        "status": "dataset_ready",
        "classes": list(class_files.keys()),
        "stats": stats,
        "total_images": sum(len(imgs) for imgs in class_files.values()),
    }


# ── GET /teach/val-images — List validation images for testing ────────────────
@app.get(
    "/teach/val-images",
    summary="List validation images",
    description="Returns validation image paths grouped by class for post-training testing.",
)
async def list_val_images():
    import base64

    val_dir = os.path.join(DATA_DIR, "val")
    if not os.path.exists(val_dir):
        return {"classes": {}, "total": 0}

    result = {}
    total = 0

    for cls_name in sorted(os.listdir(val_dir)):
        cls_dir = os.path.join(val_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        images = []
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            if not os.path.isfile(fpath):
                continue

            # Determine content type
            ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
            content_type = {
                "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp",
                "bmp": "image/bmp",
            }.get(ext, "image/jpeg")

            try:
                with open(fpath, "rb") as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode("ascii")
                images.append({
                    "filename": fname,
                    "data_url": f"data:{content_type};base64,{b64}",
                })
                total += 1
            except Exception:
                continue

        if images:
            result[cls_name] = images

    return {"classes": result, "total": total}
