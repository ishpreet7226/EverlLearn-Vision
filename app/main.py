"""
EverLearn Vision – FastAPI Backend
====================================
Production-ready REST API for image classification with feedback storage
and self-improving model management.

Endpoints:
    GET  /              → Health check + model metadata
    POST /predict       → Upload an image, receive predicted label + confidence
    POST /feedback      → Store user correction of a prediction
    GET  /feedback      → Retrieve all stored feedback (paginated)
    POST /reload-model  → Hot-reload the latest model without restarting
    POST /retrain       → Trigger retraining pipeline in the background

Run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

from contextlib import asynccontextmanager
from io import BytesIO
import subprocess
import sys

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
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
    allow_origins=["http://localhost:3000"],
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
