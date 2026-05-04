"""
EverLearn Vision – FastAPI Backend
====================================
Production-ready REST API for image classification.

Endpoints:
    GET  /          → Health check + model metadata
    POST /predict   → Upload an image, receive predicted label + confidence

Run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.model_loader import ModelBundle, load_model
from app.inference import run_prediction

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
    """Load the model once at startup; release on shutdown."""
    global model_bundle
    print("🚀  Starting EverLearn Vision API...")
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


# ── Response schemas ──────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
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
