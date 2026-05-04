"""
EverLearn Vision – Inference Pipeline (FastAPI)
================================================
Handles the full image → prediction pipeline:

  1. Accept a PIL Image (already decoded from upload bytes)
  2. Preprocess: Resize → CenterCrop → ToTensor → Normalize (ImageNet stats)
  3. Forward pass through the model (no gradient tracking)
  4. Softmax: convert raw logits → class probabilities
  5. Return PredictionResult with label, confidence, and all probabilities

The preprocessing is identical to the validation transforms used during
training, ensuring the model receives inputs in the same distribution
it was trained on.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.model_loader import ModelBundle


# ── Preprocessing (identical to validation transforms) ────────────────────────
IMAGE_SIZE = (224, 224)

_preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),         # Scale shorter edge to 224
    transforms.CenterCrop(IMAGE_SIZE),     # Crop exactly 224×224 from centre
    transforms.ToTensor(),                 # PIL Image → FloatTensor [0, 1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],        # ImageNet channel means
        std=[0.229, 0.224, 0.225],         # ImageNet channel stds
    ),
])


@dataclass
class PredictionResult:
    """Structured prediction output returned to the API caller."""
    label: str
    confidence: float
    all_probabilities: dict[str, float]


def run_prediction(image: Image.Image, bundle: ModelBundle) -> PredictionResult:
    """
    Run the full prediction pipeline on a single PIL image.

    Args:
        image  : A PIL Image already converted to RGB.
        bundle : ModelBundle from model_loader (model, class_names, device).

    Returns:
        PredictionResult with predicted label, confidence, and per-class probs.

    Raises:
        RuntimeError: If the forward pass fails (e.g. tensor shape mismatch).
    """
    # Step 1 — Preprocess: PIL Image → normalised tensor (3, 224, 224)
    tensor = _preprocess(image)

    # Step 2 — Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
    # Models always expect a batch, even for a single image
    tensor = tensor.unsqueeze(0).to(bundle.device)

    # Step 3 — Forward pass with no gradient tracking (saves memory + speed)
    with torch.no_grad():
        logits = bundle.model(tensor)  # shape: (1, num_classes)

    # Step 4 — Softmax: convert raw logits → probabilities (sum to 1.0)
    probs = F.softmax(logits, dim=1).squeeze(0)  # shape: (num_classes,)

    # Step 5 — Extract top prediction
    confidence, class_idx = probs.max(dim=0)
    label = bundle.class_names[class_idx.item()]

    # Step 6 — Build per-class probability dict
    all_probabilities = {
        cls: round(probs[i].item(), 4)
        for i, cls in enumerate(bundle.class_names)
    }

    return PredictionResult(
        label=label,
        confidence=round(confidence.item(), 4),
        all_probabilities=all_probabilities,
    )
