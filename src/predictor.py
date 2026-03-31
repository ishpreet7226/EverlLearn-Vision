"""
EverLearn Vision – Prediction Engine
======================================
Loads a saved checkpoint (model.pth) and predicts the class of a single image.

VIVA NOTE — Prediction Pipeline (step-by-step):
  1. Load checkpoint → restore model architecture + weights + class names
  2. Preprocess image → same transforms used during validation
     (Resize → CenterCrop → ToTensor → Normalize)
  3. Forward pass → model outputs raw logits for each class
  4. Softmax → convert logits to probabilities (sum to 1.0)
  5. argmax → pick the class with the highest probability
  6. Return class name + confidence score

VIVA NOTE — Why do we need preprocessing at inference time?
  The model was trained on 224×224 normalised tensors.
  If we feed a raw 1080p photo, the model gets inputs it has never seen —
  it would produce garbage predictions. Preprocessing ensures the input
  looks exactly like the training data.

Usage:
    from src.predictor import Predictor
    p = Predictor("checkpoints/model.pth")
    label, confidence = p.predict("cat.jpg")
"""

from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import build_model, get_device


# ── Inference-time transform (same as val — deterministic, no augmentation) ──
def _inference_transforms(image_size: tuple[int, int] = (224, 224)) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class Predictor:
    """
    Loads a saved EverLearn Vision checkpoint and runs single-image inference.

    Args:
        checkpoint_path : Path to model.pth saved by train.py
        device          : torch.device; auto-detected if None
    """

    def __init__(self, checkpoint_path: str, device: torch.device | None = None):
        self.device = device or get_device()
        self._load(checkpoint_path)
        self.transform = _inference_transforms()

    def _load(self, path: str) -> None:
        """Load checkpoint and restore model weights + class names."""
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: '{path}'\n"
                "Run train.py first to generate checkpoints/model.pth"
            )

        # map_location ensures the checkpoint loads on the current device
        # even if it was saved on a different one (e.g. CUDA → CPU)
        ckpt = torch.load(path, map_location=self.device)

        self.class_names: list[str] = ckpt["class_names"]
        num_classes = ckpt["num_classes"]
        backbone    = ckpt["backbone"]

        # Rebuild the exact same architecture and load saved weights
        self.model = build_model(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False,   # weights come from the checkpoint, not ImageNet
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state"])

        # model.eval() is critical at inference — disables dropout and fixes BatchNorm
        self.model.eval()

        print(f"✅  Loaded '{backbone}' checkpoint")
        print(f"    Classes : {self.class_names}")
        print(f"    Device  : {self.device}")

    @torch.no_grad()
    def predict(self, image_path: str) -> tuple[str, float]:
        """
        Predict the class of a single image.

        Args:
            image_path : Path to any .jpg / .png image file

        Returns:
            (class_name, confidence)  e.g. ("cat", 0.974)
        """
        # Step 1 — Load image as RGB (convert removes alpha channel if PNG)
        image = Image.open(image_path).convert("RGB")

        # Step 2 — Preprocess: PIL Image → normalised tensor of shape (3, 224, 224)
        tensor = self.transform(image)

        # Step 3 — Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
        # Models always expect a batch — even for a single image
        tensor = tensor.unsqueeze(0).to(self.device)

        # Step 4 — Forward pass → raw logits, shape (1, num_classes)
        logits = self.model(tensor)

        # Step 5 — Convert logits to probabilities using Softmax
        # dim=1 means softmax across the class dimension
        probs = F.softmax(logits, dim=1)

        # Step 6 — Pick class with highest probability
        confidence, class_idx = probs.max(dim=1)

        class_name   = self.class_names[class_idx.item()]
        confidence   = confidence.item()

        return class_name, confidence

    def predict_all_probs(self, image_path: str) -> dict[str, float]:
        """Return probabilities for ALL classes (useful for bar charts in UI)."""
        image  = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0)
        return {cls: round(probs[i].item(), 4) for i, cls in enumerate(self.class_names)}
