"""
EverLearn Vision – Model Loader (FastAPI)
==========================================
Loads a trained checkpoint (model.pth) and returns a ready-to-use ModelBundle.

The checkpoint saved by train.py contains:
    - backbone      : architecture name (e.g. "resnet18")
    - num_classes   : number of output classes
    - class_names   : list of class label strings
    - model_state   : saved model weights (state_dict)

This module rebuilds the exact architecture, loads the saved weights,
switches to eval mode, and packages everything into a ModelBundle.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from src.model import build_model, get_device


@dataclass
class ModelBundle:
    """Container holding everything needed for inference."""
    model: nn.Module
    class_names: list[str]
    device: torch.device
    backbone: str


def load_model(checkpoint_path: str) -> ModelBundle:
    """
    Load a trained EverLearn Vision checkpoint and return a ModelBundle.

    Args:
        checkpoint_path: Path to the .pth checkpoint file saved by train.py.

    Returns:
        ModelBundle with the model in eval mode, class names, device, and backbone.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint is corrupt or incompatible.
    """
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'\n"
            "Run `python train.py` first to generate checkpoints/model.pth"
        )

    # Auto-detect the best available compute device
    device = get_device()

    # map_location ensures the checkpoint loads on the current device
    # even if it was saved on a different one (e.g. CUDA → MPS)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint '{checkpoint_path}': {e}"
        ) from e

    # Extract metadata baked into the checkpoint
    backbone = ckpt["backbone"]
    num_classes = ckpt["num_classes"]
    class_names = ckpt["class_names"]

    # Rebuild the exact same architecture (pretrained=False because
    # weights come from the checkpoint, not from ImageNet)
    model = build_model(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False,
    ).to(device)

    # Load the trained weights into the model
    model.load_state_dict(ckpt["model_state"])

    # Switch to eval mode — disables dropout and fixes BatchNorm statistics
    model.eval()

    print(f"✅  Model loaded: {backbone} ({num_classes} classes)")
    print(f"    Classes : {class_names}")
    print(f"    Device  : {device}")

    return ModelBundle(
        model=model,
        class_names=class_names,
        device=device,
        backbone=backbone,
    )
