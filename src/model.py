"""
EverLearn Vision – Model Builder
==================================
Wraps pretrained torchvision backbones with a custom classifier head.
Supports: resnet18, resnet50, efficientnet_b0, mobilenet_v3_small

VIVA NOTE — What is Transfer Learning?
  Instead of training a model from random weights (which needs millions of images),
  we start from a model already trained on ImageNet (1.2M images, 1000 classes).
  The backbone has learnt to detect edges, textures, and shapes.
  We KEEP all that knowledge and only REPLACE the final classification layer
  to match our own number of classes.
  Result: we need far less data and train much faster.

VIVA NOTE — What is a Residual Connection?
  In a normal neural network, data flows like a chain:
      Input → Layer1 → Layer2 → Layer3 → Output

  In ResNet (Residual Network), a "skip connection" adds the original input
  directly to the output of a block:
      Input ──────────────────────────────┐
        ↓                                 │  (skip / identity connection)
      Conv → BN → ReLU → Conv → BN       │
        ↓                                 │
      (+) ←────────────────────────────── ┘
        ↓
      ReLU → next block

  Why does this help?
  - Solves the vanishing gradient problem (gradients can flow directly
    through the skip connection without shrinking to zero).
  - Lets the network learn the DIFFERENCE (residual) from the input,
    which is easier than learning the full transformation from scratch.
  - Enables very deep networks (ResNet-18 has 18 layers, ResNet-50 has 50).

Usage:
    from src.model import build_model, get_device
    device = get_device()
    model  = build_model(num_classes=2).to(device)
"""

import torch
import torch.nn as nn
from torchvision import models


SUPPORTED_BACKBONES = ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_small"]


def get_device() -> torch.device:
    """
    Auto-detect the best available compute device.

    VIVA NOTE — Device priority:
      1. CUDA  — NVIDIA GPU (fastest)
      2. MPS   — Apple Silicon GPU (Mac M1/M2/M3)
      3. CPU   — Fallback (slowest)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(
    num_classes: int,
    backbone: str = "resnet18",
    pretrained: bool = True,
) -> nn.Module:
    """
    Load a pretrained backbone and replace its final layer with a new
    Linear layer that outputs `num_classes` scores.

    Args:
        num_classes : Number of output classes (e.g., 2 for cat/dog)
        backbone    : Architecture name (see SUPPORTED_BACKBONES)
        pretrained  : If True, load ImageNet-pretrained weights

    Returns:
        nn.Module ready to be moved to a device and trained

    VIVA NOTE — Why only change the FINAL layer?
      The early layers of a pretrained ResNet detect generic features:
        Layer 1-3: edges, colours, gradients
        Layer 4-6: textures, patterns
        Layer 7+ : shapes, object parts
      These are UNIVERSAL — helpful for any image task.
      Only the LAST fully-connected (fc) layer is dataset-specific
      (ImageNet has 1000 classes; we need just 2).
      So we keep all the good feature extraction, swap the classifier.
    """
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"backbone must be one of {SUPPORTED_BACKBONES}, got '{backbone}'"
        )

    # "DEFAULT" loads the best available pretrained weights for the architecture.
    # None trains from random initialisation (worse without lots of data).
    weights_arg = "DEFAULT" if pretrained else None

    if backbone == "resnet18":
        # ResNet-18: 18 layers deep, ~11M parameters, fast and lightweight
        model = models.resnet18(weights=weights_arg)

        # model.fc is the original 512→1000 Linear layer (ImageNet 1000 classes).
        # model.fc.in_features = 512 (features going INTO the final layer).
        # We replace it with a new 512→num_classes Linear layer.
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "resnet50":
        # ResNet-50: 50 layers, ~25M parameters, more accurate but slower
        model = models.resnet50(weights=weights_arg)
        # in_features = 2048 for ResNet-50 (wider than ResNet-18)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "efficientnet_b0":
        # EfficientNet scales width, depth, and resolution together.
        # classifier is a Sequential; index [1] holds the final Linear layer.
        model = models.efficientnet_b0(weights=weights_arg)
        in_features = model.classifier[1].in_features  # 1280
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif backbone == "mobilenet_v3_small":
        # MobileNet uses depthwise separable convolutions — tiny and fast,
        # designed for mobile/edge devices. classifier[-1] is index [3].
        model = models.mobilenet_v3_small(weights=weights_arg)
        in_features = model.classifier[3].in_features  # 1024
        model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count total and trainable parameters in the model.

    VIVA NOTE — Trainable vs frozen parameters:
      When we load a pretrained model, ALL parameters are trainable by default.
      In fine-tuning you can freeze early layers (requires_grad=False) to
      speed up training and prevent overwriting good lower-level features.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
