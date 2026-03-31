"""
EverLearn Vision – Model Builder
==================================
Wraps torchvision pretrained models with a custom classifier head.
Supports: resnet18, resnet50, efficientnet_b0, mobilenet_v3_small

Usage:
    from src.model import build_model
    model = build_model(num_classes=2, backbone="resnet18", pretrained=True)
"""

import torch
import torch.nn as nn
from torchvision import models


SUPPORTED_BACKBONES = ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_small"]


def build_model(
    num_classes: int,
    backbone: str = "resnet18",
    pretrained: bool = True,
) -> nn.Module:
    """
    Load a pretrained backbone and swap the classifier head
    to match `num_classes`.
    """
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"backbone must be one of {SUPPORTED_BACKBONES}, got '{backbone}'")

    weights_arg = "DEFAULT" if pretrained else None

    if backbone == "resnet18":
        model = models.resnet18(weights=weights_arg)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "resnet50":
        model = models.resnet50(weights=weights_arg)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights_arg)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    return model
