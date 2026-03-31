"""
EverLearn Vision – Model Demo
================================
Run this to inspect the ResNet-18 model: device, architecture,
parameter count, and a forward-pass shape check.

Usage:
    python model_demo.py
    python model_demo.py --backbone resnet50 --num_classes 5
"""

import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.model import build_model, get_device, count_parameters, SUPPORTED_BACKBONES


def main(backbone: str, num_classes: int) -> None:
    print("\n" + "=" * 55)
    print("  EverLearn Vision — Model Inspector")
    print("=" * 55)

    # ── 1. Device ─────────────────────────────────────────────
    device = get_device()
    print(f"\n🖥️   Device         : {device}")
    if str(device) == "mps":
        print("     (Apple Silicon GPU detected)")
    elif str(device) == "cuda":
        print(f"     ({torch.cuda.get_device_name(0)})")
    else:
        print("     (No GPU found — using CPU)")

    # ── 2. Build model ────────────────────────────────────────
    print(f"\n🏗️   Backbone       : {backbone}")
    print(f"    Num classes    : {num_classes}")
    print(f"    Pretrained     : Yes (ImageNet weights)")
    print("\n    Loading model...")

    model = build_model(num_classes=num_classes, backbone=backbone, pretrained=True)

    # Move entire model to best device
    # VIVA NOTE: .to(device) moves all model weights (tensors) to the device.
    # Your input batches must also be on the same device, or you'll get an error.
    model = model.to(device)
    print(f"    ✅  Model moved to {device}")

    # ── 3. Parameter count ────────────────────────────────────
    total, trainable = count_parameters(model)
    print(f"\n📊  Parameters:")
    print(f"    Total      : {total:,}")
    print(f"    Trainable  : {trainable:,}")
    print(f"    Frozen     : {total - trainable:,}")

    # ── 4. Forward-pass shape check ───────────────────────────
    # Create a dummy batch: 4 images, 3 channels (RGB), 224×224 pixels
    # VIVA NOTE: We use torch.randn to generate random noise in the same
    # shape as real image batches, just to confirm the model runs end-to-end.
    dummy_input = torch.randn(4, 3, 224, 224).to(device)

    model.eval()  # Disable dropout/batchnorm randomness for the check
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n🔁  Forward Pass (dummy input):")
    print(f"    Input  shape : {list(dummy_input.shape)}")
    print(f"    Output shape : {list(output.shape)}")
    print(f"    → Output is raw logits (one score per class, per image)")
    print(f"    → Pass through softmax to get probabilities")

    # ── 5. Classifier head inspection ─────────────────────────
    print(f"\n🔍  Final Classifier Layer:")
    if backbone in ("resnet18", "resnet50"):
        print(f"    {model.fc}")
        print(f"    (model.fc  — replaced the original 1000-class head)")
    elif backbone == "efficientnet_b0":
        print(f"    {model.classifier[1]}")
        print(f"    (model.classifier[1] — replaced for {num_classes} classes)")
    elif backbone == "mobilenet_v3_small":
        print(f"    {model.classifier[3]}")
        print(f"    (model.classifier[3] — replaced for {num_classes} classes)")

    print("\n✅  Model is ready for training!\n")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",    default="resnet18", choices=SUPPORTED_BACKBONES)
    parser.add_argument("--num_classes", type=int, default=2)
    args = parser.parse_args()
    main(args.backbone, args.num_classes)
