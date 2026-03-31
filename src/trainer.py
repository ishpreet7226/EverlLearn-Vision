"""
EverLearn Vision – Training & Validation Loops
================================================
Contains train_one_epoch() and evaluate() — the core of the pipeline.

VIVA NOTE — What is a Loss Function?
  A loss function measures HOW WRONG the model's predictions are.
  It outputs a single number (the "loss") — the goal of training is to
  MINIMISE this number.

  We use CrossEntropyLoss, which:
    1. Takes raw model outputs (logits), e.g. [2.3, -0.5] for 2 classes
    2. Applies Softmax internally to convert to probabilities [0.88, 0.12]
    3. Computes -log(probability of the correct class)
       → If the model is confident and correct: loss ≈ 0 (good)
       → If the model is wrong or uncertain:    loss is large (bad)

VIVA NOTE — What is an Optimizer?
  The optimizer is the algorithm that UPDATES model weights to reduce the loss.
  It uses gradients (computed by loss.backward()) to know which direction
  to adjust each weight.

  We use Adam (Adaptive Moment Estimation):
  - Keeps a running average of past gradients (momentum)
  - Adapts the learning rate per parameter automatically
  - More stable and faster than basic SGD for most tasks

  The update cycle every batch:
    1. optimizer.zero_grad()  → clear old gradients (they accumulate otherwise)
    2. output = model(input)  → forward pass: compute predictions
    3. loss = criterion(output, labels) → measure how wrong we are
    4. loss.backward()        → backprop: compute gradient for every weight
    5. optimizer.step()       → update weights using gradients

Usage:
    from src.trainer import train_one_epoch, evaluate
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full pass over the training set.

    Returns:
        avg_loss  — average loss across all training images
        accuracy  — fraction of correctly classified training images
    """
    # model.train() enables:
    #   - Dropout (randomly zeros neurons to prevent overfitting)
    #   - BatchNorm in training mode (uses batch statistics, not running stats)
    model.train()

    total_loss = 0.0
    correct    = 0
    total      = 0

    # tqdm wraps the loader and draws a live progress bar in the terminal
    for images, labels in tqdm(loader, desc="  Training ", leave=False):
        # Move data to same device as model (GPU/MPS/CPU)
        images = images.to(device)
        labels = labels.to(device)

        # ── Step 1: Clear gradients ────────────────────────────────────────────
        # Gradients accumulate by default — always zero them before a new batch
        optimizer.zero_grad()

        # ── Step 2: Forward pass ───────────────────────────────────────────────
        # Feed images through the model to get raw class scores (logits)
        # outputs shape: (batch_size, num_classes)
        outputs = model(images)

        # ── Step 3: Compute loss ───────────────────────────────────────────────
        # CrossEntropyLoss compares model logits against true labels
        # Returns a single scalar: the average loss over the batch
        loss = criterion(outputs, labels)

        # ── Step 4: Backward pass ──────────────────────────────────────────────
        # Compute gradient of the loss w.r.t. every trainable parameter
        # (uses the chain rule of calculus — "backpropagation")
        loss.backward()

        # ── Step 5: Update weights ─────────────────────────────────────────────
        # Adam uses the gradients to nudge each weight in the right direction
        optimizer.step()

        # ── Track metrics ──────────────────────────────────────────────────────
        # loss.item() returns the Python float value of the loss tensor
        # Multiply by batch size so we can compute the true average at the end
        total_loss += loss.item() * images.size(0)

        # outputs.max(1) returns (max_value, index_of_max)
        # The index of the highest logit is the predicted class
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()   # Disables gradient tracking — saves memory and speeds up eval
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model on validation/test set. No weight updates.

    Returns:
        avg_loss  — average loss across all validation images
        accuracy  — fraction correctly classified
    """
    # model.eval() disables:
    #   - Dropout (use all neurons for deterministic results)
    #   - BatchNorm training mode (use learnt running statistics)
    model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in tqdm(loader, desc="  Validating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # No backward pass — just forward + loss
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total
