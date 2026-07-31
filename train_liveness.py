"""
train_liveness.py
=================
Training script for the liveness detection model.

PREREQUISITES:
    Populate these folders with ~100-150 images each:
        data/real/    <- photos of REAL faces (live selfies)
        data/spoof/   <- photos of SPOOFED faces (printed photos, screen replays)

    Images can be any size/format (jpg, png) — the pipeline resizes them to 224x224.

HOW TRANSFER LEARNING TRAINING WORKS:
    ─────────────────────────────────────
    1. Load MobileNetV2 with ImageNet weights (knows generic visual features)
    2. Freeze the base layers (keep those features locked)
    3. Replace the classification head (1000 classes -> 2 classes)
    4. Train ONLY the new head for a few epochs
    5. The head learns: "these features = real face" vs "these features = spoof"

    Because we're training so few parameters (~660K out of ~3.5M), we only
    need a small dataset and a few epochs. Overfitting risk is managed with
    dropout layers and a validation split.

USAGE:
    python train_liveness.py

    Output: models/liveness_model.pt
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Import our model builder
from src.liveness_model import build_model


# ── Configuration ────────────────────────────────────────────────────
DATA_DIR = "data/liveness"               # Contains real/ and spoof/ subfolders
MODEL_SAVE_PATH = "models/liveness_model.pt"
BATCH_SIZE = 16                          # Images per training step
NUM_EPOCHS = 10                          # Full passes through the training data
LEARNING_RATE = 0.001                    # How fast the optimizer updates weights
VAL_SPLIT = 0.2                          # 20% of data reserved for validation


# ── Data augmentation & preprocessing ────────────────────────────────
# Training: we add random augmentations to artificially increase data diversity.
# This helps prevent overfitting when we have a small dataset.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),    # Mirror images 50% of the time
    transforms.RandomRotation(10),        # Slight rotation +-10 degrees
    transforms.ColorJitter(               # Random color variations
        brightness=0.2, contrast=0.2, saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Validation: NO augmentation — we want to evaluate on clean images
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def train():
    """Run the full training pipeline."""

    print("=" * 60)
    print("  Liveness Model Training")
    print("=" * 60)

    # ── Step 1: Load dataset ─────────────────────────────────────────
    # ImageFolder automatically maps subfolder names to class labels:
    #   data/real/   -> class 0 ("real")
    #   data/spoof/  -> class 1 ("spoof")
    # This is why the folder structure matters!
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

    # Check that we have the expected classes
    print(f"\nClasses found: {full_dataset.classes}")
    print(f"Class mapping: {full_dataset.class_to_idx}")
    print(f"Total images:  {len(full_dataset)}")

    if len(full_dataset) < 10:
        print("\nERROR: Too few images! Need at least 10.")
        print("Populate data/real/ and data/spoof/ with face images first.")
        return

    # ── Step 2: Split into training and validation sets ──────────────
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    # Apply validation transform to val set (no augmentation)
    # Note: random_split shares the parent dataset, so we create a
    # wrapper that applies the correct transform
    val_dataset.dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)

    print(f"Train set: {train_size} images")
    print(f"Val set:   {val_size} images")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Step 3: Build model ──────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = build_model(num_classes=2, freeze_base=True)
    model = model.to(device)

    # ── Step 4: Define loss function and optimizer ────────────────────
    # CrossEntropyLoss: standard loss for classification tasks.
    # It combines Softmax + Negative Log-Likelihood in one step.
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer: adaptive learning rate, works well with small datasets.
    # We only optimize parameters that require gradients (the unfrozen head).
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # ── Step 5: Training loop ────────────────────────────────────────
    print(f"\nTraining for {NUM_EPOCHS} epochs...\n")
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Train Acc':<12} {'Val Loss':<14} {'Val Acc':<12}")
    print("-" * 60)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        # ── Training phase ───────────────────────────────────────────
        model.train()  # Enable dropout, batch norm in training mode
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()         # Reset gradients from previous step
            outputs = model(images)       # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()               # Backward pass (compute gradients)
            optimizer.step()              # Update weights

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validation phase ─────────────────────────────────────────
        model.eval()  # Disable dropout, batch norm in eval mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # No gradient computation for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total if val_total > 0 else 1
        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"{epoch+1:<8} {train_loss:<14.4f} {train_acc:<12.4f} {val_loss:<14.4f} {val_acc:<12.4f}")

        # Save best model (by validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"         ^ New best model saved (val_acc={val_acc:.4f})")

    # ── Step 6: Final summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Model saved to: {MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()
