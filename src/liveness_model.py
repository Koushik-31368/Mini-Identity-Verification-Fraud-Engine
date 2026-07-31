"""
src/liveness_model.py
=====================
Liveness detection using a fine-tuned MobileNetV2 CNN.

WHAT IT DOES:
    Determines whether a face image is "real" (live person in front of the camera)
    or "spoof" (printed photo, screen replay, mask, etc.).

WHY MOBILENETV2 + TRANSFER LEARNING?
    ──────────────────────────────────
    Training a CNN from scratch for image classification requires:
    - Millions of images
    - Days/weeks of training on GPUs
    - Careful architecture design

    TRANSFER LEARNING shortcuts this by:
    1. Starting with a model (MobileNetV2) that was already trained on
       ImageNet (1.2M images, 1000 classes). Its convolutional layers have
       already learned to detect edges, textures, shapes, faces, etc.

    2. FREEZING the base layers — we keep those learned features locked.
       They're generic enough to be useful for our liveness task too.
       (Why retrain edge/texture detectors when they already work?)

    3. Adding a small NEW "classification head" on top — just 2-3 dense
       layers that learn to map those generic features to our specific
       task: "real" vs "spoof".

    This way, we only train ~1000 parameters instead of ~3 million,
    and we can get good accuracy with just 100-150 images per class
    instead of the millions we'd otherwise need.

USAGE:
    # Inference (after training):
    from src.liveness_model import predict_liveness
    result = predict_liveness("selfie.jpg")
    print(result)  # {"label": "real", "confidence": 0.92}
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# ── Image preprocessing pipeline ────────────────────────────────────
# MobileNetV2 was trained on 224x224 images normalized with ImageNet stats.
# We must apply the SAME preprocessing to our images so the features match.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),          # Resize to expected input size
    transforms.ToTensor(),                   # Convert PIL -> tensor (0-1)
    transforms.Normalize(                    # Normalize to ImageNet distribution
        mean=[0.485, 0.456, 0.406],          # These are the ImageNet channel means
        std=[0.229, 0.224, 0.225],           # and standard deviations
    ),
])


def build_model(num_classes: int = 2, freeze_base: bool = True) -> nn.Module:
    """
    Build a MobileNetV2 model with a custom classification head.

    Architecture:
        [MobileNetV2 base (frozen)] -> [Dropout] -> [Dense 512] -> [ReLU]
                                    -> [Dropout] -> [Dense 2]   -> [Softmax]

    Args:
        num_classes: Number of output classes (2: real vs spoof).
        freeze_base: If True, freeze the pretrained convolutional layers.

    Returns:
        PyTorch model ready for training or inference.
    """
    # Load MobileNetV2 with pretrained ImageNet weights
    # These weights contain learned feature detectors for edges, textures,
    # shapes, and high-level patterns (trained on 1.2M images).
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # ── FREEZE the base convolutional layers ─────────────────────────
    # WHY: These layers already know how to extract visual features.
    # We don't want to overwrite that knowledge with our small dataset.
    # "Freezing" means: during training, gradients won't flow back to
    # these layers, so their weights stay fixed.
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False
        print("[liveness] Base layers FROZEN (pretrained features preserved)")

    # ── REPLACE the classification head ──────────────────────────────
    # MobileNetV2's original head was trained for 1000 ImageNet classes.
    # We replace it with a smaller head for our 2-class problem.
    #
    # The "classifier" in MobileNetV2 takes a 1280-dimensional feature
    # vector (output of the last conv layer after global avg pooling)
    # and maps it to class predictions.
    in_features = model.classifier[1].in_features  # = 1280

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),            # Prevent overfitting on small dataset
        nn.Linear(in_features, 512),  # Compress 1280 features -> 512
        nn.ReLU(),                    # Non-linearity
        nn.Dropout(p=0.2),            # More dropout
        nn.Linear(512, num_classes),  # Final classification: 512 -> 2
    )
    # NOTE: We do NOT add Softmax here because PyTorch's CrossEntropyLoss
    # expects raw logits (it applies Softmax internally for numerical stability).

    print(f"[liveness] New classification head: 1280 -> 512 -> {num_classes}")

    # Count trainable vs frozen parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[liveness] Parameters: {trainable:,} trainable / {total:,} total "
          f"({trainable/total:.1%} trainable)")

    return model


def predict_liveness(
    image_path: str,
    model_path: str = "models/liveness_model.pt",
) -> dict:
    """
    Run liveness inference on a single image.

    Args:
        image_path: Path to the face/selfie image.
        model_path: Path to the trained model file.

    Returns:
        {
            "label":      "real" or "spoof",
            "confidence": float (0.0 to 1.0),
            "details":    str,
        }
    """
    result = {
        "label": "unknown",
        "confidence": 0.0,
        "details": "",
    }

    # Check model exists
    if not os.path.exists(model_path):
        result["details"] = (
            f"Liveness model not found at: {model_path}. "
            "Run train_liveness.py first."
        )
        return result

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2, freeze_base=False)  # freeze doesn't matter for inference
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Set to inference mode (disables dropout)

    # Load and preprocess the image
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        result["details"] = f"Could not load image: {e}"
        return result

    img_tensor = TRANSFORM(img).unsqueeze(0).to(device)  # Add batch dimension

    # Run inference
    with torch.no_grad():  # No gradient computation needed for inference
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)  # Convert logits -> probabilities
        confidence, predicted_class = torch.max(probabilities, dim=1)

    # Class mapping (must match training order)
    class_names = ["real", "spoof"]
    label = class_names[predicted_class.item()]

    result["label"] = label
    result["confidence"] = round(confidence.item(), 4)
    result["details"] = f"Prediction: {label} ({confidence.item():.1%} confidence)"

    print(f"[liveness] {result['details']}")
    return result


# -----------------------------------------------------------------------
# CLI test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.liveness_model <image_path>")
        print("\nOr test model architecture:")
        model = build_model()
        print("\nModel built successfully. Train it with: python train_liveness.py")
        sys.exit(0)

    result = predict_liveness(sys.argv[1])
    print(result)
