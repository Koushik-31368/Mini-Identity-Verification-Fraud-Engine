"""Quick test: one inference per class for liveness model."""
import os
from src.liveness_model import predict_liveness

# Pick one original (non-augmented) image from each class
real_images = [f for f in sorted(os.listdir("data/real"))
               if not f.startswith(".") and not f.startswith("aug_")
               and f.lower().endswith((".jpg", ".jpeg", ".png"))]
spoof_images = [f for f in sorted(os.listdir("data/spoof"))
                if not f.startswith(".") and not f.startswith("aug_")
                and f.lower().endswith((".jpg", ".jpeg", ".png"))]

print("=== Test: Real face (should predict 'real') ===")
real_path = os.path.join("data/real", real_images[0])
print("Image:", real_images[0])
r = predict_liveness(real_path)
print("Label:", r["label"])
print("Confidence:", round(r["confidence"], 4))
print("Details:", r["details"])

print()
print("=== Test: Spoof face (should predict 'spoof') ===")
spoof_path = os.path.join("data/spoof", spoof_images[0])
print("Image:", spoof_images[0])
r = predict_liveness(spoof_path)
print("Label:", r["label"])
print("Confidence:", round(r["confidence"], 4))
print("Details:", r["details"])
