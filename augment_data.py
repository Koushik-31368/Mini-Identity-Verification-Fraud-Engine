"""
augment_data.py
===============
Augment liveness training images to reach 100-120 images per class.

Applies the following augmentations:
  - Horizontal flip
  - Brightness / contrast jitter
  - Rotation ±15°
  - Zoom (center crop + resize back to original size)

Each original image produces multiple augmented variants until the
target count is reached. Augmented images are saved alongside originals
with an 'aug_' prefix.

USAGE:
    python augment_data.py
"""

import os
import random
from PIL import Image, ImageEnhance, ImageFilter

# ── Configuration ────────────────────────────────────────────────────
REAL_DIR = "data/real"
SPOOF_DIR = "data/spoof"
TARGET_MIN = 100
TARGET_MAX = 120
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def load_originals(directory: str) -> list[str]:
    """Return list of original (non-augmented) image paths in a directory."""
    supported = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    originals = []
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("aug_"):
            continue  # Skip previously augmented images
        if fname.lower().endswith(supported) and not fname.startswith("."):
            originals.append(os.path.join(directory, fname))
    return originals


def augment_flip(img: Image.Image) -> Image.Image:
    """Horizontal flip."""
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def augment_brightness_contrast(img: Image.Image) -> Image.Image:
    """Random brightness and contrast jitter."""
    brightness_factor = random.uniform(0.7, 1.3)
    contrast_factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img


def augment_rotation(img: Image.Image) -> Image.Image:
    """Random rotation ±15 degrees."""
    angle = random.uniform(-15, 15)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))


def augment_zoom(img: Image.Image) -> Image.Image:
    """Random center crop (zoom 1.1x-1.3x) then resize back."""
    w, h = img.size
    zoom_factor = random.uniform(1.1, 1.3)
    crop_w = int(w / zoom_factor)
    crop_h = int(h / zoom_factor)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), Image.BICUBIC)


def augment_combined(img: Image.Image) -> Image.Image:
    """Apply a random combination of 2-3 augmentations."""
    augmentations = [augment_flip, augment_brightness_contrast, augment_rotation, augment_zoom]
    # Pick 2-3 random augmentations
    n = random.choice([2, 3])
    chosen = random.sample(augmentations, n)
    for aug_fn in chosen:
        img = aug_fn(img)
    return img


# All single augmentation functions + combined
AUGMENTATION_PIPELINE = [
    ("flip", augment_flip),
    ("bright", augment_brightness_contrast),
    ("rot", augment_rotation),
    ("zoom", augment_zoom),
    ("combo", augment_combined),
]


def augment_directory(directory: str, target_min: int, target_max: int):
    """Augment images in a directory to reach the target count."""
    originals = load_originals(directory)
    current_count = len([
        f for f in os.listdir(directory)
        if not f.startswith(".") and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])
    
    print(f"\n  Directory: {directory}")
    print(f"  Original images: {len(originals)}")
    print(f"  Current total:   {current_count}")

    if current_count >= target_min:
        print(f"  Already have {current_count} images (target: {target_min}-{target_max}). Skipping.")
        return current_count

    target = random.randint(target_min, target_max)
    needed = target - current_count
    print(f"  Target: {target} images -> need {needed} augmented images")

    if len(originals) == 0:
        print("  ERROR: No original images found!")
        return current_count

    aug_count = 0
    aug_idx = 0

    while aug_count < needed:
        # Cycle through originals
        src_path = originals[aug_idx % len(originals)]
        src_name = os.path.splitext(os.path.basename(src_path))[0]

        # Pick an augmentation
        aug_name, aug_fn = AUGMENTATION_PIPELINE[aug_count % len(AUGMENTATION_PIPELINE)]

        try:
            img = Image.open(src_path).convert("RGB")
            augmented = aug_fn(img)
            
            out_name = f"aug_{aug_count:03d}_{aug_name}_{src_name}.jpg"
            out_path = os.path.join(directory, out_name)
            augmented.save(out_path, "JPEG", quality=90)
            aug_count += 1
        except Exception as e:
            print(f"  WARNING: Failed to augment {src_path}: {e}")
            aug_count += 1  # Skip this one

        aug_idx += 1

    final_count = len([
        f for f in os.listdir(directory)
        if not f.startswith(".") and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])
    print(f"  Done! Final count: {final_count} images ({aug_count} augmented)")
    return final_count


def main():
    print("=" * 60)
    print("  Image Augmentation for Liveness Training")
    print("=" * 60)

    real_count = augment_directory(REAL_DIR, TARGET_MIN, TARGET_MAX)
    spoof_count = augment_directory(SPOOF_DIR, TARGET_MIN, TARGET_MAX)

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"  Real:  {real_count} images")
    print(f"  Spoof: {spoof_count} images")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
