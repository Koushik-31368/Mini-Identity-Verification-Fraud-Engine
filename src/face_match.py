"""
src/face_match.py
=================
Face detection + identity matching module.

WHAT IT DOES:
    1. Takes two images: an ID document photo and a live selfie
    2. Uses DeepFace.verify() to compare the two faces
    3. Returns a match/no-match verdict with a confidence score

WHY DEEPFACE?
    DeepFace is a high-level library that bundles both face DETECTION
    and face VERIFICATION in one call. Under the hood it:
    - Detects faces using RetinaFace (or other backends)
    - Aligns/crops the faces automatically
    - Computes 128-d or 2622-d face embeddings
    - Calculates distance (cosine, euclidean, etc.)
    - Compares against a threshold to decide match/no-match

    This means we get detection + matching in a single API call,
    which is simpler and less error-prone than chaining two libraries.

USAGE:
    from src.face_match import verify_faces
    result = verify_faces("id_photo.jpg", "selfie.jpg")
    print(result)
"""

import os
import sys
import cv2
import numpy as np

# Fix Windows console encoding (DeepFace logs contain emoji that crash cp1252)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from deepface import DeepFace


def detect_face(image_path: str) -> bool:
    """
    Check if a face can be detected in the given image.

    Uses DeepFace's extract_faces() to find faces. This is a quick
    pre-check before running the full verification.

    Returns True if at least one face is found, False otherwise.
    """
    try:
        # Using 'skip' backend: treats the whole image as a face.
        # This works well for selfie-style photos where the face fills
        # the frame. OpenCV 5.0 broke the 'ssd' and 'opencv' backends
        # (removed cv2.dnn.readNetFromCaffe), so 'skip' is the most
        # reliable cross-version option.
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="skip",
            enforce_detection=False,
        )
        found = len(faces) > 0
        print(f"[face_match] Face {'detected' if found else 'NOT detected'} "
              f"in {os.path.basename(image_path)}")
        return found
    except Exception as e:
        print(f"[face_match] Face check issue in {os.path.basename(image_path)}: {e}")
        return False


def verify_faces(
    id_image_path: str,
    selfie_image_path: str,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
) -> dict:
    """
    Compare two face images and determine if they belong to the same person.

    Args:
        id_image_path:      Path to the ID/document photo.
        selfie_image_path:  Path to the live selfie photo.
        model_name:         DeepFace model: "VGG-Face", "Facenet", "ArcFace", etc.
        distance_metric:    "cosine", "euclidean", or "euclidean_l2".

    Returns:
        {
            "is_match":             bool,
            "distance":             float (lower = more similar),
            "threshold":            float (the cutoff used),
            "confidence":           float 0-1 (1 = perfect match),
            "model_used":           str,
            "id_face_detected":     bool,
            "selfie_face_detected": bool,
            "details":              str (human-readable summary),
        }
    """
    result = {
        "is_match": False,
        "distance": None,
        "threshold": None,
        "confidence": 0.0,
        "model_used": model_name,
        "id_face_detected": False,
        "selfie_face_detected": False,
        "details": "",
    }

    # ── Step 1: Quick face detection check ───────────────────────────
    print(f"[face_match] Checking for faces...")
    result["id_face_detected"] = detect_face(id_image_path)
    result["selfie_face_detected"] = detect_face(selfie_image_path)

    if not result["id_face_detected"] or not result["selfie_face_detected"]:
        missing = []
        if not result["id_face_detected"]:
            missing.append("ID photo")
        if not result["selfie_face_detected"]:
            missing.append("selfie")
        result["details"] = f"Face not detected in: {', '.join(missing)}"
        return result

    # ── Step 2: Use DeepFace to verify (compare) the two faces ───────
    #
    # DeepFace.verify() does everything in one call:
    #   1. Detects + aligns faces in both images
    #   2. Computes face embeddings using the chosen model
    #   3. Calculates the distance between embeddings
    #   4. Compares distance to a model-specific threshold
    #
    # Returns: {"verified": bool, "distance": float, "threshold": float, ...}
    try:
        print(f"[face_match] Running DeepFace verification ({model_name})...")
        verification = DeepFace.verify(
            img1_path=id_image_path,
            img2_path=selfie_image_path,
            model_name=model_name,
            distance_metric=distance_metric,
            detector_backend="skip",    # Skip detection (selfie photos)
            enforce_detection=False,      # Don't crash if detection fails
        )

        distance = verification["distance"]
        threshold = verification.get("threshold", 0.40)

        # Convert distance to a confidence score (0-1, higher = more similar)
        # For cosine distance: confidence = 1 - distance (clamped to 0-1)
        confidence = max(0.0, min(1.0, 1.0 - distance))

        result["distance"] = round(distance, 4)
        result["threshold"] = round(threshold, 4)
        result["confidence"] = round(confidence, 4)
        result["is_match"] = verification.get("verified", distance <= threshold)

        if result["is_match"]:
            result["details"] = (
                f"MATCH: Faces match with {confidence:.1%} confidence "
                f"(distance={distance:.4f}, threshold={threshold:.4f})"
            )
        else:
            result["details"] = (
                f"NO MATCH: Faces do not match "
                f"(distance={distance:.4f} > threshold={threshold:.4f})"
            )

    except Exception as e:
        result["details"] = f"DeepFace verification error: {str(e)}"
        print(f"[face_match] ERROR: {e}")

    return result


# -----------------------------------------------------------------------
# CLI test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys

    print("\n" + "=" * 55)
    print("  Face Match Module - Test")
    print("=" * 55)

    # Check for test images
    id_photo = "data/sample_docs/test_id_photo.jpg"
    selfie = "data/sample_docs/test_selfie.jpg"

    if len(sys.argv) >= 3:
        id_photo = sys.argv[1]
        selfie = sys.argv[2]

    if not os.path.exists(id_photo) or not os.path.exists(selfie):
        print(f"\nTo test face matching, provide two face images:")
        print(f"  python -m src.face_match <id_photo> <selfie>")
        print(f"\nOr place test images at:")
        print(f"  {id_photo}")
        print(f"  {selfie}")
        sys.exit(0)

    result = verify_faces(id_photo, selfie)

    print("\n" + "-" * 55)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 55)
