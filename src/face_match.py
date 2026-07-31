"""
src/face_match.py
=================
Face detection + identity matching module.

WHAT IT DOES:
    1. Takes two images: an ID document photo and a live selfie
    2. Uses Mediapipe Face Detection to find and crop faces from both
    3. Uses DeepFace.verify() to compare the two face embeddings
    4. Returns a match/no-match verdict with a confidence score

WHY TWO TOOLS?
    - Mediapipe:  lightweight, fast face DETECTION (finding where faces are).
                  We use it to crop just the face region, removing background noise.
    - DeepFace:   high-accuracy face VERIFICATION (are these the same person?).
                  It uses deep neural network embeddings (like ArcFace or VGG-Face)
                  to compute a similarity score between two faces.

    In a real KYC system, the ID photo is often small and low-quality (scanned),
    while the selfie is higher resolution. Cropping faces first improves accuracy.

USAGE:
    from src.face_match import verify_faces
    result = verify_faces("id_photo.jpg", "selfie.jpg")
    print(result)
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace


def detect_and_crop_face(image_path: str, padding: float = 0.2) -> np.ndarray | None:
    """
    Detect the largest face in an image and return a cropped face region.

    Uses Mediapipe's Face Detection model (fast, works on CPU).

    Args:
        image_path: Path to the image file.
        padding:    Extra margin around the face crop (0.2 = 20% padding).
                    Some padding helps DeepFace get a better embedding because
                    it can see the hairline, chin, etc.

    Returns:
        Cropped face as a numpy array (BGR), or None if no face found.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[face_match] ERROR: Could not read image: {image_path}")
        return None

    h, w, _ = img.shape

    # Convert BGR -> RGB (Mediapipe expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize Mediapipe Face Detection
    # model_selection=1 uses the full-range model (better for varied distances)
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:

        results = face_detection.process(img_rgb)

        if not results.detections:
            print(f"[face_match] No face detected in: {image_path}")
            return None

        # Take the first (highest confidence) detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        # Convert relative coords (0-1) to pixel coords
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        fw = int(bbox.width * w)
        fh = int(bbox.height * h)

        # Add padding (but stay within image bounds)
        pad_x = int(fw * padding)
        pad_y = int(fh * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)

        face_crop = img[y1:y2, x1:x2]

        confidence = detection.score[0]
        print(f"[face_match] Face detected in {os.path.basename(image_path)} "
              f"(confidence: {confidence:.2f}, crop: {x2-x1}x{y2-y1}px)")

        return face_crop


def verify_faces(
    id_image_path: str,
    selfie_image_path: str,
    distance_metric: str = "cosine",
    match_threshold: float = 0.40,
) -> dict:
    """
    Compare two face images and determine if they belong to the same person.

    Args:
        id_image_path:      Path to the ID/document photo.
        selfie_image_path:  Path to the live selfie photo.
        distance_metric:    "cosine", "euclidean", or "euclidean_l2".
                            Cosine is most common for face embeddings.
        match_threshold:    Maximum distance to consider a match.
                            Lower = stricter. DeepFace's default for cosine
                            with VGG-Face is ~0.40.

    Returns:
        {
            "is_match":          bool,
            "distance":          float (lower = more similar),
            "threshold":         float (the cutoff used),
            "confidence":        float 0-1 (1 = perfect match),
            "model_used":        str (which DeepFace model was used),
            "id_face_detected":  bool,
            "selfie_face_detected": bool,
            "details":           str (human-readable summary),
        }
    """
    result = {
        "is_match": False,
        "distance": None,
        "threshold": match_threshold,
        "confidence": 0.0,
        "model_used": "VGG-Face",
        "id_face_detected": False,
        "selfie_face_detected": False,
        "details": "",
    }

    # ── Step 1: Detect faces using Mediapipe ─────────────────────────
    print(f"[face_match] Detecting faces...")
    id_face = detect_and_crop_face(id_image_path)
    selfie_face = detect_and_crop_face(selfie_image_path)

    result["id_face_detected"] = id_face is not None
    result["selfie_face_detected"] = selfie_face is not None

    if id_face is None or selfie_face is None:
        missing = []
        if id_face is None:
            missing.append("ID photo")
        if selfie_face is None:
            missing.append("selfie")
        result["details"] = f"Face not detected in: {', '.join(missing)}"
        return result

    # ── Step 2: Save cropped faces to temp files for DeepFace ────────
    # DeepFace.verify() expects file paths (not numpy arrays in some versions)
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "temp_faces")
    os.makedirs(temp_dir, exist_ok=True)

    id_crop_path = os.path.join(temp_dir, "id_face_crop.jpg")
    selfie_crop_path = os.path.join(temp_dir, "selfie_face_crop.jpg")
    cv2.imwrite(id_crop_path, id_face)
    cv2.imwrite(selfie_crop_path, selfie_face)

    # ── Step 3: Use DeepFace to compare the two faces ────────────────
    # DeepFace.verify() runs a face recognition model and returns:
    #   {"verified": bool, "distance": float, "threshold": float, ...}
    #
    # We use enforce_detection=False because we already cropped the faces
    # ourselves with Mediapipe. This avoids DeepFace failing if its own
    # face detector can't find a face in the tight crop.
    try:
        print(f"[face_match] Running DeepFace verification...")
        verification = DeepFace.verify(
            img1_path=id_crop_path,
            img2_path=selfie_crop_path,
            model_name="VGG-Face",
            distance_metric=distance_metric,
            enforce_detection=False,  # we already did detection
        )

        distance = verification["distance"]
        threshold = verification.get("threshold", match_threshold)

        # Convert distance to a confidence score (0-1, higher = more similar)
        # For cosine distance: confidence = 1 - distance (clamped to 0-1)
        confidence = max(0.0, min(1.0, 1.0 - distance))

        result["distance"] = round(distance, 4)
        result["threshold"] = threshold
        result["confidence"] = round(confidence, 4)
        result["is_match"] = verification.get("verified", distance <= threshold)
        result["model_used"] = verification.get("model", "VGG-Face")

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

    # ── Cleanup temp files ───────────────────────────────────────────
    for f in [id_crop_path, selfie_crop_path]:
        if os.path.exists(f):
            os.remove(f)

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
    # Users should provide two images: an ID photo and a selfie
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
        print(f"\nYou can use any two photos of faces for testing.")
        sys.exit(0)

    result = verify_faces(id_photo, selfie)

    print("\n" + "-" * 55)
    display = {k: v for k, v in result.items()}
    print(json.dumps(display, indent=2, ensure_ascii=False))
    print("=" * 55)
