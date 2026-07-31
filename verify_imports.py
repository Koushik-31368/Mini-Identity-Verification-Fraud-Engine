"""
verify_imports.py
=================
Run this script ONCE after installing requirements.txt to make sure
every library needed for the project imports without errors.

Usage:
    python verify_imports.py

Each library is imported inside a try/except block so you can see
exactly which one fails (if any) rather than getting a single
cryptic ImportError.
"""

import sys
import os

# Force UTF-8 output on Windows so status symbols render correctly
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Helper: print OK / FAIL line
# ---------------------------------------------------------------------------
def check(label: str, import_fn):
    """Try calling import_fn(); print OK or FAIL with the error."""
    try:
        result = import_fn()
        version = getattr(result, "__version__", "n/a")
        print(f"  [OK]   {label:<30} version: {version}")
        return True
    except Exception as exc:
        print(f"  [FAIL] {label:<30} ERROR: {exc}")
        return False


print("\n" + "=" * 60)
print("  Mini KYC Engine -- Library Import Check")
print("=" * 60)

failures = 0

# -- Core numerics & ML ------------------------------------------------
print("\n[Core numerics & ML]")
failures += not check("numpy",        lambda: __import__("numpy"))
failures += not check("pandas",       lambda: __import__("pandas"))
failures += not check("scikit-learn", lambda: __import__("sklearn"))
failures += not check("scipy",        lambda: __import__("scipy"))

# -- Deep Learning ------------------------------------------------------
print("\n[Deep Learning]")
failures += not check("torch",        lambda: __import__("torch"))
failures += not check("torchvision",  lambda: __import__("torchvision"))

# -- Computer Vision ----------------------------------------------------
print("\n[Computer Vision]")
failures += not check("opencv-python",lambda: __import__("cv2"))
failures += not check("Pillow",       lambda: __import__("PIL"))

# -- OCR ----------------------------------------------------------------
print("\n[OCR]")
failures += not check("easyocr",      lambda: __import__("easyocr"))

# -- Face Recognition ---------------------------------------------------
print("\n[Face Recognition]")
failures += not check("deepface",     lambda: __import__("deepface"))

# -- Mediapipe ----------------------------------------------------------
print("\n[Pose / Landmark detection]")
failures += not check("mediapipe",    lambda: __import__("mediapipe"))

# -- Streamlit ----------------------------------------------------------
print("\n[Streamlit UI]")
failures += not check("streamlit",    lambda: __import__("streamlit"))

# -- Visualisation ------------------------------------------------------
print("\n[Visualisation]")
failures += not check("matplotlib",   lambda: __import__("matplotlib"))
failures += not check("seaborn",      lambda: __import__("seaborn"))

# -- Utilities ----------------------------------------------------------
print("\n[Utilities]")
failures += not check("tqdm",         lambda: __import__("tqdm"))
failures += not check("imutils",      lambda: __import__("imutils"))

# -- Final summary ------------------------------------------------------
print("\n" + "=" * 60)
if failures == 0:
    print("  ALL PASSED -- All libraries imported successfully!")
else:
    print(f"  {failures} library/libraries FAILED. Install them with:")
    print("       pip install -r requirements.txt")
print("=" * 60 + "\n")

sys.exit(failures)
