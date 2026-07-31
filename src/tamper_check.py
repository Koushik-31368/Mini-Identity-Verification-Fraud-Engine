"""
src/tamper_check.py
===================
Document tamper detection using OCR bounding-box analysis.

WHAT IT DOES:
    Given the OCR bounding boxes from document_parser.py, this module
    checks whether the text on the document looks visually consistent.

    Specifically, it looks at:
    1. Font HEIGHT consistency  - all text lines should be roughly the
       same height if they're the same font/size.
    2. SPACING consistency      - gaps between consecutive lines should
       be roughly even in a legitimate document.

    If one text block has a wildly different height or spacing, it may
    have been pasted/edited into the document (e.g. someone photoshopped
    a different name onto a certificate).

WHY THIS WORKS (at a basic level):
    Real documents use consistent fonts and layouts. When someone edits
    a document image, they often:
    - Paste text at a slightly different scale -> height outlier
    - Position it imprecisely -> spacing outlier
    This is a heuristic, not forensic-grade, but it catches obvious edits.

USAGE:
    from src.tamper_check import check_tamper
    result = check_tamper(ocr_blocks)  # ocr_blocks from document_parser
    print(result)
"""

import numpy as np


def _get_line_heights(ocr_blocks: list[dict]) -> list[dict]:
    """
    Compute the pixel height of each OCR text block from its bounding box.

    Each OCR block has a bbox with 4 corners:
        [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        Typically: top-left, top-right, bottom-right, bottom-left

    The height of the text = average of left-side height and right-side height.
    This handles slightly rotated bounding boxes.
    """
    measurements = []
    for block in ocr_blocks:
        bbox = block["bbox"]
        # bbox corners: [top-left, top-right, bottom-right, bottom-left]
        # Height on left side:  distance from top-left[1] to bottom-left[1]
        left_height = abs(bbox[3][1] - bbox[0][1])
        # Height on right side: distance from top-right[1] to bottom-right[1]
        right_height = abs(bbox[2][1] - bbox[1][1])
        avg_height = (left_height + right_height) / 2

        measurements.append({
            "text": block["text"],
            "height": avg_height,
            "y_center": (bbox[0][1] + bbox[2][1]) / 2,  # vertical center
        })

    # Sort by vertical position (top to bottom) for spacing analysis
    measurements.sort(key=lambda m: m["y_center"])
    return measurements


def _get_line_spacings(measurements: list[dict]) -> list[float]:
    """
    Compute the vertical spacing (gap) between consecutive text lines.

    Spacing = distance between the y_center of line N and line N+1.
    In a well-formatted document, these gaps should be fairly consistent.
    """
    spacings = []
    for i in range(1, len(measurements)):
        gap = measurements[i]["y_center"] - measurements[i - 1]["y_center"]
        spacings.append(gap)
    return spacings


def _find_outliers(values: list[float], threshold_zscore: float = 2.0) -> list[int]:
    """
    Find indices of values that are statistical outliers using z-score.

    Z-score = how many standard deviations a value is from the mean.
    If |z-score| > threshold, it's flagged as suspicious.

    WHY z-score = 2.0?
        - Within 2 standard deviations: ~95% of normal data
        - Beyond 2 SD: only ~5% — suspicious enough to flag
        - This is a common threshold in anomaly detection
    """
    if len(values) < 3:
        # Not enough data points to meaningfully detect outliers
        return []

    arr = np.array(values)
    mean = arr.mean()
    std = arr.std()

    if std == 0:
        # All values identical = perfectly consistent = no outliers
        return []

    outliers = []
    for i, val in enumerate(values):
        z = abs(val - mean) / std
        if z > threshold_zscore:
            outliers.append(i)
    return outliers


def check_tamper(ocr_blocks: list[dict], zscore_threshold: float = 2.0) -> dict:
    """
    Main entry point: analyze OCR bounding boxes for signs of tampering.

    Args:
        ocr_blocks:       List of OCR detections from document_parser.extract_text()
                          Each must have "text" and "bbox" keys.
        zscore_threshold: How many std-devs a value must deviate to be flagged.
                          Lower = more sensitive, higher = fewer false positives.

    Returns:
        {
            "tamper_score":   float 0.0 (clean) to 1.0 (heavily tampered),
            "is_suspicious":  bool,
            "flagged_lines":  list of dicts with details on suspicious text blocks,
            "height_stats":   {"mean": float, "std": float},
            "spacing_stats":  {"mean": float, "std": float},
            "details":        human-readable summary string,
        }
    """
    result = {
        "tamper_score": 0.0,
        "is_suspicious": False,
        "flagged_lines": [],
        "height_stats": {},
        "spacing_stats": {},
        "details": "",
    }

    # Need at least 3 text blocks to do meaningful analysis
    if len(ocr_blocks) < 3:
        result["details"] = "Too few text blocks to analyze (need >= 3)."
        return result

    # ── Step 1: Analyze font heights ─────────────────────────────────
    measurements = _get_line_heights(ocr_blocks)
    heights = [m["height"] for m in measurements]
    height_mean = np.mean(heights)
    height_std = np.std(heights)

    result["height_stats"] = {
        "mean": round(float(height_mean), 2),
        "std": round(float(height_std), 2),
    }

    # Find height outliers
    height_outliers = _find_outliers(heights, zscore_threshold)

    # ── Step 2: Analyze line spacings ────────────────────────────────
    spacings = _get_line_spacings(measurements)
    if spacings:
        spacing_mean = np.mean(spacings)
        spacing_std = np.std(spacings)
        result["spacing_stats"] = {
            "mean": round(float(spacing_mean), 2),
            "std": round(float(spacing_std), 2),
        }
        spacing_outliers = _find_outliers(spacings, zscore_threshold)
    else:
        spacing_outliers = []

    # ── Step 3: Build list of flagged lines ──────────────────────────
    flagged = []

    for idx in height_outliers:
        m = measurements[idx]
        z = abs(m["height"] - height_mean) / height_std if height_std > 0 else 0
        flagged.append({
            "text": m["text"],
            "reason": "height_outlier",
            "value": round(m["height"], 2),
            "mean": round(float(height_mean), 2),
            "z_score": round(z, 2),
        })

    for idx in spacing_outliers:
        # The spacing outlier is the gap BEFORE line (idx+1)
        if idx + 1 < len(measurements):
            m = measurements[idx + 1]
            z = abs(spacings[idx] - spacing_mean) / spacing_std if spacing_std > 0 else 0
            flagged.append({
                "text": m["text"],
                "reason": "spacing_outlier",
                "value": round(spacings[idx], 2),
                "mean": round(float(spacing_mean), 2),
                "z_score": round(z, 2),
            })

    result["flagged_lines"] = flagged

    # ── Step 4: Compute overall tamper score ─────────────────────────
    # Score = proportion of checks that flagged something
    # Two checks: height + spacing, each weighted equally
    total_checks = len(heights) + len(spacings)
    total_flags = len(height_outliers) + len(spacing_outliers)

    if total_checks > 0:
        result["tamper_score"] = round(total_flags / total_checks, 4)
    result["is_suspicious"] = result["tamper_score"] > 0.1  # >10% flagged

    # ── Step 5: Human-readable summary ───────────────────────────────
    if not flagged:
        result["details"] = (
            f"No anomalies detected. "
            f"Analyzed {len(heights)} text blocks with consistent "
            f"font height (mean={height_mean:.1f}px, std={height_std:.1f}px)."
        )
    else:
        lines = [f"SUSPICIOUS: {len(flagged)} anomalies detected:"]
        for f in flagged:
            lines.append(
                f"  - '{f['text']}' flagged for {f['reason']} "
                f"(value={f['value']}, mean={f['mean']}, z={f['z_score']})"
            )
        result["details"] = "\n".join(lines)

    return result


# -----------------------------------------------------------------------
# CLI test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys
    import os

    # We need ocr_blocks from document_parser, so import and run it
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.document_parser import parse_document

    test_image = "data/sample_docs/sample_certificate.png"
    if not os.path.exists(test_image):
        print(f"ERROR: Image not found: {test_image}")
        sys.exit(1)

    # Parse the document to get OCR blocks
    doc_result = parse_document(test_image)
    ocr_blocks = doc_result["ocr_blocks"]

    # Run tamper check
    tamper_result = check_tamper(ocr_blocks)

    print("\n" + "=" * 55)
    print("  Tamper Check Results")
    print("=" * 55)

    # Print without ocr_blocks for readability
    display = {k: v for k, v in tamper_result.items()}
    print(json.dumps(display, indent=2, ensure_ascii=False))
    print("=" * 55)
