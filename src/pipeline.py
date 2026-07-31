"""
src/pipeline.py
===============
Unified KYC verification pipeline.

WHAT IT DOES:
    Combines ALL modules into a single verification function:
    1. Document parsing    (extract name, date, ID from uploaded document)
    2. Tamper check        (is the document visually consistent?)
    3. Face matching       (does the ID photo match the selfie?)
    4. Liveness detection  (is the selfie from a live person, not a printout?)
    5. Fraud scoring       (does the transaction data look suspicious?)

    Each module produces a score, and this pipeline combines them into
    a final weighted verdict:
        - VERIFIED           = all checks pass
        - FLAGGED FOR REVIEW = some checks are borderline
        - REJECTED           = critical check(s) failed

WEIGHTING LOGIC:
    ────────────────
    Not all checks are equally important. In real KYC:
    - Face match is CRITICAL — wrong face = wrong person (weight: 0.35)
    - Liveness is CRITICAL   — spoof = not a real person  (weight: 0.25)
    - Tamper check is HIGH   — forged document = fraud     (weight: 0.20)
    - Document parsing is MEDIUM — missing fields = incomplete (weight: 0.10)
    - Fraud scoring is MEDIUM — behavioral signal, not definitive (weight: 0.10)

    Weights sum to 1.0. The final score is a weighted average of individual
    scores normalized to 0-1 (1 = fully verified, 0 = definitely reject).

USAGE:
    from src.pipeline import run_pipeline
    result = run_pipeline(
        document_image="path/to/document.jpg",
        selfie_image="path/to/selfie.jpg",
        transaction_data={"V1": -1.3, ..., "Amount": 149.62},  # optional
    )
    print(result["verdict"])  # "VERIFIED" / "FLAGGED FOR REVIEW" / "REJECTED"
"""

from src.document_parser import parse_document
from src.tamper_check import check_tamper
from src.face_match import verify_faces
from src.liveness_model import predict_liveness
from src.fraud_tabular import predict_fraud


# ── Verdict thresholds ───────────────────────────────────────────────
# These define the cutoffs for the final weighted score.
# Above 0.7 = verified, 0.4-0.7 = review, below 0.4 = rejected.
VERIFIED_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.40


def run_pipeline(
    document_image: str,
    selfie_image: str,
    transaction_data: dict | None = None,
) -> dict:
    """
    Run the full KYC verification pipeline.

    Args:
        document_image:   Path to the uploaded document/certificate image.
        selfie_image:     Path to the selfie/live photo.
        transaction_data: Optional dict of transaction features for fraud check.
                         If None, fraud check is skipped and its weight is
                         redistributed to other modules.

    Returns:
        {
            "verdict":          "VERIFIED" / "FLAGGED FOR REVIEW" / "REJECTED",
            "final_score":      float (0-1),
            "module_results": {
                "document_parsing": {...},
                "tamper_check":     {...},
                "face_match":       {...},
                "liveness":         {...},
                "fraud_check":      {...},   # None if no transaction_data
            },
            "module_scores": {
                "document_parsing": float,
                "tamper_check":     float,
                "face_match":       float,
                "liveness":         float,
                "fraud_check":      float,
            },
            "details":          str (human-readable summary),
        }
    """
    print("\n" + "=" * 60)
    print("  KYC VERIFICATION PIPELINE")
    print("=" * 60)

    results = {}
    scores = {}

    # ── Module 1: Document Parsing ───────────────────────────────────
    print("\n[1/5] Document Parsing...")
    try:
        doc_result = parse_document(document_image)
        results["document_parsing"] = {
            "name": doc_result.get("name"),
            "date": doc_result.get("date"),
            "id_number": doc_result.get("id_number"),
        }

        # Score: how many of the 3 key fields did we extract?
        fields_found = sum(1 for v in [
            doc_result.get("name"),
            doc_result.get("date"),
            doc_result.get("id_number"),
        ] if v is not None)
        scores["document_parsing"] = fields_found / 3.0
        print(f"       Score: {scores['document_parsing']:.2f} "
              f"({fields_found}/3 fields extracted)")
    except Exception as e:
        results["document_parsing"] = {"error": str(e)}
        scores["document_parsing"] = 0.0
        print(f"       ERROR: {e}")

    # ── Module 2: Tamper Check ───────────────────────────────────────
    print("\n[2/5] Tamper Check...")
    try:
        ocr_blocks = doc_result.get("ocr_blocks", [])
        tamper_result = check_tamper(ocr_blocks)
        results["tamper_check"] = {
            "tamper_score": tamper_result["tamper_score"],
            "is_suspicious": tamper_result["is_suspicious"],
            "flagged_lines": len(tamper_result["flagged_lines"]),
            "details": tamper_result["details"],
        }

        # Score: 1.0 = clean, 0.0 = heavily tampered
        scores["tamper_check"] = 1.0 - tamper_result["tamper_score"]
        print(f"       Score: {scores['tamper_check']:.2f} "
              f"(tamper_score={tamper_result['tamper_score']:.4f})")
    except Exception as e:
        results["tamper_check"] = {"error": str(e)}
        scores["tamper_check"] = 0.5  # Unknown = neutral
        print(f"       ERROR: {e}")

    # ── Module 3: Face Matching ──────────────────────────────────────
    print("\n[3/5] Face Matching...")
    try:
        face_result = verify_faces(document_image, selfie_image)
        results["face_match"] = {
            "is_match": face_result["is_match"],
            "confidence": face_result["confidence"],
            "distance": face_result["distance"],
            "details": face_result["details"],
        }

        # Score: confidence if matched, 0 if not matched
        if face_result["is_match"]:
            scores["face_match"] = face_result["confidence"]
        else:
            # No match = low score, but not necessarily zero
            # (could be a bad photo angle, not necessarily fraud)
            scores["face_match"] = max(0.0, face_result["confidence"] * 0.3)
        print(f"       Score: {scores['face_match']:.2f} "
              f"(match={face_result['is_match']})")
    except Exception as e:
        results["face_match"] = {"error": str(e)}
        scores["face_match"] = 0.0  # Can't verify = fail
        print(f"       ERROR: {e}")

    # ── Module 4: Liveness Detection ─────────────────────────────────
    print("\n[4/5] Liveness Detection...")
    try:
        liveness_result = predict_liveness(selfie_image)
        results["liveness"] = {
            "label": liveness_result["label"],
            "confidence": liveness_result["confidence"],
            "details": liveness_result["details"],
        }

        # Score: high confidence "real" = 1.0, high confidence "spoof" = 0.0
        if liveness_result["label"] == "real":
            scores["liveness"] = liveness_result["confidence"]
        elif liveness_result["label"] == "spoof":
            scores["liveness"] = 1.0 - liveness_result["confidence"]
        else:
            scores["liveness"] = 0.5  # Unknown / model not trained yet
        print(f"       Score: {scores['liveness']:.2f} "
              f"(label={liveness_result['label']})")
    except Exception as e:
        results["liveness"] = {"error": str(e)}
        scores["liveness"] = 0.5
        print(f"       ERROR: {e}")

    # ── Module 5: Fraud Check (optional) ─────────────────────────────
    print("\n[5/5] Fraud Check...")
    if transaction_data is not None:
        try:
            fraud_result = predict_fraud(transaction_data)
            results["fraud_check"] = {
                "is_fraud": fraud_result["is_fraud"],
                "probability": fraud_result["probability"],
                "details": fraud_result["details"],
            }

            # Score: 1.0 = not fraud, 0.0 = definitely fraud
            scores["fraud_check"] = 1.0 - fraud_result["probability"]
            print(f"       Score: {scores['fraud_check']:.2f} "
                  f"(fraud_prob={fraud_result['probability']:.4f})")
        except Exception as e:
            results["fraud_check"] = {"error": str(e)}
            scores["fraud_check"] = 0.5
            print(f"       ERROR: {e}")
    else:
        results["fraud_check"] = None
        scores["fraud_check"] = None
        print("       Skipped (no transaction data provided)")

    # ── Compute final weighted score ─────────────────────────────────
    # Weights reflect importance (see module docstring for rationale)
    weights = {
        "document_parsing": 0.10,
        "tamper_check":     0.20,
        "face_match":       0.35,
        "liveness":         0.25,
        "fraud_check":      0.10,
    }

    # If fraud check was skipped, redistribute its weight proportionally
    if scores["fraud_check"] is None:
        fraud_weight = weights.pop("fraud_check")
        remaining = sum(weights.values())
        for key in weights:
            weights[key] *= (1.0 + fraud_weight / remaining)
        scores["fraud_check"] = 0.0  # placeholder

    # Calculate weighted average
    final_score = sum(
        scores[module] * weight
        for module, weight in weights.items()
        if scores.get(module) is not None
    )
    final_score = round(final_score, 4)

    # ── Determine verdict ────────────────────────────────────────────
    if final_score >= VERIFIED_THRESHOLD:
        verdict = "VERIFIED"
    elif final_score >= REVIEW_THRESHOLD:
        verdict = "FLAGGED FOR REVIEW"
    else:
        verdict = "REJECTED"

    # ── Build summary ────────────────────────────────────────────────
    details_lines = [
        f"Final Score: {final_score:.2f} / 1.00",
        f"Verdict: {verdict}",
        "",
        "Module Breakdown:",
    ]
    for module, score in scores.items():
        w = weights.get(module, 0)
        if score is not None:
            details_lines.append(f"  {module:<20} score={score:.2f}  weight={w:.2f}")

    details = "\n".join(details_lines)

    print(f"\n{'='*60}")
    print(f"  VERDICT: {verdict}")
    print(f"  Score:   {final_score:.2f} / 1.00")
    print(f"{'='*60}")

    return {
        "verdict": verdict,
        "final_score": final_score,
        "module_results": results,
        "module_scores": scores,
        "weights_used": weights,
        "details": details,
    }


# -----------------------------------------------------------------------
# CLI test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python -m src.pipeline <document_image> <selfie_image>")
        sys.exit(0)

    result = run_pipeline(sys.argv[1], sys.argv[2])

    print("\n" + json.dumps(
        {k: v for k, v in result.items() if k != "module_results"},
        indent=2, default=str
    ))
