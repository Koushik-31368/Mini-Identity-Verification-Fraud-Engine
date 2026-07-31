"""
src/document_parser.py
======================
OCR-based document parsing module.

WHAT IT DOES:
    1. Takes a document/certificate image as input
    2. Uses EasyOCR to extract all text + bounding boxes from the image
    3. Applies regex/heuristic rules to pull out structured fields:
       - Name, Date, ID/Certificate number
    4. Returns a clean dict with the extracted fields

WHY:
    In a KYC pipeline, the first step is extracting identity information
    from uploaded documents (passport, certificate, license). We need
    structured data (not just raw text) to compare against other signals
    like face match and fraud scores.

USAGE:
    from src.document_parser import parse_document
    result = parse_document("data/sample_docs/sample_certificate.png")
    print(result)
"""

import re
import easyocr

# -----------------------------------------------------------------------
# Initialize the EasyOCR reader ONCE (it loads a neural network model,
# so we don't want to do this on every function call).
# gpu=False because not every machine has a CUDA GPU.
# -----------------------------------------------------------------------
reader = easyocr.Reader(["en"], gpu=False)


def extract_text(image_path: str) -> list[dict]:
    """
    Run EasyOCR on the image and return a list of detected text blocks.

    Each block is a dict with:
        - "text":       the recognized string
        - "bbox":       [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] corner coords
        - "confidence": OCR confidence score (0.0 to 1.0)

    WHY return bounding boxes?
        -> Stage 3 (tamper check) needs them to analyze font consistency.
        -> We extract them once here and reuse them downstream.
    """
    # detail=1 returns (bbox, text, confidence) triples
    raw_results = reader.readtext(image_path, detail=1)

    blocks = []
    for bbox, text, confidence in raw_results:
        blocks.append({
            "text": text.strip(),
            "bbox": bbox,
            "confidence": round(confidence, 4),
        })

    return blocks


def parse_fields(text_blocks: list[dict]) -> dict:
    """
    Extract structured fields from OCR text blocks using regex patterns.

    STRATEGY:
        We look for common label patterns like "Name:", "Date:", "Certificate No:"
        These are simple heuristics — a production system would use NER (Named
        Entity Recognition) models, but regex works well for structured documents
        with predictable layouts.

    Returns a dict like:
        {
            "name": "Koushik Reddy",
            "date": "15-June-2025",
            "id_number": "CERT-2025-78432",
            "raw_text": "full concatenated text...",
            "all_fields": { ... any other key:value pairs found ... }
        }
    """
    # Combine all text blocks into one string for regex matching
    full_text = "\n".join(block["text"] for block in text_blocks)

    result = {
        "name": None,
        "date": None,
        "id_number": None,
        "raw_text": full_text,
        "all_fields": {},
    }

    # ------------------------------------------------------------------
    # Pattern 1: Name
    # Looks for "Name:" or "Name :" followed by the actual name.
    # The name is captured as one or more words (letters, spaces, dots).
    # ------------------------------------------------------------------
    name_pattern = r"[Nn]ame\s*[:\-]\s*([A-Za-z \.]+)"
    name_match = re.search(name_pattern, full_text)
    if name_match:
        result["name"] = name_match.group(1).strip()

    # ------------------------------------------------------------------
    # Pattern 2: Date
    # Matches common date formats:
    #   - DD-MM-YYYY, DD/MM/YYYY
    #   - DD-Month-YYYY (e.g. 15-June-2025)
    #   - Also handles "Date:" prefix
    # ------------------------------------------------------------------
    date_pattern = r"[Dd]ate\s*[:\-]\s*([\d]{1,2}[\-/\.][A-Za-z0-9]+[\-/\.][\d]{2,4})"
    date_match = re.search(date_pattern, full_text)
    if date_match:
        result["date"] = date_match.group(1).strip()

    # ------------------------------------------------------------------
    # Pattern 3: ID / Certificate Number
    # Matches patterns like:
    #   - "Certificate No: CERT-2025-78432"
    #   - "ID: ABC123456"
    #   - "Reg No: 12345"
    # Captures alphanumeric strings with dashes.
    # ------------------------------------------------------------------
    id_pattern = r"(?:[Cc]ertificate|[Ii][Dd]|[Rr]eg(?:istration)?)\s*(?:[Nn]o\.?|[Nn]umber)?\s*[:\-]\s*([A-Za-z0-9\-]+)"
    id_match = re.search(id_pattern, full_text)
    if id_match:
        result["id_number"] = id_match.group(1).strip()

    # ------------------------------------------------------------------
    # Generic key:value extraction
    # Catches any "Label: Value" patterns we didn't specifically handle.
    # Useful for debugging and for fields we didn't anticipate.
    # ------------------------------------------------------------------
    generic_pattern = r"([A-Za-z\s]+?)\s*:\s*(.+)"
    for match in re.finditer(generic_pattern, full_text):
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        if key and value:
            result["all_fields"][key] = value

    return result


def parse_document(image_path: str) -> dict:
    """
    Main entry point: run OCR on the image and extract structured fields.

    Args:
        image_path: Path to the document image file.

    Returns:
        dict with keys: name, date, id_number, raw_text, all_fields,
                        ocr_blocks (list of raw OCR detections with bboxes)
    """
    print(f"[document_parser] Running OCR on: {image_path}")

    # Step 1: Extract text + bounding boxes
    text_blocks = extract_text(image_path)
    print(f"[document_parser] Detected {len(text_blocks)} text blocks")

    # Step 2: Parse structured fields from the text
    fields = parse_fields(text_blocks)

    # Attach the raw OCR blocks (needed by tamper_check.py later)
    fields["ocr_blocks"] = text_blocks

    return fields


# -----------------------------------------------------------------------
# CLI test: run directly to test on a sample image
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys
    import os

    # Default test image
    test_image = "data/sample_docs/sample_certificate.png"

    if len(sys.argv) > 1:
        test_image = sys.argv[1]

    if not os.path.exists(test_image):
        print(f"ERROR: Image not found: {test_image}")
        print("Run 'python generate_sample_doc.py' first to create a test image.")
        sys.exit(1)

    result = parse_document(test_image)

    # Pretty-print the result (excluding ocr_blocks for readability)
    display = {k: v for k, v in result.items() if k != "ocr_blocks"}
    print("\n" + "=" * 50)
    print("  Parsed Document Fields")
    print("=" * 50)
    print(json.dumps(display, indent=2, ensure_ascii=False))
    print(f"\nOCR blocks extracted: {len(result['ocr_blocks'])}")
    print("=" * 50)
