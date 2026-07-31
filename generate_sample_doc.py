"""
generate_sample_doc.py
======================
Creates a synthetic certificate/ID document image for testing
the OCR document parser. The text is known, so we can verify
that EasyOCR extracts it correctly.

Usage:
    python generate_sample_doc.py
    -> saves to data/sample_docs/sample_certificate.png
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_certificate(output_path: str):
    """Generate a realistic-looking certificate image with known text fields."""

    # Canvas: white background, typical document size
    width, height = 800, 500
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Use a basic font (available on all systems)
    # On Windows, arial.ttf is available; fall back to default if not
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        body_font  = ImageFont.truetype("arial.ttf", 18)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        # Fallback to PIL default if arial not found
        title_font = ImageFont.load_default()
        body_font  = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # --- Draw a border ---
    draw.rectangle([(20, 20), (width - 20, height - 20)], outline="black", width=3)
    draw.rectangle([(30, 30), (width - 30, height - 30)], outline="gray", width=1)

    # --- Title ---
    draw.text((200, 50), "CERTIFICATE OF COMPLETION", fill="black", font=title_font)

    # --- Body text with structured fields ---
    draw.text((100, 120), "This is to certify that", fill="black", font=body_font)

    # Name field (the parser should extract this)
    draw.text((100, 160), "Name: Koushik Reddy", fill="black", font=body_font)

    # Date field
    draw.text((100, 200), "Date: 15-June-2025", fill="black", font=body_font)

    # ID / Certificate number
    draw.text((100, 240), "Certificate No: CERT-2025-78432", fill="black", font=body_font)

    # Course / description
    draw.text((100, 280), "Course: Machine Learning Fundamentals", fill="black", font=body_font)

    # Issuing authority
    draw.text((100, 320), "Issued by: National Institute of Technology", fill="black", font=body_font)

    # Score
    draw.text((100, 360), "Score: 92/100  Grade: A+", fill="black", font=body_font)

    # Footer
    draw.text((250, 430), "Authorized Signature", fill="gray", font=small_font)
    draw.line([(250, 425), (550, 425)], fill="gray", width=1)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Sample certificate saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_sample_certificate("data/sample_docs/sample_certificate.png")
