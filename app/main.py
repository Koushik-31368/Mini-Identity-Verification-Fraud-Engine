"""
app/main.py
===========
Streamlit UI for the Mini Identity Verification & Fraud Engine.

Run with:
    streamlit run app/main.py

WHAT IT DOES:
    Provides a simple web interface where users can:
    1. Upload a document image (certificate, ID, etc.)
    2. Upload a selfie image (or capture via webcam)
    3. Click "Verify" to run the full KYC pipeline
    4. See step-by-step results from each module
    5. See the final verdict (Verified / Flagged / Rejected)
"""

import os
import sys
import json
import streamlit as st
from PIL import Image

# Add project root to path so we can import src modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.pipeline import run_pipeline


# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mini KYC Engine",
    page_icon="🔒",
    layout="wide",
)

st.title("Mini Identity Verification & Fraud Engine")
st.markdown("Upload a document and selfie to run KYC verification.")
st.divider()

# ── File uploaders ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Document Image")
    st.caption("Upload a certificate, ID card, or other identity document.")
    doc_file = st.file_uploader(
        "Upload document",
        type=["jpg", "jpeg", "png", "bmp"],
        key="doc_upload",
    )
    if doc_file:
        st.image(doc_file, caption="Uploaded Document", use_container_width=True)

with col2:
    st.subheader("2. Selfie / Live Photo")
    st.caption("Upload a selfie or photo of your face.")

    # Option to use webcam or file upload
    selfie_mode = st.radio("Input method:", ["Upload file", "Use webcam"], horizontal=True)

    if selfie_mode == "Upload file":
        selfie_file = st.file_uploader(
            "Upload selfie",
            type=["jpg", "jpeg", "png", "bmp"],
            key="selfie_upload",
        )
    else:
        selfie_file = st.camera_input("Take a selfie")

    if selfie_file:
        st.image(selfie_file, caption="Selfie", use_container_width=True)

st.divider()

# ── Run pipeline ─────────────────────────────────────────────────────
if st.button("Run Verification", type="primary", use_container_width=True):
    if not doc_file or not selfie_file:
        st.error("Please upload both a document image and a selfie before running verification.")
    else:
        # Save uploaded files to temp directory
        temp_dir = os.path.join(project_root, "data", "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)

        doc_path = os.path.join(temp_dir, "document.png")
        selfie_path = os.path.join(temp_dir, "selfie.png")

        # Save document
        doc_img = Image.open(doc_file)
        doc_img.save(doc_path)

        # Save selfie
        selfie_img = Image.open(selfie_file)
        selfie_img.save(selfie_path)

        # Run the pipeline with a progress indicator
        with st.spinner("Running KYC verification pipeline..."):
            result = run_pipeline(doc_path, selfie_path)

        # ── Display results ──────────────────────────────────────────
        st.divider()

        # Final verdict - big and prominent
        verdict = result["verdict"]
        score = result["final_score"]

        if verdict == "VERIFIED":
            st.success(f"## {verdict}", icon="✅")
        elif verdict == "FLAGGED FOR REVIEW":
            st.warning(f"## {verdict}", icon="⚠️")
        else:
            st.error(f"## {verdict}", icon="❌")

        st.metric("Overall Score", f"{score:.2f} / 1.00")
        st.divider()

        # Module-by-module results
        st.subheader("Step-by-Step Results")

        modules = result["module_results"]
        module_scores = result["module_scores"]

        # ── 1. Document Parsing ──────────────────────────────────────
        with st.expander(
            f"1. Document Parsing (score: {module_scores.get('document_parsing', 0):.2f})",
            expanded=True,
        ):
            doc_data = modules.get("document_parsing", {})
            if "error" in doc_data:
                st.error(f"Error: {doc_data['error']}")
            else:
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Name", doc_data.get("name", "Not found"))
                col_b.metric("Date", doc_data.get("date", "Not found"))
                col_c.metric("ID Number", doc_data.get("id_number", "Not found"))

        # ── 2. Tamper Check ──────────────────────────────────────────
        with st.expander(
            f"2. Tamper Check (score: {module_scores.get('tamper_check', 0):.2f})",
            expanded=True,
        ):
            tamper_data = modules.get("tamper_check", {})
            if "error" in tamper_data:
                st.error(f"Error: {tamper_data['error']}")
            else:
                if tamper_data.get("is_suspicious"):
                    st.warning(tamper_data.get("details", "Suspicious"))
                else:
                    st.success(tamper_data.get("details", "Clean"))
                st.metric("Tamper Score", f"{tamper_data.get('tamper_score', 0):.4f}")

        # ── 3. Face Match ────────────────────────────────────────────
        with st.expander(
            f"3. Face Match (score: {module_scores.get('face_match', 0):.2f})",
            expanded=True,
        ):
            face_data = modules.get("face_match", {})
            if "error" in face_data:
                st.error(f"Error: {face_data['error']}")
            else:
                if face_data.get("is_match"):
                    st.success(face_data.get("details", "Match"))
                else:
                    st.error(face_data.get("details", "No match"))
                col_a, col_b = st.columns(2)
                col_a.metric("Confidence", f"{face_data.get('confidence', 0):.4f}")
                col_b.metric("Distance", f"{face_data.get('distance', 'N/A')}")

        # ── 4. Liveness ──────────────────────────────────────────────
        with st.expander(
            f"4. Liveness Detection (score: {module_scores.get('liveness', 0):.2f})",
            expanded=True,
        ):
            live_data = modules.get("liveness", {})
            if "error" in live_data:
                st.error(f"Error: {live_data['error']}")
            else:
                label = live_data.get("label", "unknown")
                if label == "real":
                    st.success(live_data.get("details", "Real"))
                elif label == "spoof":
                    st.error(live_data.get("details", "Spoof"))
                else:
                    st.info(live_data.get("details", "Model not trained"))
                st.metric("Confidence", f"{live_data.get('confidence', 0):.4f}")

        # ── 5. Fraud Check ───────────────────────────────────────────
        with st.expander(
            f"5. Fraud Check (score: {module_scores.get('fraud_check', 0):.2f})",
            expanded=True,
        ):
            fraud_data = modules.get("fraud_check")
            if fraud_data is None:
                st.info("Skipped (no transaction data provided)")
            elif "error" in fraud_data:
                st.error(f"Error: {fraud_data['error']}")
            else:
                if fraud_data.get("is_fraud"):
                    st.error(fraud_data.get("details", "Fraud detected"))
                else:
                    st.success(fraud_data.get("details", "Legitimate"))

        # ── Weights breakdown ────────────────────────────────────────
        st.divider()
        with st.expander("Scoring Weights & Details"):
            st.json(result.get("weights_used", {}))
            st.text(result.get("details", ""))

        # Cleanup temp files
        for f in [doc_path, selfie_path]:
            if os.path.exists(f):
                os.remove(f)
