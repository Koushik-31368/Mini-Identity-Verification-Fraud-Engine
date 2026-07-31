# Mini Identity Verification & Fraud Engine

A scaled-down KYC (Know Your Customer) identity verification pipeline that combines
document intelligence, face recognition, liveness detection, and tabular fraud detection
into a single system. Built as a portfolio project demonstrating applied ML engineering.

## The Problem

Financial institutions and online platforms need to verify that users are who they claim
to be. A real KYC pipeline involves:
- Checking uploaded identity documents for authenticity
- Matching the person's face against their ID photo
- Ensuring the person is physically present (not using a printed photo)
- Detecting fraudulent transaction patterns

This project implements a **simplified but functional** version of this pipeline.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app/main.py)                │
│         Upload Document + Selfie → Run Verification         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Pipeline Orchestrator (src/pipeline.py)         │
│          Runs all modules, computes weighted verdict         │
└──┬──────┬──────────┬──────────┬─────────────┬───────────────┘
   │      │          │          │             │
   ▼      ▼          ▼          ▼             ▼
┌──────┐┌──────┐ ┌────────┐ ┌────────┐ ┌───────────┐
│ Doc  ││Tamper│ │  Face  │ │Liveness│ │  Fraud    │
│Parser││Check │ │ Match  │ │  CNN   │ │ Tabular   │
│      ││      │ │        │ │        │ │           │
│EasyOCR│ BBox ││DeepFace││MobileNet││RandomForest│
│+Regex││Stats │ │+Media- │ │V2 fine-│ │ on credit │
│      ││      │ │ pipe   │ │ tuned  │ │ card data │
└──────┘└──────┘ └────────┘ └────────┘ └───────────┘
```

### Module Details

| Module | File | What It Does | Key Tech |
|--------|------|-------------|----------|
| **Document Parser** | `src/document_parser.py` | OCR text extraction + regex field parsing (name, date, ID) | EasyOCR |
| **Tamper Check** | `src/tamper_check.py` | Analyzes OCR bounding box consistency (font height, spacing) | NumPy, z-score |
| **Face Match** | `src/face_match.py` | Detects faces, compares ID photo vs selfie | Mediapipe, DeepFace |
| **Liveness Detection** | `src/liveness_model.py` | Classifies real vs spoof faces (transfer learning CNN) | PyTorch, MobileNetV2 |
| **Fraud Detection** | `src/fraud_tabular.py` | Classifies transactions as legit vs fraud | scikit-learn, Random Forest |
| **Pipeline** | `src/pipeline.py` | Orchestrates all modules, weighted scoring, final verdict | Python |
| **UI** | `app/main.py` | Web interface for uploads and results display | Streamlit |

### Verdict Logic

The pipeline assigns each module a score (0-1) and computes a weighted average:

| Module | Weight | Rationale |
|--------|--------|-----------|
| Face Match | 0.35 | Most critical — wrong face = wrong person |
| Liveness | 0.25 | Critical — spoof = not physically present |
| Tamper Check | 0.20 | High — forged document = fraud attempt |
| Document Parsing | 0.10 | Medium — missing fields = incomplete verification |
| Fraud Check | 0.10 | Medium — behavioral signal, supplementary |

**Verdict thresholds:**
- Score >= 0.70 → **VERIFIED**
- Score 0.40–0.70 → **FLAGGED FOR REVIEW**
- Score < 0.40 → **REJECTED**

## Setup

### Prerequisites
- Python 3.11 (TensorFlow/DeepFace don't support 3.14 yet)
- pip

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify all libraries
python verify_imports.py
```

### Data Setup

1. **Liveness training data** (Stage 5):
   - Put ~100-150 real face images in `data/real/`
   - Put ~100-150 spoof face images in `data/spoof/`
   - Run: `python train_liveness.py`

2. **Fraud detection data** (Stage 6):
   - Download [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Save as `data/creditcard.csv`
   - Run: `python -m src.fraud_tabular train`

### Running the App

```bash
streamlit run app/main.py
```

## What's Simplified vs. Production

| Aspect | This Project | Production System |
|--------|-------------|-------------------|
| **OCR** | EasyOCR + regex | Cloud OCR APIs (Google Vision, AWS Textract) + NER models |
| **Document validation** | Font height heuristics | Forensic analysis: EXIF metadata, compression artifacts, watermark verification |
| **Face matching** | Single model (VGG-Face) | Ensemble of models, 3D face reconstruction, pose normalization |
| **Liveness** | 2-class CNN (real/spoof) | Multi-modal: depth sensors, 3D face mesh, challenge-response (blink, turn head) |
| **Fraud detection** | Single Random Forest | Gradient boosting ensemble, real-time feature engineering, graph neural networks |
| **Data scale** | ~300 images, 284K transactions | Millions of images, billions of transactions |
| **Infrastructure** | Local Streamlit app | Microservices, GPU inference servers, message queues, audit logging |
| **Compliance** | None | GDPR, PCI-DSS, SOC2, data encryption at rest/in transit |

## Results & Metrics

### Liveness Model (MobileNetV2 Transfer Learning)
- **Architecture**: MobileNetV2 base (frozen) + custom head (1280→512→2)
- **Trainable params**: 656,898 / 2,880,770 (22.8%)
- **Training**: ~10 epochs on ~300 images with augmentation
- *(Accuracy/loss metrics will be populated after training with your dataset)*

### Fraud Detection (Random Forest)
- **Model**: 100 trees, max_depth=10, balanced class weights
- **Dataset**: Kaggle Credit Card Fraud (284,807 transactions, 0.17% fraud rate)
- *(Precision/recall/F1 metrics will be populated after training)*

## Project Structure

```
Mini Identity Verification & Fraud Engine/
├── app/
│   └── main.py              # Streamlit web UI
├── data/
│   ├── real/                 # Real face images (for liveness training)
│   ├── spoof/                # Spoof face images (for liveness training)
│   ├── sample_docs/          # Sample document images for testing
│   └── creditcard.csv        # Kaggle fraud dataset (you download)
├── models/
│   ├── liveness_model.pt     # Trained liveness CNN
│   └── fraud_model.pkl       # Trained fraud classifier
├── src/
│   ├── document_parser.py    # OCR + field extraction
│   ├── tamper_check.py       # Document tamper detection
│   ├── face_match.py         # Face detection + matching
│   ├── liveness_model.py     # Liveness CNN model + inference
│   ├── fraud_tabular.py      # Tabular fraud classifier
│   └── pipeline.py           # Unified verification pipeline
├── train_liveness.py         # Liveness model training script
├── generate_sample_doc.py    # Generate test certificate images
├── verify_imports.py         # Library installation checker
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Tech Stack

- **Python 3.11**
- **PyTorch** — CNN fine-tuning (MobileNetV2)
- **OpenCV** — Image processing
- **EasyOCR** — Optical character recognition
- **DeepFace** — Face verification (VGG-Face embeddings)
- **Mediapipe** — Face detection
- **scikit-learn** — Random Forest, evaluation metrics
- **Streamlit** — Web UI
- **NumPy / Pandas** — Data processing

## License

This is a portfolio/educational project. Use freely for learning purposes.
