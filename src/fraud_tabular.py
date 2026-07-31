"""
src/fraud_tabular.py
====================
Tabular fraud detection using traditional ML.

WHAT IT DOES:
    Trains a Random Forest classifier on a tabular dataset (e.g. Kaggle's
    credit card fraud dataset) to detect fraudulent transactions.

    This module is separate from the visual/biometric checks — it handles
    the "behavioral" side of fraud detection: does the transaction data
    look suspicious based on patterns learned from historical fraud?

WHY RANDOM FOREST?
    For tabular data with ~30 numerical features:
    - Random Forest works out-of-the-box with no feature scaling needed
    - It handles class imbalance better than Logistic Regression
    - It's interpretable (feature importance scores)
    - It's fast to train on medium datasets
    - It rarely overfits with default hyperparameters

EXPECTED DATASET:
    Kaggle Credit Card Fraud Detection dataset:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

    File: data/creditcard.csv
    Columns: Time, V1-V28 (PCA features), Amount, Class (0=legit, 1=fraud)
    Size: ~284,807 rows, 31 columns
    Class distribution: 99.83% legit, 0.17% fraud (highly imbalanced!)

USAGE:
    # Training:
    python -m src.fraud_tabular train

    # Inference:
    from src.fraud_tabular import predict_fraud
    result = predict_fraud({"V1": -1.3, "V2": 0.5, ..., "Amount": 149.62})
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler


# ── Configuration ────────────────────────────────────────────────────
DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/fraud_scaler.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_model(use_random_forest: bool = True):
    """
    Train a fraud detection model on the credit card fraud dataset.

    Args:
        use_random_forest: If True, use Random Forest. If False, use
                          Logistic Regression.
    """
    print("=" * 60)
    print("  Tabular Fraud Detection - Training")
    print("=" * 60)

    # ── Step 1: Load the dataset ─────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Dataset not found at: {DATA_PATH}")
        print("Download it from:")
        print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"And save the CSV file as: {DATA_PATH}")
        return

    print(f"\nLoading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df["Class"].value_counts())
    print(f"Fraud rate: {df['Class'].mean():.4%}")

    # ── Step 2: Prepare features and labels ──────────────────────────
    # Drop 'Time' column — it's not useful for our purposes
    # (it's just the elapsed seconds from the first transaction)
    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]

    # Feature names for later use in inference
    feature_names = list(X.columns)

    # ── Step 3: Train/test split ─────────────────────────────────────
    # stratify=y ensures the same fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples ({y_train.sum()} fraud)")
    print(f"Test set:  {X_test.shape[0]} samples ({y_test.sum()} fraud)")

    # ── Step 4: Scale features ───────────────────────────────────────
    # The 'Amount' column has a very different scale from V1-V28 (which
    # are already PCA-transformed). Scaling helps both models.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Step 5: Train the model ──────────────────────────────────────
    if use_random_forest:
        print("\nTraining Random Forest classifier...")
        model = RandomForestClassifier(
            n_estimators=100,          # 100 decision trees
            max_depth=10,              # Limit depth to prevent overfitting
            class_weight="balanced",   # Auto-adjust weights for imbalanced classes
            random_state=RANDOM_STATE,
            n_jobs=-1,                 # Use all CPU cores
        )
    else:
        print("\nTraining Logistic Regression classifier...")
        model = LogisticRegression(
            class_weight="balanced",   # Handle class imbalance
            max_iter=1000,
            random_state=RANDOM_STATE,
        )

    model.fit(X_train_scaled, y_train)
    print("Training complete!")

    # ── Step 6: Evaluate ─────────────────────────────────────────────
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print("  Evaluation Results")
    print(f"{'='*60}")
    print(f"\n  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}  (of predicted fraud, how many are real fraud)")
    print(f"  Recall:    {recall:.4f}  (of actual fraud, how many did we catch)")
    print(f"  F1 Score:  {f1:.4f}  (harmonic mean of precision & recall)")

    # WHY precision & recall matter more than accuracy here:
    # With 99.83% legit transactions, a model that always says "legit"
    # gets 99.83% accuracy but catches ZERO fraud. That's useless.
    # Recall tells us: of all actual fraud, what % did we detect?
    # Precision tells us: of all our fraud alerts, what % are real fraud?

    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Legit   Fraud")
    print(f"  Actual Legit  {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"  Actual Fraud  {cm[1][0]:>6}  {cm[1][1]:>6}")

    print(f"\n  Detailed Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # ── Step 7: Save model and scaler ────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"scaler": scaler, "feature_names": feature_names}, SCALER_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")

    # ── Step 8: Feature importance (Random Forest only) ──────────────
    if use_random_forest and hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"\nTop 10 most important features:")
        for name, imp in importances[:10]:
            print(f"  {name:<10} {imp:.4f}")

    print(f"\n{'='*60}")


def predict_fraud(features: dict) -> dict:
    """
    Run fraud inference on a single transaction.

    Args:
        features: Dict of feature values. Must include the same features
                 the model was trained on (V1-V28, Amount).
                 Missing features are filled with 0.

    Returns:
        {
            "is_fraud":    bool,
            "probability": float (0-1, probability of fraud),
            "details":     str,
        }
    """
    result = {
        "is_fraud": False,
        "probability": 0.0,
        "details": "",
    }

    # Load model and scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        result["details"] = (
            "Fraud model not found. Run 'python -m src.fraud_tabular train' first."
        )
        return result

    model = joblib.load(MODEL_PATH)
    meta = joblib.load(SCALER_PATH)
    scaler = meta["scaler"]
    feature_names = meta["feature_names"]

    # Build feature vector (fill missing features with 0)
    feature_vector = [features.get(name, 0) for name in feature_names]
    X = np.array([feature_vector])

    # Scale and predict
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]  # Probability of class 1 (fraud)

    result["is_fraud"] = bool(prediction == 1)
    result["probability"] = round(float(probability), 4)

    if result["is_fraud"]:
        result["details"] = f"FRAUD DETECTED ({probability:.1%} probability)"
    else:
        result["details"] = f"Transaction appears legitimate ({probability:.1%} fraud probability)"

    return result


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model(use_random_forest=True)
    else:
        print("Usage:")
        print("  python -m src.fraud_tabular train    # Train the model")
        print()
        print("Expected dataset: data/creditcard.csv")
        print("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")

        # Quick check: does the dataset exist?
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH, nrows=5)
            print(f"\nDataset found! Columns: {list(df.columns)}")
            print(f"Run 'python -m src.fraud_tabular train' to train.")
        else:
            print(f"\nDataset NOT found at: {DATA_PATH}")
