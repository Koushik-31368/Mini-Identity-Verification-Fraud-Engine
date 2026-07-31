"""
End-to-end pipeline test.

Uses:
  - data/sample_docs/test_id_photo.jpg  (document/ID image)
  - data/sample_docs/test_selfie.jpg    (selfie image)
  - A sample row from data/creditcard.csv (transaction data)
"""
import json
import pandas as pd
from src.pipeline import run_pipeline

# Load a sample transaction row from the CSV
# Pick a known fraud row (Class=1) for a more interesting test
df = pd.read_csv("data/creditcard.csv")
fraud_rows = df[df["Class"] == 1]
sample_row = fraud_rows.iloc[0]

# Build transaction_data dict (V1-V28 + Amount, exclude Time and Class)
transaction_data = {}
for col in df.columns:
    if col not in ("Time", "Class"):
        transaction_data[col] = float(sample_row[col])

print("Sample transaction (fraud row):")
print(f"  Amount: {transaction_data['Amount']}")
print(f"  Class label in CSV: {int(sample_row['Class'])} (1=fraud)")
print()

# Run the full pipeline
result = run_pipeline(
    document_image="data/sample_docs/test_id_photo.jpg",
    selfie_image="data/sample_docs/test_selfie.jpg",
    transaction_data=transaction_data,
)

# Print full results (excluding raw module_results for readability)
print("\n" + "=" * 60)
print("  FULL RESULT SUMMARY")
print("=" * 60)
summary = {
    "verdict": result["verdict"],
    "final_score": result["final_score"],
    "module_scores": result["module_scores"],
    "weights_used": result["weights_used"],
}
print(json.dumps(summary, indent=2, default=str))

# Print module details
print("\nModule Details:")
for module, details in result["module_results"].items():
    print(f"\n  {module}:")
    if details is None:
        print("    Skipped")
    elif isinstance(details, dict):
        for k, v in details.items():
            print(f"    {k}: {v}")
