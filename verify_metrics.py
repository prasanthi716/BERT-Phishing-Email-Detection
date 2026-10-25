from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
reports = root / "reports"

metrics_path = reports / "metrics_large.csv"
matrix_path = reports / "confusion_matrix_large.csv"

for path in (metrics_path, matrix_path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

metrics = pd.read_csv(metrics_path)
matrix = pd.read_csv(matrix_path, index_col=0)

if metrics.empty:
    raise ValueError("The metrics file is empty.")

values = matrix.to_numpy(dtype=int)

if values.shape != (2, 2):
    raise ValueError(f"Expected a 2x2 confusion matrix, got {values.shape}")

tn, fp, fn, tp = values.ravel()
total = tn + fp + fn + tp

if total == 0:
    raise ValueError("The confusion matrix contains no observations.")

accuracy = (tn + tp) / total
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)

print("\n=== SAVED METRICS ===")
print(metrics.to_string(index=False))

print("\n=== CONFUSION MATRIX ===")
print(matrix.to_string())

print("\n=== RECALCULATED METRICS ===")
print(f"Total test records: {total}")
print(f"Accuracy:           {accuracy:.4%}")
print(f"Phishing precision: {precision:.4%}")
print(f"Phishing recall:    {recall:.4%}")
print(f"Phishing F1-score:  {f1:.4%}")

print("\n=== CONFUSION MATRIX COUNTS ===")
print(f"True negatives:  {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives:  {tp}")

print("\nVerification script completed.")
