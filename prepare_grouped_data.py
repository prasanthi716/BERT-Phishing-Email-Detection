
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"

for path in (TRAIN_PATH, TEST_PATH):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("=== DATASET INSPECTION ===")
print("Training rows:", len(train))
print("Test rows:", len(test))
print("Training columns:", list(train.columns))
print("Test columns:", list(test.columns))

required = {"text", "label"}

if not required.issubset(train.columns):
    raise ValueError("Training data must contain text and label columns.")

if not required.issubset(test.columns):
    raise ValueError("Test data must contain text and label columns.")

combined = pd.concat([train, test], ignore_index=True)

print("\nCombined rows:", len(combined))
print("Missing email texts:", int(combined["text"].isna().sum()))
print("Missing labels:", int(combined["label"].isna().sum()))
print("\nLabel distribution:")
print(combined["label"].value_counts().sort_index())

print("\nExact duplicate normalized emails:")
normalized = (
    combined["text"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
print(int(normalized.duplicated(keep=False).sum()))

print("\nInspection completed. No files were modified.")
