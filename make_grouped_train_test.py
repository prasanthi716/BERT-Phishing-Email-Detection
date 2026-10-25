
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data" / "grouped_eval" / "grouped_emails.csv"
OUTPUT_DIR = ROOT / "data" / "grouped_eval"

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Missing grouped dataset: {INPUT_PATH}")

data = pd.read_csv(INPUT_PATH)

required = {"text", "label", "group_id"}
if not required.issubset(data.columns):
    raise ValueError("Dataset must contain text, label, and group_id.")

if data["text"].isna().any() or data["label"].isna().any():
    raise ValueError("Missing text or labels found.")

if data["group_id"].isna().any():
    raise ValueError("Missing group IDs found.")

splitter = GroupShuffleSplit(
    n_splits=100,
    test_size=0.20,
    random_state=42,
)

best_split = None
best_score = float("inf")

for train_indices, test_indices in splitter.split(
    data,
    data["label"],
    groups=data["group_id"],
):
    train = data.iloc[train_indices].copy()
    test = data.iloc[test_indices].copy()

    if train["label"].nunique() != 2 or test["label"].nunique() != 2:
        continue

    test_fraction = len(test) / len(data)
    train_phishing_rate = train["label"].mean()
    test_phishing_rate = test["label"].mean()

    score = (
        abs(test_fraction - 0.20)
        + abs(train_phishing_rate - test_phishing_rate)
    )

    if score < best_score:
        best_score = score
        best_split = (train, test)

if best_split is None:
    raise RuntimeError(
        "Could not find a split containing both labels. "
        "Review group sizes and label distributions."
    )

train, test = best_split

overlap = set(train["group_id"]) & set(test["group_id"])
if overlap:
    raise RuntimeError("Group leakage detected: groups overlap.")

train_path = OUTPUT_DIR / "train_grouped.csv"
test_path = OUTPUT_DIR / "test_grouped.csv"

train[["text", "label", "group_id"]].to_csv(train_path, index=False)
test[["text", "label", "group_id"]].to_csv(test_path, index=False)

print("=== GROUPED SPLIT SUMMARY ===")
print("Total emails:", len(data))
print("Training emails:", len(train))
print("Test emails:", len(test))
print("Training groups:", train["group_id"].nunique())
print("Test groups:", test["group_id"].nunique())
print("Groups shared between sets:", len(overlap))

print("\nTraining label distribution:")
print(train["label"].value_counts().sort_index())

print("\nTest label distribution:")
print(test["label"].value_counts().sort_index())

print("\nTraining phishing rate:", round(train["label"].mean(), 4))
print("Test phishing rate:", round(test["label"].mean(), 4))

print("\nSaved:", train_path)
print("Saved:", test_path)
print("Original files and model were not modified.")
