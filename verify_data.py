from pathlib import Path
import pandas as pd
import re

root = Path(__file__).resolve().parent
train_path = root / "data" / "processed" / "train.csv"
test_path = root / "data" / "processed" / "test.csv"

def normalize(value):
    value = str(value).lower().strip()
    return re.sub(r"\s+", " ", value)

for path in (train_path, test_path):
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("\n=== DATASET STRUCTURE ===")
print("Training rows:", len(train))
print("Test rows:", len(test))
print("Training columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())

text_candidates = ["text", "Email Text", "email_text", "email", "content"]
label_candidates = ["label", "Email Type", "email_type", "target"]

def find_column(frame, candidates):
    lookup = {str(col).lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None

train_text = find_column(train, text_candidates)
test_text = find_column(test, text_candidates)
train_label = find_column(train, label_candidates)
test_label = find_column(test, label_candidates)

if not train_text or not test_text:
    raise ValueError("Could not identify the email-text columns.")
if not train_label or not test_label:
    raise ValueError("Could not identify the label columns.")

train_emails = train[train_text].fillna("").map(normalize)
test_emails = test[test_text].fillna("").map(normalize)

train_nonempty = train_emails[train_emails != ""]
test_nonempty = test_emails[test_emails != ""]

train_set = set(train_nonempty)
test_set = set(test_nonempty)
overlap = train_set.intersection(test_set)

print("\n=== EXACT DUPLICATE CHECK ===")
print("Unique nonempty training emails:", len(train_set))
print("Unique nonempty test emails:", len(test_set))
print("Repeated emails within training data:", int(train_nonempty.duplicated().sum()))
print("Repeated emails within test data:", int(test_nonempty.duplicated().sum()))
print("Exact normalized emails shared across splits:", len(overlap))

if overlap:
    print("\nExamples of overlapping email text:")
    for email in list(overlap)[:3]:
        print("-", email[:180].replace("\n", " "))

print("\n=== LABEL DISTRIBUTION ===")
print("Training labels:")
print(train[train_label].value_counts(dropna=False).to_string())
print("\nTest labels:")
print(test[test_label].value_counts(dropna=False).to_string())

print("\n=== BASIC CHECKS ===")
print("Empty training emails:", int((train_emails == "").sum()))
print("Empty test emails:", int((test_emails == "").sum()))
print("Missing training labels:", int(train[train_label].isna().sum()))
print("Missing test labels:", int(test[test_label].isna().sum()))

print("\nVerification completed. Interpret the findings before changing any files.")
