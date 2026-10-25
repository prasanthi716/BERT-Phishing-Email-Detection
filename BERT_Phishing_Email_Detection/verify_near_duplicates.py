
from pathlib import Path
import re
from difflib import SequenceMatcher

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"


def normalize(text):
    text = str(text).lower().strip()
    return re.sub(r"\s+", " ", text)


for path in (TRAIN_PATH, TEST_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

required_columns = {"text", "label"}

if not required_columns.issubset(train.columns):
    raise ValueError("Training data must contain text and label columns.")

if not required_columns.issubset(test.columns):
    raise ValueError("Test data must contain text and label columns.")

train_texts = train["text"].fillna("").map(normalize)
test_texts = test["text"].fillna("").map(normalize)

if (train_texts == "").any() or (test_texts == "").any():
    raise ValueError("Empty email text was found.")

print("Training emails:", len(train_texts))
print("Test emails:", len(test_texts))
print("\nChecking for highly similar emails across splits...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

combined = pd.concat(
    [train_texts, test_texts],
    ignore_index=True,
)

features = vectorizer.fit_transform(combined)

train_features = features[:len(train_texts)]
test_features = features[len(train_texts):]

neighbors = NearestNeighbors(
    n_neighbors=1,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1,
)

neighbors.fit(train_features)

distances, indices = neighbors.kneighbors(test_features)

similarities = 1 - distances.ravel()

thresholds = [0.80, 0.90, 0.95]

print("\n=== SIMILARITY SUMMARY ===")

for threshold in thresholds:
    count = int((similarities >= threshold).sum())
    print(
        f"Test emails with a training match >= {threshold:.0%}: "
        f"{count} / {len(test_texts)}"
    )

print("\n=== TOP 10 MOST SIMILAR CROSS-SPLIT PAIRS ===")

top_indices = similarities.argsort()[::-1][:10]

for test_index in top_indices:
    train_index = int(indices[test_index, 0])
    similarity = float(similarities[test_index])

    print("\n" + "-" * 70)
    print(f"Similarity: {similarity:.4f}")
    print(f"Test label: {test.iloc[test_index]['label']}")
    print(f"Train label: {train.iloc[train_index]['label']}")
    print("Test email:", test_texts.iloc[test_index][:250])
    print("Train email:", train_texts.iloc[train_index][:250])

print("\nNear-duplicate review completed.")
print(
    "Similarity is a screening signal, not proof of data leakage. "
    "Review the displayed pairs before drawing conclusions."
)
