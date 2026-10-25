
from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
OUTPUT_DIR = ROOT / "data" / "grouped_eval"

for path in (TRAIN_PATH, TEST_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
data = pd.concat([train, test], ignore_index=True)

texts = (
    data["text"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

if texts.eq("").any():
    raise ValueError("Empty email text found.")

if data["label"].isna().any():
    raise ValueError("Missing labels found.")

print("Emails:", len(data))
print("Building TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    dtype="float32",
)

features = vectorizer.fit_transform(texts)

print("Finding similar email pairs...")

neighbors = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_jobs=-1,
)
neighbors.fit(features)

threshold = 0.90
batch_size = 100

rows = []
cols = []

for start in range(0, len(data), batch_size):
    end = min(start + batch_size, len(data))
    distances, indices = neighbors.radius_neighbors(
        features[start:end],
        radius=1 - threshold,
        return_distance=True,
        sort_results=True,
    )

    for local_index, (row_distances, row_indices) in enumerate(
        zip(distances, indices)
    ):
        source_index = start + local_index

        for distance, target_index in zip(row_distances, row_indices):
            target_index = int(target_index)

            if source_index != target_index:
                rows.append(source_index)
                cols.append(target_index)

print("Similarity links found:", len(rows))

graph = csr_matrix(
    ([1] * len(rows), (rows, cols)),
    shape=(len(data), len(data)),
    dtype="int8",
)

graph = graph.maximum(graph.T)
group_count, group_ids = connected_components(
    graph,
    directed=False,
    return_labels=True,
)

data = data.copy()
data["group_id"] = group_ids
data["text"] = data["text"].fillna("").astype(str)

print("\n=== GROUP SUMMARY ===")
print("Email groups:", group_count)
print("Largest group:", int(pd.Series(group_ids).value_counts().max()))
print("\nGroup sizes:")
print(pd.Series(group_ids).value_counts().describe())

print("\n=== LABEL DISTRIBUTION ===")
print(data["label"].value_counts().sort_index())

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / "grouped_emails.csv"
data.to_csv(output_path, index=False)

print("\nSaved grouped dataset to:", output_path)
print("Original train/test files were not modified.")
print(
    "Note: connected groups can grow through chains of similar emails. "
    "Review the group summary before choosing a final split."
)
