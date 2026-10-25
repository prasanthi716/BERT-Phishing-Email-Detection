
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "emails_large.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def main():
    print("Loading the cleaned email dataset...", flush=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Expected columns: {required_columns}. "
            f"Found: {df.columns.tolist()}"
        )

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() >= 20].copy()
    df = df[df["label"].isin([0, 1])].copy()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable email records remain.")

    if df["label"].nunique() != 2:
        raise ValueError(
            "Both legitimate and phishing email classes are required."
        )

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_FILE, index=False, encoding="utf-8")
    test_df.to_csv(TEST_FILE, index=False, encoding="utf-8")

    print("\nDataset preparation completed successfully.")
    print("Total usable emails:", len(df))
    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))

    print("\nTraining label counts (0=legitimate, 1=phishing):")
    print(train_df["label"].value_counts().sort_index())

    print("\nTesting label counts (0=legitimate, 1=phishing):")
    print(test_df["label"].value_counts().sort_index())

    print("\nSaved training dataset:", TRAIN_FILE)
    print("Saved testing dataset:", TEST_FILE)


if __name__ == "__main__":
    main()
