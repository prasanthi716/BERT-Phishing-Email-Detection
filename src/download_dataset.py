
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "emails_large.csv"

DATASET_NAME = "manhattin/phishing-email-dataset"
MAX_EMAIL_LENGTH = 10000
SAMPLE_SIZE = 2000
RANDOM_STATE = 42


def main():
    print("Downloading the public phishing email dataset...", flush=True)

    dataset = load_dataset(DATASET_NAME, split="train")
    df = dataset.to_pandas()

    print("Dataset columns:", df.columns.tolist(), flush=True)

    required_columns = {"Email Text", "Email Type"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Expected columns 'Email Text' and 'Email Type' were not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[["Email Text", "Email Type"]].copy()
    df.columns = ["text", "email_type"]

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["email_type"] = (
        df["email_type"].fillna("").astype(str).str.strip().str.lower()
    )

    label_mapping = {
        "safe email": 0,
        "phishing email": 1,
    }

    df["label"] = df["email_type"].map(label_mapping)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    df = df[
        (df["text"].str.len() >= 20)
        & (df["text"].str.len() <= MAX_EMAIL_LENGTH)
    ].copy()

    df = df[["text", "label"]]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable emails remain after cleaning.")

    if df["label"].nunique() != 2:
        raise ValueError(
            "Both legitimate and phishing email classes are required."
        )

    print("\nUsable records after cleaning:", len(df), flush=True)
    print("Class counts before sampling:")
    print(df["label"].value_counts().sort_index())

    if len(df) > SAMPLE_SIZE:
        try:
            df, _ = train_test_split(
                df,
                train_size=SAMPLE_SIZE,
                random_state=RANDOM_STATE,
                stratify=df["label"],
            )
        except ValueError as error:
            raise ValueError(
                "Could not create a stratified sample. "
                "Check the number of records in each class."
            ) from error

    df = df.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\nDataset saved successfully.", flush=True)
    print("File:", OUTPUT_FILE, flush=True)
    print("Total records:", len(df), flush=True)
    print("Columns:", df.columns.tolist(), flush=True)
    print("\nLabel counts (0=legitimate, 1=phishing):")
    print(df["label"].value_counts().sort_index())

    print("\nMissing values:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
