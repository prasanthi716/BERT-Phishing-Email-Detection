
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "bert_phishing_large"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 128
EPOCHS = 1
BATCH_SIZE = 4
RANDOM_STATE = 42


class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )
        return item


def main():
    set_seed(RANDOM_STATE)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_file = DATA_DIR / "train.csv"
    test_file = DATA_DIR / "test.csv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            "Training or testing data is missing. "
            "Run: python src/prepare_data.py"
        )

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    required_columns = {"text", "label"}

    for name, frame in [
        ("training", train_df),
        ("testing", test_df),
    ]:
        if not required_columns.issubset(frame.columns):
            raise ValueError(
                f"The {name} dataset must contain text and label columns."
            )

    train_df = train_df.dropna(subset=["text", "label"]).copy()
    test_df = test_df.dropna(subset=["text", "label"]).copy()

    train_df["text"] = train_df["text"].astype(str).str.strip()
    test_df["text"] = test_df["text"].astype(str).str.strip()
    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    train_df = train_df[train_df["text"].str.len() > 0]
    test_df = test_df[test_df["text"].str.len() > 0]

    if set(train_df["label"].unique()) != {0, 1}:
        raise ValueError(
            "Training data must contain both labels: 0 and 1."
        )

    if set(test_df["label"].unique()) != {0, 1}:
        raise ValueError(
            "Testing data must contain both labels: 0 and 1."
        )

    print("Training records:", len(train_df), flush=True)
    print("Testing records:", len(test_df), flush=True)
    print("Device: CPU", flush=True)
    print("Epochs:", EPOCHS, flush=True)
    print("Batch size:", BATCH_SIZE, flush=True)

    print("\nLoading pretrained BERT tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading pretrained BERT classification model...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    train_dataset = EmailDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
    )

    test_dataset = EmailDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=str(
            REPORT_DIR / "training_checkpoints_large"
        ),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
        seed=RANDOM_STATE,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting BERT fine-tuning...", flush=True)
    trainer.train()

    print("\nEvaluating on the held-out test dataset...", flush=True)
    predictions = trainer.predict(test_dataset)

    logits = predictions.predictions
    predicted_labels = np.argmax(logits, axis=1)
    true_labels = test_df["label"].to_numpy()

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    if MODEL_DIR.exists():
        print(
            f"\nRemoving previous large-model output: {MODEL_DIR}",
            flush=True,
        )
        shutil.rmtree(MODEL_DIR)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    metrics = {
        "model": MODEL_NAME,
        "training_examples": int(len(train_df)),
        "test_examples": int(len(test_df)),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "accuracy": float(accuracy),
        "precision_phishing": float(precision),
        "recall_phishing": float(recall),
        "f1_phishing": float(f1),
    }

    pd.DataFrame([metrics]).to_csv(
        REPORT_DIR / "metrics_large.csv",
        index=False,
    )

    pd.DataFrame(
        matrix,
        index=["Actual Legitimate", "Actual Phishing"],
        columns=[
            "Predicted Legitimate",
            "Predicted Phishing",
        ],
    ).to_csv(
        REPORT_DIR / "confusion_matrix_large.csv"
    )

    with open(
        REPORT_DIR / "classification_report_large.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    print("\nBERT evaluation results", flush=True)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(report)

    print("\nConfusion matrix:")
    print(matrix)

    print(f"\nNew model saved to: {MODEL_DIR}")
    print(f"Metrics saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
