
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

METRICS_FILE = REPORTS_DIR / "metrics_large.csv"
CONFUSION_FILE = REPORTS_DIR / "confusion_matrix_large.csv"
CLASSIFICATION_FILE = REPORTS_DIR / "classification_report_large.txt"

OUTPUT_DIR = REPORTS_DIR / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_confusion_matrix():
    matrix = pd.read_csv(CONFUSION_FILE, index_col=0)

    values = matrix.to_numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=matrix.columns,
        yticklabels=matrix.index,
    )
    plt.title("BERT Phishing Email Detection - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    output_file = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


def generate_performance_chart():
    metrics = pd.read_csv(METRICS_FILE)

    row = metrics.iloc[0]

    performance = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Phishing Precision",
                "Phishing Recall",
                "Phishing F1-score",
            ],
            "Score": [
                row["accuracy"],
                row["precision_phishing"],
                row["recall_phishing"],
                row["f1_phishing"],
            ],
        }
    )

    plt.figure(figsize=(9, 6))
    bars = plt.bar(
        performance["Metric"],
        performance["Score"],
    )

    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("BERT Model Performance on Held-Out Test Data")
    plt.xticks(rotation=15, ha="right")

    for bar, score in zip(bars, performance["Score"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{score:.2%}",
            ha="center",
        )

    plt.tight_layout()

    output_file = OUTPUT_DIR / "performance_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


def generate_summary():
    metrics = pd.read_csv(METRICS_FILE)

    summary = metrics[
        [
            "model",
            "training_examples",
            "test_examples",
            "epochs",
            "batch_size",
            "accuracy",
            "precision_phishing",
            "recall_phishing",
            "f1_phishing",
        ]
    ].copy()

    summary.to_csv(
        OUTPUT_DIR / "evaluation_summary.csv",
        index=False,
    )

    print("\nEvaluation summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'evaluation_summary.csv'}")


def main():
    required_files = [
        METRICS_FILE,
        CONFUSION_FILE,
        CLASSIFICATION_FILE,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        for path in missing_files:
            print(f"Missing required file: {path}")
        raise FileNotFoundError(
            "One or more evaluation files are missing."
        )

    generate_confusion_matrix()
    generate_performance_chart()
    generate_summary()

    print("\nAll evaluation charts and summaries were generated.")


if __name__ == "__main__":
    main()
