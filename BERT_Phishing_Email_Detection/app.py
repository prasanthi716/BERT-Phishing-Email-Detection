
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

if (PARENT_DIR / "models" / "bert_phishing_grouped").exists():
    PROJECT_ROOT = PARENT_DIR
else:
    PROJECT_ROOT = CURRENT_DIR

MODEL_DIR = PROJECT_ROOT / "models" / "bert_phishing_grouped"
REPORTS_DIR = PROJECT_ROOT / "reports"
GROUPED_REPORTS_DIR = REPORTS_DIR / "grouped_eval"

st.set_page_config(
    page_title="BERT Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #101b33;
    }

    [data-testid="stSidebar"] * {
        color: #f1f5ff;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #102344, #1d4e89);
        color: white;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
        font-size: 34px;
    }

    .hero p {
        color: #e0ebff;
        font-size: 16px;
        margin-bottom: 0;
    }

    div.stButton > button[kind="primary"] {
        background-color: #1769aa;
        border: none;
        border-radius: 9px;
        min-height: 44px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #104e82;
        border: none;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e1e8f2;
        padding: 16px;
        border-radius: 12px;
    }

    .small-note {
        color: #64748b;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Trained grouped model not found at: {MODEL_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )

    model.eval()

    return tokenizer, model


def load_csv(path):
    if path.exists():
        return pd.read_csv(path)

    return None


def render_hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("# 🛡️ Find Threats")
    st.markdown("### BERT Email Security")
    st.caption("AI-powered phishing email analysis")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Email Analyzer",
            "Model Performance",
            "About Project",
        ],
        index=0,
    )

    st.divider()

    st.markdown("### Model Information")
    st.write("Architecture: BERT")
    st.write("Task: Binary text classification")
    st.write("Input: Email text")

    st.divider()

    st.caption("Academic prototype")
    st.caption("Predictions require independent verification.")


if page == "Email Analyzer":
    render_hero(
        "🛡️ BERT Phishing Email Detector",
        "Analyze email text with a fine-tuned BERT model to identify "
        "potentially legitimate and phishing messages.",
    )

    st.warning(
        "Academic prototype only. The model was trained using 2,000 "
        "email records and evaluated on a held-out test set. It may "
        "misclassify emails and does not guarantee real-world protection."
    )

    left, right = st.columns([1.7, 1], gap="large")

    with left:
        st.markdown("### Analyze an Email")

        st.write(
            "Paste the email subject and body below. Remove passwords, "
            "personal information, and other sensitive data first."
        )

        with st.form("email_analysis_form"):
            email_text = st.text_area(
                "Email content",
                height=260,
                placeholder=(
                    "Subject: Your email subject\n\n"
                    "Paste the email body here..."
                ),
                help=(
                    "The model processes up to 128 tokens. Longer emails "
                    "may be truncated."
                ),
            )

            submitted = st.form_submit_button(
                "Analyze Email",
                type="primary",
                width="stretch",
            )

        if submitted:
            if not email_text.strip():
                st.warning("Please paste an email before analyzing.")
            else:
                try:
                    with st.spinner("Analyzing email with BERT..."):
                        tokenizer, model = load_model()

                        inputs = tokenizer(
                            email_text.strip(),
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                            max_length=128,
                        )

                        with torch.no_grad():
                            outputs = model(**inputs)
                            probabilities = torch.softmax(
                                outputs.logits,
                                dim=1,
                            )[0]

                        predicted_label = int(
                            torch.argmax(probabilities).item()
                        )

                        confidence = float(
                            probabilities[predicted_label].item()
                        )

                    st.session_state["last_prediction"] = {
                        "label": predicted_label,
                        "confidence": confidence,
                        "legitimate_probability": float(
                            probabilities[0].item()
                        ),
                        "phishing_probability": float(
                            probabilities[1].item()
                        ),
                    }

                except Exception as error:
                    st.error(
                        "Analysis failed. Check that the trained model "
                        f"files are available. Details: {error}"
                    )

        result = st.session_state.get("last_prediction")

        if result:
            st.divider()
            st.markdown("### Analysis Result")

            result_col, confidence_col = st.columns(2)

            with result_col:
                if result["label"] == 1:
                    st.error("🚨 Potential Phishing Email")
                    st.write(
                        "Treat this message with caution. Do not click "
                        "links or open attachments unless verified."
                    )
                else:
                    st.success("Potentially Legitimate Email")
                    st.write(
                        "The model classified this email as legitimate. "
                        "This does not prove that the email is safe."
                    )

            with confidence_col:
                st.metric(
                    "Model confidence",
                    f'{result["confidence"]:.1%}',
                )

                st.progress(result["confidence"])

            st.markdown("#### Class probabilities")

            probability_col1, probability_col2 = st.columns(2)

            with probability_col1:
                st.metric(
                    "Legitimate",
                    f'{result["legitimate_probability"]:.1%}',
                )

            with probability_col2:
                st.metric(
                    "Phishing",
                    f'{result["phishing_probability"]:.1%}',
                )

            st.caption(
                "Confidence and class probabilities are model outputs, "
                "not verified measures of correctness or safety."
            )

            st.info(
                "Verify unexpected requests through a trusted channel "
                "using contact details obtained independently. This "
                "prototype does not inspect live URLs, sender reputation, "
                "email headers, or attachments."
            )

    with right:
        st.markdown("### How It Works")

        st.markdown(
            """
            **01 — Input**

            Provide email subject and body text.

            **02 — Tokenization**

            BERT's tokenizer converts text into model inputs.

            **03 — Classification**

            The fine-tuned model predicts one of two classes.

            **04 — Result**

            View the predicted category and class probabilities.
            """
        )

        st.divider()

        st.markdown("### Model Details")

        st.write("Model: BERT base uncased")
        st.write("Training records: 2,000")
        st.write("Maximum input length: 128 tokens")
        st.write("Classes: Legitimate / Phishing")

        st.warning(
            "Long messages may be truncated. The application is an "
            "academic demonstration, not a replacement for email security."
        )


elif page == "Model Performance":
    render_hero(
        "📊 Model Performance",
        "Review the evaluation results for the grouped BERT phishing detection model.",
    )

    metrics = load_csv(
        GROUPED_REPORTS_DIR / "metrics_grouped.csv"
    )

    if metrics is None or metrics.empty:
        st.error(
            "Grouped model evaluation metrics were not found. "
            "Check that reports/grouped_eval/metrics_grouped.csv exists."
        )
    else:
        row = metrics.iloc[0]

        accuracy = float(row["accuracy"])
        precision = float(row["precision_phishing"])
        recall = float(row["recall_phishing"])
        f1_score = float(row["f1_phishing"])

        st.markdown("### Evaluation Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.2%}")
        col2.metric("Phishing Precision", f"{precision:.2%}")
        col3.metric("Phishing Recall", f"{recall:.2%}")
        col4.metric("Phishing F1-score", f"{f1_score:.2%}")

        st.caption(
            "These results are from a held-out, group-aware test split. "
            "They do not guarantee equivalent performance on new emails."
        )

        detail_col1, detail_col2, detail_col3 = st.columns(3)

        detail_col1.metric(
            "Training examples",
            str(int(row["training_examples"])),
        )

        detail_col2.metric(
            "Test examples",
            str(int(row["test_examples"])),
        )

        detail_col3.metric(
            "Training epochs",
            str(int(row["epochs"])),
        )

        st.divider()

        st.markdown("### Confusion Matrix")

        confusion_data = load_csv(
            GROUPED_REPORTS_DIR / "confusion_matrix_grouped.csv"
        )

        if confusion_data is not None and not confusion_data.empty:
            st.dataframe(
                confusion_data,
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "The confusion matrix shows correct and incorrect "
                "predictions for the two classes."
            )
        else:
            st.info("The grouped confusion matrix was not found.")

        st.markdown("### Detailed Classification Report")

        classification_path = (
            GROUPED_REPORTS_DIR / "classification_report_grouped.txt"
        )

        if classification_path.exists():
            with st.expander("View classification report"):
                st.text(
                    classification_path.read_text(
                        encoding="utf-8"
                    )
                )
        else:
            st.info("The classification report was not found.")

        st.divider()

        st.markdown("### Evaluation Configuration")

        st.write("Model: google-bert/bert-base-uncased")
        st.write("Evaluation type: Group-aware test split")
        st.write("Maximum input length: 128 tokens")
        st.write(f"Batch size: {int(row['batch_size'])}")
        st.write(f"Training epochs: {int(row['epochs'])}")
        st.write(f"Training examples: {int(row['training_examples'])}")
        st.write(f"Test examples: {int(row['test_examples'])}")

        st.info(
            "Grouping reduces overlap between highly similar emails "
            "across the training and test sets. It does not guarantee "
            "generalization to every real-world phishing attack."
        )


elif page == "About Project":
    render_hero(
        "ℹ️ About the Project",
        "An academic implementation of transformer-based email text classification.",
    )

    st.markdown("### Project Overview")

    st.write(
        "The BERT-Based Phishing Email Detection System applies Natural "
        "Language Processing and a fine-tuned BERT model to classify "
        "email text as potentially legitimate or phishing."
    )

    st.markdown("### Project Objectives")

    st.markdown(
        """
        - Prepare labelled email data for binary classification.
        - Fine-tune a pretrained BERT language model.
        - Evaluate classification performance using standard metrics.
        - Provide an interactive interface for email text analysis.
        - Present saved model evaluation results.
        """
    )

    technology_col1, technology_col2 = st.columns(2)

    with technology_col1:
        st.markdown("### Machine Learning")
        st.write("Python")
        st.write("PyTorch")
        st.write("Hugging Face Transformers")
        st.write("BERT")

    with technology_col2:
        st.markdown("### Application and Evaluation")
        st.write("Streamlit")
        st.write("Pandas")
        st.write("Matplotlib / Seaborn")
        st.write("Classification metrics")

    st.markdown("### Limitations")

    st.markdown(
        """
        - The model was trained using a sample of 2,000 email records.
        - Inputs are truncated to a maximum of 128 tokens.
        - Only email text is classified; live links, headers, senders,
          and attachments are not independently inspected.
        - The model may incorrectly classify legitimate or phishing emails.
        - Confidence is not proof that a prediction is correct.
        - The application does not connect to an email inbox or
          automatically block messages.
        """
    )

    st.markdown("### Future Enhancements")

    st.markdown(
        """
        - Evaluate on larger, more diverse, independently collected data.
        - Investigate duplicates and data leakage between splits.
        - Improve handling of long emails.
        - Add explainability and more comprehensive test coverage.
        - Explore safe integration with email security workflows.
        """
    )

    st.info(
        "This project is an academic prototype. Always verify suspicious "
        "messages independently and use established security controls."
    )


st.divider()

st.caption(
    "B.Tech AI & DS | BERT-Based Phishing Email Detection System"
)
