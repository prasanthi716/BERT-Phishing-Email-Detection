
# BERT-Based Phishing Email Detection System

## Project Overview

An academic project that uses a fine-tuned BERT transformer model to classify email text as potentially legitimate or phishing.

The Streamlit application allows users to paste email content and view a predicted category and class probabilities.

## Objectives

- Prepare labelled email data for classification.
- Fine-tune a pretrained BERT model.
- Evaluate the model using classification metrics.
- Build an interactive Streamlit dashboard.
- Display model evaluation results.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- BERT
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn

## Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | 96.25% |
| Phishing precision | 96.53% |
| Phishing recall | 93.29% |
| Phishing F1-score | 94.88% |

Evaluation used 1,600 training examples and 400 held-out test examples. These results do not guarantee equivalent performance on new or real-world emails.

## Run Locally

1. Install Python and Git.
2. Create and activate a virtual environment.
3. Install the dependencies from `requirements.txt`.
4. Obtain the trained model files separately.
5. Run the application from the project root:

```powershell
streamlit run .\BERT_Phishing_Email_Detection\app.py
```

The trained model weights are excluded from this repository and must be made available separately before predictions can run.

## Limitations

- This is an academic prototype.
- Email input is limited to 128 tokens.
- The model may misclassify emails.
- The app does not independently inspect live URLs, sender reputation, email headers, or attachments.
- Confidence scores do not prove that predictions are correct.

## Disclaimer

Do not use this prototype as your only email security control. Verify suspicious messages independently.
