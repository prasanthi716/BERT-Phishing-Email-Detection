
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "bert_phishing_grouped"

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")

print("Loading grouped BERT model...")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.eval()

device = torch.device("cpu")
model.to(device)

print("Model loaded successfully.")
print("Device:", device)
print("Number of classes:", model.config.num_labels)
print("Label mapping:", model.config.id2label)

test_emails = [
    (
        "Legitimate example",
        "Hello team, our weekly project meeting is scheduled for "
        "Tuesday at 10 AM in the main conference room. "
        "Please review the attached agenda before the meeting."
    ),
    (
        "Phishing example",
        "URGENT! Your bank account will be suspended today. "
        "Verify your password and account immediately using this link "
        "to avoid permanent account closure."
    ),
]

for name, email_text in test_emails:
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    predicted_label = int(torch.argmax(probabilities).item())

    print(f"\n{name}")
    print("Predicted label:", predicted_label)
    print(f"Legitimate probability: {probabilities[0].item() * 100:.2f}%")
    print(f"Phishing probability: {probabilities[1].item() * 100:.2f}%")
    print(f"Probabilities sum: {probabilities.sum().item():.4f}")

    if not torch.isfinite(probabilities).all():
        raise ValueError("Model produced invalid probabilities.")

    if abs(probabilities.sum().item() - 1.0) > 1e-5:
        raise ValueError("Probabilities do not sum to one.")

print("\nIndependent model verification completed.")
