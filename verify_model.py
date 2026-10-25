from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

root = Path(__file__).resolve().parent
model_dir = root / "models" / "bert_phishing_large"

print("Model directory:", model_dir)

if not model_dir.exists():
    raise FileNotFoundError(f"Model directory not found: {model_dir}")

tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
model.eval()

samples = [
    (
        "Legitimate example",
        "Subject: Project meeting\n\nHello team, our project meeting "
        "is scheduled for Tuesday at 10 AM. Thank you."
    ),
    (
        "Phishing example",
        "Subject: Urgent account verification\n\nYour account will be "
        "suspended today. Click immediately to verify your password "
        "and banking information."
    ),
]

print("\nModel loaded successfully.")
print("Device: CPU")
print("Model classes:", model.config.num_labels)

for name, email in samples:
    inputs = tokenizer(
        email,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_label = int(torch.argmax(probabilities).item())

    print(f"\n{name}")
    print("Predicted label:", predicted_label)
    print(f"Legitimate probability: {probabilities[0].item():.4%}")
    print(f"Phishing probability: {probabilities[1].item():.4%}")
    print("Probabilities valid:", bool(torch.isfinite(probabilities).all()))
    print("Probabilities sum to one:", abs(probabilities.sum().item() - 1.0) < 1e-5)

print("\nIndependent model verification completed.")
