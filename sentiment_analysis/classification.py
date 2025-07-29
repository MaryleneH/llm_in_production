import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Chargement du tokenizer & modèle
model_name = "azizbarank/distilroberta-base-sst2-distilled"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()  # désactive dropout


# 3. Fonction de classification
def classify_sentiment(text: str):
    # Tokenize et envoie sur le device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)
   
    # Inférence
    with torch.no_grad():
        logits = model(**inputs).logits
   
    # Softmax + choix de la classe
    probs = torch.softmax(logits, dim=1)[0]
    label_id = torch.argmax(probs).item()
    
    # Récupère le mapping id→label
    labels = model.config.id2label
    return labels[label_id], probs[label_id].item()


# 4. Exécution
label, confidence = classify_sentiment("I am very happy")
print(f"Sentiment: {label} (confidence: {confidence:.2f})")
