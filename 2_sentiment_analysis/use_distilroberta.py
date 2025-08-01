# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
# 1. Choix du device (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "distilbert/distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
model.to(device)

model.eval()  # passe en mode évaluation (désactive dropout)

# 3. Préparation du texte et tokenization
prompt = "I am very happy"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 4. Génération de texte / Query the model
output = model(**inputs).logits.argmax(axis=1)

print(output)