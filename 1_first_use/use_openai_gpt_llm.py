import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Choix du device (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Chargement du tokenizer et du modèle, puis bascule du modèle sur le device
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large")
model.to(device)
model.eval()  # passe en mode évaluation (désactive dropout)

# 3. Préparation du texte et tokenization
prompt = "Can you explain to me Linear Regression?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 4. Génération de texte
output_ids = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    top_p=0.92,
    top_k=0,
    temperature=0.6,
)

# 5. Décodage et affichage
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)


####################
# Autre exemple  ###
####################

# 3. Préparation du texte et tokenization
prompt = "Comment vas tu?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 4. Génération de texte
output_ids = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    top_p=0.92,
    top_k=0,
    temperature=0.6,
)

# 5. Décodage et affichage
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)