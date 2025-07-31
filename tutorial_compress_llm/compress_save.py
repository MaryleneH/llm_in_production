# Rappel librairies à installer : transformers, optimum, accelerate et bitsandbytes

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 1) Config quant 4-bit
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                # use 4-bit weights
    bnb_4bit_compute_dtype="float16", # compute in fp16
    bnb_4bit_quant_type="nf4"         # normal float-4
)

# 2) Choisissez votre modèle local plus puissant
model_id = "tiiuae/falcon-7b"

# 3) Tokenizer + Chargement du modèle (auto place sur GPU via accelerate)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_config
)
model.eval()

# 4) Génération d’un exemple
prompt = "I enjoy my day"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))

# ───────────────────────────────────────────────────────────────
# 5) Sauvegarde du modèle quantifié et du tokenizer
save_dir = "./quantized_falcon7b"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model and tokenizer saved to {save_dir}")

# 6) (Re)chargement rapide ultérieur
# tokenizer = AutoTokenizer.from_pretrained(save_dir)
# model = AutoModelForCausalLM.from_pretrained(
#     save_dir,
#     device_map="auto",
# )
# model.eval()
