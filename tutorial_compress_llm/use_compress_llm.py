from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# (Re)chargement rapide ultérieur

save_dir = "./quantized_falcon7b"

tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(
    save_dir,
    device_map="auto",
     )

model.eval()

#  Génération d’un exemple
prompt = "What is the capital of the United States"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=18)
print(tokenizer.decode(out[0], skip_special_tokens=True))