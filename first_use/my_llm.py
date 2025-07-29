from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch_device = "cuda"

# Fetch the model and Tokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id = tokenizer.eos_token_id).to(torch_device)

print("OK!")

# Tokenize our input Text

model_inputs = tokenizer('Can you explain to me Linear Regression', return_tensors='pt').to(torch_device)

# Query the model

output = model.generate(**model_inputs, max_new_tokens = 200, do_sample = True, top_p = 0.92, top_k = 0, temperature = 0.6)

# Decode the model output

print(tokenizer.decode(output[0], skip_special_tokens = True))