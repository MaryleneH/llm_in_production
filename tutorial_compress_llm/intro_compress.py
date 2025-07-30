from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Tell the model to load in 4-bit
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                # use 4-bit weights
    bnb_4bit_compute_dtype="float16", # compute in fp16
    bnb_4bit_quant_type="nf4"         # normal float-4
)


# Tokenizer et model
model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# examples = [tokenizer("I enjoy this day")]

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",                # auto-place on GPU
    quantization_config=quant_config  # apply 4-bit quantization
)

model.eval()
prompt = "What is the capital of France ?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# generate 10 new tokens
out = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(out[0], skip_special_tokens=True))
