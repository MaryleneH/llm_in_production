# Remember to install transformers and datasets tools

from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from datasets import load_dataset
from torch.utils.data import Dataset

dataset = load_dataset("sst2")
print("import OK!")

# for row in dataset['train']:
#     print(row)


tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                          bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>',
                                          pad_token='<|pad|>')
# Preprocess Data (on le split train)
texts = []
for row in dataset['train']:
    texts.append(
        f"<|startoftext|> {row['sentence']}<|pad|>Sentiment:{row['label']}<|endoftext|>"
    )

# Tokenisation en batch + création d'un Dataset PyTorch
encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')


class GPT2Dataset(Dataset):
    def __init__(self, enc):
        self.input_ids = enc.input_ids
        self.attention_mask = enc.attention_mask

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            # pour un LM causal on peut utiliser les mêmes ids comme labels
            'labels':         self.input_ids[idx],
        }


train_dataset = GPT2Dataset(encodings)

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Quand on injecte des bos_token, eos_token et surtout un pad_token non présents dans le vocabulaire de base, 
# le tokenizer leur attribue des IDs au-delà de la taille originale (50257). 
# Il faut donc appeler model.resize_token_embeddings(...) 
# pour étendre la table d’embeddings du modèle et éviter le « device-side assert » CUDA lié à un index hors bornes.
model.resize_token_embeddings(len(tokenizer))

# Fine-tuning GPT-2

training_args = TrainingArguments(output_dir='results',
                                  num_train_epochs=1,
                                  warmup_steps=180,
                                  weight_decay=0.01)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Query the model

model.eval()

test_sentence = "I absolutely loved this movie"
prompt = f"<|startoftext|> {test_sentence}<|pad|>Sentiment:"

# 1. Tokenize + to(device)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 2. Génération restreinte
output_ids = model.generate(
    **inputs,
    max_new_tokens=2,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False
)

# 3. Décodage
decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 4. Extraction
sentiment = decoded.split("Sentiment:")[-1].strip()
print(f"Input     : {test_sentence}")
print(f"Prediction: Sentiment = {sentiment}")
