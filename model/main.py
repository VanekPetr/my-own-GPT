import torch
from model.language_model import BigramLanguageModel
from model.train import train_gpt
from model.data_loader import get_tran_val_spit, get_vocabulary_size

# Load the data
with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# pre-process data
vocab_size = get_vocabulary_size(text)
train_data, val_data = get_tran_val_spit(text)

# decoder: take a list of integers, output a string
chars = sorted(list(set(text)))
itos = {i: ch for i, ch in enumerate(chars)}
decode = lambda l: ''.join([itos[i] for i in l])

# Initialization
batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
m = BigramLanguageModel(vocab_size)     # get bigram language model

# train the model
trained_model = train_gpt(m, train_data, val_data,
                          batch_size=32, block_size=8, number_of_epochs=10000, eval_interval=1000)

# generate text from the model
context = torch.zeros((1, 1), dtype=torch.long)
generated_text = decode(trained_model.generate(context, max_new_tokens=5000)[0].tolist())
print(generated_text)
