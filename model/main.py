import torch
from model.language_model import BigramLanguageModel
from model.train import train_gpt
from model.data_loader import get_tran_val_spit, get_vocabulary_size

# *** DATA ***
with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# *** MODEL ***
# hyperparameters
vocab_size = get_vocabulary_size(text)
n_embeddings = 384
n_heads = 6
n_layers = 6
block_size = 256    # what is the maximum context length for predictions?

# initialization
m = BigramLanguageModel(vocab_size, n_embeddings, n_heads, n_layers, block_size)

# *** TRAINING ***
# pre-process data
train_data, val_data = get_tran_val_spit(text)

# train parameters
batch_size = 32     # how many independent sequences will we process in parallel?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

trained_model = train_gpt(m, train_data, val_data,
                          batch_size=batch_size, block_size=block_size, number_of_epochs=max_iters,
                          eval_interval=eval_interval, eval_iters=eval_iters)

# *** SHOW RESULTS ***
# decoder: take a list of integers, output a string
chars = sorted(list(set(text)))
itos = {i: ch for i, ch in enumerate(chars)}
decode = lambda l: ''.join([itos[i] for i in l])

# generate text from the model
context = torch.zeros((1, 1), dtype=torch.long)
generated_text = decode(trained_model.generate(context, max_new_tokens=500)[0].tolist())
print(generated_text)
