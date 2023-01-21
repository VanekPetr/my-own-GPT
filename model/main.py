import torch
from model.language_model import BigramLanguageModel
from model.train import train_gpt, get_batch

# Load the data
with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all characters and investigate vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# create a mapping from characters to integers
# TODO test Tiktoken from OpenAI or SentencePiece from Google
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# test it
print(encode("hii there"))
print(decode(encode("hii there")))

# encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

# split up the data into train and validation sets
n = int(0.9*len(data))  # first 90% will be a train, rest val
train_data = data[:n]
val_data = data[n:]

batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?

xb, yb = get_batch(train_data, batch_size, block_size)

# get bigram language model
m = BigramLanguageModel(vocab_size)
out, out_loss = m(xb, yb)
print(out.shape, out_loss)
# test it with no hope
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


# train the model
trained_model = train_gpt(m, train_data, batch_size=32, block_size=8, number_of_epochs=10000)
# generate text from the model
context = torch.zeros((1, 1), dtype=torch.long)
generated_text = decode(trained_model.generate(context, max_new_tokens=500)[0].tolist())
print(generated_text)
