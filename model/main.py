import torch

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


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch('train')


for b in range(batch_size):     # batch dimension
    for t in range(block_size):     # time dimension
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")

# TODO: Bigram Language model
