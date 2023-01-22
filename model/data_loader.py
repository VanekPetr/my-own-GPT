import torch


def get_batch(data, batch_size, block_size):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


def get_vocabulary_size(text):
    # Get all characters and investigate vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    return vocab_size


def get_tran_val_spit(text, train_ratio=0.9):
    # Get all characters and investigate vocabulary
    chars = sorted(list(set(text)))

    # create a mapping from characters to integers
    # TODO test Tiktoken from OpenAI or SentencePiece from Google
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    # decoder: take a list of integers, output a string
    # itos = {i: ch for i, ch in enumerate(chars)}
    # decode = lambda l: ''.join([itos[i] for i in l])

    # encode the entire text dataset and store it into a torch.Tensor
    data = torch.tensor(encode(text), dtype=torch.long)

    # split up the data into train and validation sets
    n = int(train_ratio * len(data))  # first 90% will be a train, rest val
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data
