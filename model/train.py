import torch
import os
from model.language_model import BigramLanguageModel


def get_batch(data, batch_size, block_size):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


def train_gpt(model, train_data, batch_size: int = 32, block_size: int = 8, number_of_epochs: int = 100):

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for steps in range(number_of_epochs):  # increase number of steps for good results...

        # sample a batch of data
        xb, yb = get_batch(train_data, batch_size, block_size)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Loss: ', loss.item())

    return model


if __name__ == '__main__':
    # Load the data
    with open(os.path.join(os.path.dirname(os.getcwd()), 'data/shakespeare.txt'), 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    # encode the entire text dataset and store it into a torch.Tensor
    all_data = torch.tensor(encode(text), dtype=torch.long)

    # split up the data into train and validation sets, use only train data
    n = int(0.9 * len(all_data))  # first 90% will be a train, rest val
    training_data = all_data[:n]

    # define model
    m = BigramLanguageModel(vocab_size)

    # *** TRAINING ***
    trained_model = train_gpt(m, training_data, batch_size=32, block_size=8, number_of_epochs=10000)
