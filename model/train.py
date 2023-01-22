import torch
import os
from model.language_model import BigramLanguageModel
from model.data_loader import get_batch


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters=200):
    out = {}
    # switch model to evaluation mode
    model.eval()
    for split in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split[1], batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split[0]] = losses.mean()
    model.train()
    return out


def train_gpt(model, train_data, val_data,
              batch_size: int = 32, block_size: int = 8, number_of_epochs: int = 100, eval_interval: int = 10):

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(number_of_epochs):  # increase number of steps for good results...

        # every once in a while evaluate the loss on train and val sets
        if epoch % eval_interval == 0 or epoch == number_of_epochs - 1:
            losses = estimate_loss(model, train_data, val_data, batch_size, block_size)
            print(f"epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(train_data, batch_size, block_size)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    validation_data = all_data[n:]

    # define model
    m = BigramLanguageModel(vocab_size)

    # *** TRAINING ***
    trained_model = train_gpt(m, training_data, validation_data,
                              batch_size=32, block_size=8, number_of_epochs=10000, eval_interval=1000)
