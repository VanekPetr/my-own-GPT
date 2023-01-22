import torch
import os
from model.language_model import BigramLanguageModel
from model.train import train_gpt
from model.data_loader import get_tran_val_spit, get_vocabulary_size


def train_and_generate_with_gpt(text):
    # *** MODEL ***
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # hyperparameters
    n_embeddings = 384 if device == 'cuda' else 32
    n_heads = 6 if device == 'cuda' else 4
    block_size = 256 if device == 'cuda' else 8   # what is the maximum context length for predictions?
    n_layers = 6

    vocab_size = get_vocabulary_size(text)

    # initialization
    model = BigramLanguageModel(vocab_size, n_embeddings, n_heads, n_layers, block_size)
    m = model.to(device)

    # *** TRAINING ***
    # pre-process data
    train_data, val_data = get_tran_val_spit(text)

    # train parameters
    batch_size = 64 if device == 'cuda' else 4     # how many independent sequences will we process in parallel?
    learning_rate = 3e-4 if device == 'cuda' else 1e-3
    max_iters = 5000
    eval_interval = 500
    eval_iters = 200

    trained_model = train_gpt(m, train_data, val_data,
                              batch_size=batch_size, block_size=block_size, number_of_epochs=max_iters,
                              eval_interval=eval_interval, eval_iters=eval_iters, learning_rate=learning_rate)

    # *** SHOW RESULTS ***
    # decoder: take a list of integers, output a string
    chars = sorted(list(set(text)))
    itos = {i: ch for i, ch in enumerate(chars)}
    decode = lambda l: ''.join([itos[i] for i in l])

    # generate text from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(trained_model.generate(context, max_new_tokens=500, block_size=block_size)[0].tolist())

    print('GENERATED TEXT:')
    print(generated_text)


if __name__ == '__main__':
    # *** DATA ***
    with open(os.path.join(os.path.dirname(os.getcwd()), 'data/shakespeare.txt'), 'r', encoding='utf-8') as f:
        data = f.read()

    train_and_generate_with_gpt(data)
