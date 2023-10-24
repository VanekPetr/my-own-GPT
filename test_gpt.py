import torch
import os
from huggingface_hub import snapshot_download


def generate_shakespeare(number_of_characters, block_size=256):
    """
    Function to test capabilities of a pre-trained GPT model
    """
    # pick available device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get data
    print('--> Loading data...')
    with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # define decoder
    chars = sorted(list(set(text)))
    itos = {i: ch for i, ch in enumerate(chars)}
    decode = lambda l: ''.join([itos[i] for i in l])

    # download model
    print('--> Downloading model...')
    model_file_name = "pre_trained_model.pt"
    repo_name = "VanekPetr/shakespeare-like-gpt"
    model_dir = snapshot_download(repo_name)
    model_path = os.path.join(model_dir, model_file_name)
    trained_model = torch.load(model_path, map_location=torch.device(device))

    # generate text from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print('--> Generating text...')
    generated_text = decode(trained_model.generate(context,
                                                   max_new_tokens=number_of_characters,
                                                   block_size=block_size)[0].tolist())

    print('Generated Shakespeare:')
    print(generated_text)


if __name__ == '__main__':
    generate_shakespeare(number_of_characters=2000)
