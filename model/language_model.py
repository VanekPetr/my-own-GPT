import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embed, block_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x)   # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5     # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)    # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)   # (B,T,C)
        out = wei @ v   # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class BigramLanguageModel(nn.Module):
    """
    Simple bigram language model
    """
    def __init__(self, vocabulary_size, n_embeddings=32, block_size=32):
        super().__init__()
        # each toke directly reads off the logits for the next token from a lookup table
        self.token_embeddings_table = nn.Embedding(vocabulary_size, n_embeddings)
        # block_size: maximum context length for predictions
        self.positional_embeddings_table = nn.Embedding(block_size, n_embeddings)
        self.self_attention_head = Head(n_embeddings, n_embeddings, block_size)
        self.lm_head = nn.Linear(n_embeddings, vocabulary_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        token_embeddings = self.token_embeddings_table(idx)   # (Batch = 4, Time = 8, Channel = n_embeddings)
        positional_embeddings = self.positional_embeddings_table(torch.arange(T))  # (Time = 8, Channel = n_embeddings)
        token_embeddings_plus_position = token_embeddings + positional_embeddings  # (B, T, C)
        x = self.self_attention_head(token_embeddings_plus_position)  # apply one head of self attention, (B, T, C)
        logits = self.lm_head(x)  # (Batch = 4, Time = 8, Channel = vocabulary_size)

        loss = None
        if targets is not None:
            # to be able to use cross entropy loss, we need to flatten the logits and targets
            _, _, channels = logits.shape
            logits = logits.view(-1, channels)  # (Batch * Time, Channel = 65)
            targets = targets.view(-1)  # (Batch * Time)

            # calculate the loss with cross entropy
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, block_size=32):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step, the new predicted token
            logits = logits[:, -1, :]  # becomes (B, C)
            # get probabilities with softmax
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution, but just one token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
