import torch

def generate_causal_mask(size):
    return torch.tril(torch.ones(size, size))