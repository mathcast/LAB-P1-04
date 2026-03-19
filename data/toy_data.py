import torch

def get_toy_data():
    encoder_input = torch.rand(1, 5, 512)
    decoder_input = torch.rand(1, 1, 512)
    return encoder_input, decoder_input