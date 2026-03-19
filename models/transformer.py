import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, d_model, num_layers, vocab_size):
        super().__init__()
        self.encoder = Encoder(d_model, num_layers)
        self.decoder = Decoder(d_model, num_layers, vocab_size)

    def forward(self, src, tgt, mask):
        encoder_output = self.encoder(src)
        output = self.decoder(tgt, encoder_output, mask)
        return output