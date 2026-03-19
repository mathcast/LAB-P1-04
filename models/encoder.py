import torch.nn as nn
from models.attention import scaled_dot_product_attention
from models.ffn import FeedForward
from models.add_norm import AddNorm

class EncoderBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = FeedForward(d_model)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x):
        print("\n ENCODER INPUT:", x.shape)

        attn_out, _ = scaled_dot_product_attention(x, x, x, debug=True)
        x = self.add_norm1(x, attn_out)

        print("After Attention:", x.shape)

        ffn_out = self.ffn(x)
        x = self.add_norm2(x, ffn_out)

        print("After FFN:", x.shape)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  