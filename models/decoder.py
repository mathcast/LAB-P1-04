import torch.nn as nn
from models.attention import scaled_dot_product_attention
from models.ffn import FeedForward
from models.add_norm import AddNorm

class DecoderBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = FeedForward(d_model)

        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)
        self.add_norm3 = AddNorm(d_model)

    def forward(self, y, encoder_output, mask):
        print("\n DECODER INPUT:", y.shape)
    
        attn1, _ = scaled_dot_product_attention(y, y, y, mask, debug=True)
        y = self.add_norm1(y, attn1)
    
        print("After Masked Attention:", y.shape)
    
        attn2, _ = scaled_dot_product_attention(y, encoder_output, encoder_output, debug=True)
        y = self.add_norm2(y, attn2)
    
        print("After Cross Attention:", y.shape)
    
        ffn_out = self.ffn(y)
        y = self.add_norm3(y, ffn_out)
    
        print("After FFN:", y.shape)
    
        return y


class Decoder(nn.Module):
    def __init__(self, d_model, num_layers, vocab_size):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, y, encoder_output, mask):
        for layer in self.layers:
            y = layer(y, encoder_output, mask)

        return self.fc_out(y)