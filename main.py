import torch
from models.transformer import Transformer
from utils.mask import generate_causal_mask
from data.toy_data import get_toy_data

d_model = 512
num_layers = 2
vocab_size = 10

model = Transformer(d_model, num_layers, vocab_size)

encoder_input, decoder_input = get_toy_data()

max_len = 5

generated = decoder_input

for step in range(max_len):
    print(f"\n================ STEP {step} ================")

    mask = generate_causal_mask(generated.size(1))

    print("Mask:\n", mask)

    output = model(encoder_input, generated, mask)

    probs = torch.softmax(output[:, -1, :], dim=-1)

    print("Probabilidades:", probs)

    next_token = probs.argmax(dim=-1)

    print("Token escolhido:", next_token.item())

    next_token_embed = torch.rand(1, 1, d_model)

    generated = torch.cat([generated, next_token_embed], dim=1)

    print("Nova sequência shape:", generated.shape)

