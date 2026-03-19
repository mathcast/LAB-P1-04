import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None, debug=False):
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention = F.softmax(scores, dim=-1)
    output = torch.matmul(attention, V)

    if debug:
        print("\n ATTENTION DEBUG")
        print("Q shape:", Q.shape)
        print("K shape:", K.shape)
        print("V shape:", V.shape)
        print("Scores:", scores[0])
        print("Attention:", attention[0])

    return output, attention