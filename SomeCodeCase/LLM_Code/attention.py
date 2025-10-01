import torch
import math

def attention(query, key, value, drouput = None):
    # based on pytorch

    # get the dimension of key vector, which is same as query vector
    d_k = query.size(-1)
    # calculate the product of query and key (Q and K) then devide by sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # softmax
    p_atten = scores.softmax(dim=-1)
    if drouput is not None:
        p_atten = drouput(p_atten)
    return torch.matmul(p_atten, value), p_atten