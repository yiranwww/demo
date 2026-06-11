# attention = softmax(QK^T/sqrt(d_k))V
# Where Q is the query, K is the key, V is the value, and d_k is the dimension of the key.
import numpy as np

def softmax(x):
    x = x-np.max(x, axis = -1, keepdims = True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis = -1, keepdims = True
                          )
    

def attention(Q, K, V):
    d_k = K.shape[-1]
    similarity = np.dot(Q, K.T) / np.sqrt(d_k)
    weights = softmax(similarity)
    output = np.dot(weights, V)
    return output