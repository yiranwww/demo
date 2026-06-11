import numpy as np

def softmax(x):
    x = np.array(X)
    return np.exp(x) / np.sum(np.exp(x), axis = 0)

def softmax_stable(x):
    x = np.array(X)
    x_max = np.max(x, axis = 0)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis = 0)


# L = -1/N sum（y_i * log(p_i))

def cross_entropy_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return -np.sum(y_true * np.log(y_pred + 1e-15)) / y_true.shape[0]
