import numpy as np

def softmax(X):
    X = np.array(X)
    return 1/(1 + np.exp(-X))

def cross_entropy_loss(y_true, y_pred):
    # L = -1/n sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return -np.sum(y_true * np.log(y_pred + 1e-15) + (1-y_true) * np.log(1-y_pred + 1e-15)) / y_true.shape[0]

def forward(X, w, b):
    z = np.dot(X, w) + b
    y_pred = softmax(z)
    return y_pred

def compute_gradient(X, y_true, y_pred):
    n = X.shape[0]
    error = y_pred - y_true
    dw = np.dot(X.T, error) / n
    db = np.sum(error) / n
    return dw, db

def training_logistic_regression(X, y_true, learning_rate = 0.01, epochs = 1000):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0

    for epoch in range(epochs):
        y_pred = forward(X, w, b)
        loss = cross_entropy_loss(y_true, y_pred)
        dw, db = compute_gradient(X, y_true, y_pred)

        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    return w, b

def predict(X, w, b):
    z = np.dot(X, w) + b
    y_pred = softmax(z)
    return (y_pred >= 0.5).astype(int)