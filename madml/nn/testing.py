import numpy as np


def fc_forward(X, W, b):
    out = X @ W + b
    cache = (W, X)
    return out, cache


def fc_backward(dY, cache):
    W, h = cache
    dW = h.T @ dY
    db = np.sum(dY, axis=0)
    dX = dY @ W.T
    return dX, dW, db


def dropout_forward(X, p_dropout):
    u = np.random.binomial(1, p_dropout, size=X.shape) / p_dropout
    out = X * u
    cache = u
    return out, cache


def dropout_backward(dY, cache):
    dX = dY * cache
    return dX


def relu_forward(X):
    out = np.maximum(X, 0)
    cache = X
    return out, cache


def relu_backward(dY, cache):
    dX = dY.copy()
    dX[cache <= 0] = 0
    return dX


def softmax(X):
    eX = np.exp((X.T - np.max(X, axis=1)).T)
    return (eX.T / eX.sum(axis=1)).T


def dcross_entropy(y_pred, y_train):
    m = y_pred.shape[0]

    grad_y = softmax(y_pred)
    grad_y[range(m), y_train] -= 1.
    grad_y /= m

    return grad_y
