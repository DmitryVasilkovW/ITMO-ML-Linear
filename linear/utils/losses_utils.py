import numpy as np


def logistic_loss(y_true, y_pred):
    return np.mean(np.log(1 + np.exp(-y_true * y_pred)))


def tanh_loss(y_true, y_pred):
    return np.mean(1 - np.tanh(y_true * y_pred))


def ridge_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def l2_norm(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred) ** 2))
