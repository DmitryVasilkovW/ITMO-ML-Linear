import numpy as np


def penalty_handler(penalty, gradient, w):
    if penalty == 'l2':
        gradient += w
    elif penalty == 'l1':
        gradient += np.sign(w)
    elif penalty == 'elastic_net':
        gradient += w + np.sign(w)

    return gradient


def logistic_risk_handler(x, y, w, n):
    predictions = sigmoid(x @ w)
    gradient = x.T @ (predictions - y) / n

    return gradient


def hinge_risk_handler(x, y, w, n):
    margins = 1 - y * (x @ w)
    gradient = -x.T @ (y * (margins > 0)) / n

    return gradient


def log_risk_handler(x, y, w, n):
    predictions = np.log(1 + np.exp(-y * (x @ w)))
    gradient = -x.T @ (y * (1 - predictions)) / n
    return gradient


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
