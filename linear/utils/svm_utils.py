import numpy as np


def linear_handler(w, x):
    return np.dot(w, x)


def polynomial_handler(w, x, degree):
    return (np.dot(w, x) + 1) ** degree


def rbf_handler(w, x, gamma):
    if gamma == 'scale':
        if w.shape[0] > 0:
            gamma_value = 1 / (w.shape[0] * np.var(w) if np.var(w) > 0 else 1.0)
        else:
            gamma_value = 1.0
    else:
        gamma_value = 1.0
    return np.exp(-gamma_value * np.linalg.norm(w - x) ** 2)
