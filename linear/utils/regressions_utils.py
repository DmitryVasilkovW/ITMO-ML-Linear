import numpy as np


def penalty_handler(penalty, gradient, w):
    if penalty == 'l2':
        gradient += w
    elif penalty == 'l1':
        gradient += np.sign(w)
    elif penalty == 'elastic_net':
        gradient += w + np.sign(w)

    return gradient
