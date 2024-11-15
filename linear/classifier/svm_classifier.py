import numpy as np

from linear.utils.svm_utils import linear_handler, polynomial_handler, rbf_handler


class SupportVectorMachine:
    def __init__(self, c=1.0, alpha=0.1, iterations=1000, kernel='linear', degree=3, gamma='scale'):
        self.c = c
        self.alpha = alpha
        self.iterations = iterations
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.w = None
        self.b = 0

    def fit(self, x, y):
        y = np.where(y == 1, 1, -1)
        n, m = x.shape
        self.w = np.zeros(m)
        self.b = 0

        self._set_weights(x, y, n)

    def _set_weights(self, x, y, n):
        for epoch in range(self.iterations):
            for i in range(n):
                condition = y[i] * (self._kernel_handler(x[i]) + self.b)
                if condition < 1:
                    self.w += self.alpha * (self.c * y[i] * x[i] - 2 * self.w)
                    self.b += self.alpha * self.c * y[i]
                else:
                    self.w -= self.alpha * 2 * self.w

    def _kernel_handler(self, x):
        if self.kernel == 'linear':
            return linear_handler(self.w, x)
        elif self.kernel == 'polynomial':
            return polynomial_handler(self.w, x, self.degree)
        elif self.kernel == 'rbf':
            return rbf_handler(self.w, x, self.gamma)
        else:
            raise ValueError("incorrect : {}".format(self.kernel))

    def predict(self, x):
        return np.sign(np.dot(x, self.w) + self.b)
