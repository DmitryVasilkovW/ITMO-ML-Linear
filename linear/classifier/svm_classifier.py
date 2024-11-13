import numpy as np


class SVM:
    def __init__(self, C=1.0, alpha=0.01, iterations=1000, kernel='linear', degree=3, gamma='scale'):
        self.C = C
        self.alpha = alpha
        self.iterations = iterations
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.W = None
        self.b = 0

    def fit(self, X, y):
        y = np.where(y == 1, 1, -1)
        n, m = X.shape
        self.W = np.zeros(m)
        self.b = 0

        for epoch in range(self.iterations):
            for i in range(n):
                condition = y[i] * (self.kernel_function(X[i]) + self.b)
                if condition < 1:
                    self.W += self.alpha * (self.C * y[i] * X[i] - 2 * self.W)
                    self.b += self.alpha * self.C * y[i]
                else:
                    self.W -= self.alpha * 2 * self.W

    def kernel_function(self, x):
        if self.kernel == 'linear':
            return np.dot(self.W, x)
        elif self.kernel == 'polynomial':
            return (np.dot(self.W, x) + 1) ** self.degree
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                if self.W.shape[0] > 0:
                    gamma_value = 1 / (self.W.shape[0] * np.var(self.W) if np.var(self.W) > 0 else 1.0)
                else:
                    gamma_value = 1.0
            else:
                gamma_value = self.gamma
            return np.exp(-gamma_value * np.linalg.norm(self.W - x) ** 2)
        else:
            raise ValueError("Неизвестное ядро: {}".format(self.kernel))

    def predict(self, X):
        return np.sign(np.dot(X, self.W) + self.b)