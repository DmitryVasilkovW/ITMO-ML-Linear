import numpy as np


class SVMClassifier:
    def __init__(self, kernel="linear", C=1.0, learning_rate=0.01, epochs=1000):
        self.kernel = kernel
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def _linear_kernel(self, X):
        return X

    def _polynomial_kernel(self, X, degree=3):
        return X ** degree

    def _rbf_kernel(self, X, gamma=0.1):
        return np.exp(-gamma * (X ** 2).sum(axis=1))

    def _compute_kernel(self, X):
        if self.kernel == "linear":
            return self._linear_kernel(X)
        elif self.kernel == "polynomial":
            return self._polynomial_kernel(X)
        elif self.kernel == "rbf":
            return self._rbf_kernel(X)
        else:
            raise ValueError("Unknown kernel")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            for i in range(n_samples):
                condition = y[i] * (np.dot(self.weights, X[i]) + self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * 1 / self.epochs * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * 1 / self.epochs * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.learning_rate * y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)
