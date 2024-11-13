import numpy as np


class LinearRegressionRidge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        n, d = X.shape
        I = np.eye(d)
        self.weights = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y

    def predict(self, X):
        return np.sign(X @ self.weights)
