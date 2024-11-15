import numpy as np


class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.W = None

    def fit(self, x, y):
        n, m = x.shape
        x_b = np.c_[np.ones((n, 1)), x]
        self.W = np.linalg.inv(x_b.T @ x_b + self.alpha * np.eye(m + 1)) @ x_b.T @ y

    def predict(self, x):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        return x_b @ self.W

    def predict_classes(self, x):
        predictions = self.predict(x)
        return np.where(predictions >= 0, 1, -1)
