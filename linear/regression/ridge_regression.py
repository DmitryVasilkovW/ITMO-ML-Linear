import numpy as np


class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.W = None

    def fit(self, X, y):
        n, m = X.shape
        X_b = np.c_[np.ones((n, 1)), X]
        self.W = np.linalg.inv(X_b.T @ X_b + self.alpha * np.eye(m + 1)) @ X_b.T @ y

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.W

    def predict_classes(self, X):
        predictions = self.predict(X)
        return np.where(predictions >= 0, 1, -1)