import numpy as np

from linear.utils.regressions_utils import penalty_handler


class LogisticRegressionGD:
    def __init__(self, alpha=0.01, iterations=1000, penalty='none', risk='logistic'):
        self.alpha = alpha
        self.iterations = iterations
        self.penalty = penalty
        self.risk = risk
        self.W = None

    def fit(self, x, y):
        n, m = x.shape
        self.W = np.zeros(m)
        self._set_weights(x, y, n)

    def _set_weights(self, x, y, n):
        for _ in range(self.iterations):
            if self.risk == 'logistic':
                predictions = self.sigmoid(x @ self.W)
                gradient = x.T @ (predictions - y) / n
            elif self.risk == 'hinge':
                margins = 1 - y * (x @ self.W)
                gradient = -x.T @ (y * (margins > 0)) / n
            elif self.risk == 'log':
                predictions = self.sigmoid(x @ self.W)
                gradient = x.T @ (predictions - y) / n

            gradient = penalty_handler(self.penalty, gradient, self.W)
            self.W -= self.alpha * gradient

    def predict(self, x):
        return self.sigmoid(x @ self.W)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def predict_classes(self, x):
        return (self.predict(x) >= 0.5).astype(int)
