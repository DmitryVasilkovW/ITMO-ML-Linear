import numpy as np


class LogisticRegressionGD:
    def __init__(self, alpha=0.01, iterations=1000, penalty='none', risk='logistic'):
        self.alpha = alpha
        self.iterations = iterations
        self.penalty = penalty
        self.risk = risk
        self.W = None

    def fit(self, X, y):
        n, m = X.shape
        self.W = np.zeros(m)

        for _ in range(self.iterations):
            if self.risk == 'logistic':
                predictions = self.sigmoid(X @ self.W)
                gradient = X.T @ (predictions - y) / n
            elif self.risk == 'hinge':
                margins = 1 - y * (X @ self.W)
                gradient = -X.T @ (y * (margins > 0)) / n
            elif self.risk == 'log':
                predictions = self.sigmoid(X @ self.W)
                gradient = X.T @ (predictions - y) / n

            if self.penalty == 'l2':
                gradient += self.W
            elif self.penalty == 'l1':
                gradient += np.sign(self.W)
            elif self.penalty == 'elastic_net':
                gradient += self.W + np.sign(self.W)

            self.W -= self.alpha * gradient

    def predict(self, X):
        return self.sigmoid(X @ self.W)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_classes(self, X):
        return (self.predict(X) >= 0.5).astype(int)
