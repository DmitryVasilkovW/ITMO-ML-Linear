import numpy as np
import pandas as pd

from linear.utils.regressions_utils import penalty_handler


class LogisticRegressionSGD:
    def __init__(self, alpha=0.01, epochs=10, batch_size=1, penalty='none', risk='logistic'):
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.penalty = penalty
        self.risk = risk
        self.w = None

    def fit(self, x, y):
        n, m = x.shape
        self.w = np.random.normal(0, 1, m)

        for _ in range(self.epochs):
            x_shuffled, y_shuffled = self._shuffle(x, y)

            for i in range(0, n, self.batch_size):
                x_batch = x_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                if self.risk == 'logistic':
                    predictions = self.sigmoid(x_batch @ self.w)
                    gradient = x_batch.T @ (predictions - y_batch) / self.batch_size
                elif self.risk == 'hinge':
                    margins = 1 - y_batch * (x_batch @ self.w)
                    gradient = -x_batch.T @ (y_batch * (margins > 0)) / self.batch_size
                elif self.risk == 'log':
                    predictions = self.sigmoid(x_batch @ self.w)
                    gradient = x_batch.T @ (predictions - y_batch) / self.batch_size

                gradient = penalty_handler(self.penalty, gradient, self.w)
                self.w -= self.alpha * gradient

    @staticmethod
    def _shuffle(x, y):
        y_with_reset_index = y.reset_index(drop=True)

        indices = np.random.permutation(len(x))
        x_shuffled = x.iloc[indices] if isinstance(x, pd.DataFrame) else x[indices]
        y_shuffled = y_with_reset_index.iloc[indices] if isinstance(y_with_reset_index, pd.Series) else \
            y_with_reset_index[indices]

        return x_shuffled, y_shuffled

    def predict(self, x):
        return self.sigmoid(x @ self.w)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def predict_classes(self, x):
        return (self.predict(x) >= 0.5).astype(int)
