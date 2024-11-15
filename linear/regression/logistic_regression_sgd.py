import numpy as np
import pandas as pd

from linear.utils.regressions_utils import penalty_handler, logistic_risk_handler, hinge_risk_handler, log_risk_handler, \
    sigmoid


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
        self._handle_all_epochs(x, y, n)

    def _handle_all_epochs(self, x, y, n):
        for _ in range(self.epochs):
            x_shuffled, y_shuffled = self._shuffle(x, y)
            self._handle_for_this_epoch(x_shuffled, y_shuffled, n)

    def _handle_for_this_epoch(self, x_shuffled, y_shuffled, n):
        for i in range(0, n, self.batch_size):
            x_batch = x_shuffled[i:i + self.batch_size]
            y_batch = y_shuffled[i:i + self.batch_size]

            if self.risk == 'logistic':
                gradient = logistic_risk_handler(x_batch, y_batch, self.w, self.batch_size)
            elif self.risk == 'hinge':
                gradient = hinge_risk_handler(x_batch, y_batch, self.w, self.batch_size)
            elif self.risk == 'log':
                gradient = log_risk_handler(x_batch, y_batch, self.w, self.batch_size)

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
        return sigmoid(x @ self.w)

    def predict_classes(self, x):
        return (self.predict(x) >= 0.5).astype(int)
