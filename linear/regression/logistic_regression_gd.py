import numpy as np

from linear.utils.regressions_utils import (penalty_handler,
                                            logistic_risk_handler,
                                            hinge_risk_handler,
                                            log_risk_handler,
                                            sigmoid)


class LogisticRegressionGD:
    def __init__(self, alpha=0.01, iterations=1000, penalty='none', risk='logistic'):
        self.alpha = alpha
        self.iterations = iterations
        self.penalty = penalty
        self.risk = risk
        self.w = None

    def fit(self, x, y):
        n, m = x.shape
        self.w = np.zeros(m)
        self._set_weights(x, y, n)

    def _set_weights(self, x, y, n):
        for _ in range(self.iterations):
            if self.risk == 'logistic':
                gradient = logistic_risk_handler(x, y, self.w, n)
            elif self.risk == 'hinge':
                gradient = hinge_risk_handler(x, y, self.w, n)
            elif self.risk == 'log':
                gradient = log_risk_handler(x, y, self.w, n)

            gradient = penalty_handler(self.penalty, gradient, self.w)
            self.w -= self.alpha * gradient

    def predict(self, x):
        return sigmoid(x @ self.w)

    def predict_classes(self, x):
        return (self.predict(x) >= 0.5).astype(int)
