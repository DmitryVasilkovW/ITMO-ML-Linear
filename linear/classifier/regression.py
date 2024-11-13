import random

import numpy as np
import pandas as pd


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


class LogisticRegressionSGD:
    def __init__(self, alpha=0.01, epochs=10, batch_size=1, penalty='none', risk='logistic'):
        self.alpha = alpha  # темп обучения
        self.epochs = epochs  # количество эпох
        self.batch_size = batch_size  # размер батча
        self.penalty = penalty
        self.risk = risk
        self.W = None

    def fit(self, X, y):
        n, m = X.shape
        self.W = np.random.normal(0, 1, m)  # начальная инициализация весов

        for _ in range(self.epochs):
            Y = y.reset_index(drop=True)

            # Перемешиваем данные
            indices = np.random.permutation(len(X))
            X_shuffled = X.iloc[indices] if isinstance(X, pd.DataFrame) else X[indices]
            y_shuffled = Y.iloc[indices] if isinstance(Y, pd.Series) else Y[indices]

            for i in range(0, n, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # Рассчёт предсказаний и градиента для текущего батча
                if self.risk == 'logistic':
                    predictions = self.sigmoid(X_batch @ self.W)
                    gradient = X_batch.T @ (predictions - y_batch) / self.batch_size
                elif self.risk == 'hinge':
                    margins = 1 - y_batch * (X_batch @ self.W)
                    gradient = -X_batch.T @ (y_batch * (margins > 0)) / self.batch_size
                elif self.risk == 'log':
                    predictions = self.sigmoid(X_batch @ self.W)
                    gradient = X_batch.T @ (predictions - y_batch) / self.batch_size

                # Применение регуляризации
                if self.penalty == 'l2':
                    gradient += self.W
                elif self.penalty == 'l1':
                    gradient += np.sign(self.W)
                elif self.penalty == 'elastic_net':
                    gradient += self.W + np.sign(self.W)

                # Обновление весов
                self.W -= self.alpha * gradient

    def predict(self, X):
        return self.sigmoid(X @ self.W)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_classes(self, X):
        return (self.predict(X) >= 0.5).astype(int)
