# import numpy as np
#
# from linear.dataset.data import X_train, y_train, X_test
#
#
# class LinearClassifierGD:
#     def __init__(self, learning_rate=0.01, alpha=1.0, l1_ratio=0.5, epochs=1000):
#         self.learning_rate = learning_rate
#         self.alpha = alpha
#         self.l1_ratio = l1_ratio
#         self.epochs = epochs
#
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.weights = np.zeros(n_features)
#         self.bias = 0
#
#         for epoch in range(self.epochs):
#             # Предсказание
#             linear_output = np.dot(X, self.weights) + self.bias
#             margins = y * linear_output
#
#             # Вычисление потерь
#             loss = np.where(margins >= 1, 0, 1 - margins)
#
#             # Градиенты
#             dW = -np.dot(X.T, y * (margins < 1)) / n_samples
#             db = -np.sum(y * (margins < 1)) / n_samples
#
#             # Регуляризация Elastic Net
#             dW += self.alpha * ((1 - self.l1_ratio) * 2 * self.weights + self.l1_ratio * np.sign(self.weights))
#
#             # Обновление параметров
#             self.weights -= self.learning_rate * dW
#             self.bias -= self.learning_rate * db
#
#     def predict(self, X):
#         linear_output = np.dot(X, self.weights) + self.bias
#         return np.sign(linear_output)
#
#
# # Пример использования
# linear_classifier = LinearClassifierGD(learning_rate=0.01, alpha=0.5, l1_ratio=0.5, epochs=1000)
# linear_classifier.fit(X_train, y_train)
# y_pred_gd = linear_classifier.predict(X_test)
