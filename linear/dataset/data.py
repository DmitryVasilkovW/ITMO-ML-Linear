import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_and_preprocess_data(url):
    data = pd.read_csv(url, header=None)
    X = data.iloc[:, 2:].values  # признаки
    y = data.iloc[:, 1].values  # метки

    # Преобразование меток в бинарный вид (+1, -1)
    y = np.where(y == 'M', 1, -1)

    # Нормализация данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
