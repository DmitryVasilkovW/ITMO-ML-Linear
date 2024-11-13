import numpy as np
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"


def get_data():
    data = pd.read_csv(url, header=None)

    data.columns = [
        'top-left-square', 'top-middle-square', 'top-right-square',
        'middle-left-square', 'middle-middle-square', 'middle-right-square',
        'bottom-left-square', 'bottom-middle-square', 'bottom-right-square',
        'Class'
    ]

    data.replace({'x': 1, 'o': 0, 'b': -1}, inplace=True)

    data['Class'] = data['Class'].apply(lambda x: 1 if x == 'positive' else -1)

    return data


def accuracy_metric(y_true, y_pred):
    y_pred = np.where(y_pred >= 0, 1, -1)
    return np.mean(y_true == y_pred)
