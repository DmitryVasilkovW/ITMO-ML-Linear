import pytest
import numpy as np

from linear.regression.ridge_regression import RidgeRegression


@pytest.fixture
def model():
    return RidgeRegression(alpha=1.0)


def test_initialization(model):
    assert model.alpha == 1.0
    assert model.W is None


def test_fit(model):
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 3, 5])

    model.fit(x, y)

    assert model.W is not None
    assert model.W.shape == (x.shape[1] + 1,)


def test_predict(model):
    x_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 3, 5])

    model.fit(x_train, y_train)

    x_test = np.array([[7, 8], [9, 10]])
    predictions = model.predict(x_test)

    assert predictions.shape == (x_test.shape[0],)


def test_predict_classes(model):
    x_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, -1, 1])

    model.fit(x_train, y_train)

    x_test = np.array([[7, 8], [9, 10]])
    predicted_classes = model.predict_classes(x_test)

    assert np.all((predicted_classes == 1) | (predicted_classes == -1))
