import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from linear.regression.logistic_regression_sgd import LogisticRegressionSGD


@pytest.fixture
def model():
    return LogisticRegressionSGD(alpha=0.1, epochs=5, batch_size=2, penalty='l2', risk='logistic')


def test_initialization(model):
    assert model.alpha == 0.1
    assert model.epochs == 5
    assert model.batch_size == 2
    assert model.penalty == 'l2'
    assert model.risk == 'logistic'
    assert model.w is None


@patch("linear.utils.regressions_utils.logistic_risk_handler", return_value=np.array([0.1, 0.2]))
@patch("linear.utils.regressions_utils.penalty_handler", side_effect=lambda penalty, grad, w: grad)
def test_fit(mock_penalty_handler, mock_logistic_risk_handler, model):
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = pd.Series([0, 1, 0, 1])

    model.fit(x, y)

    assert model.w is not None
    assert len(model.w) == x.shape[1]
    assert mock_logistic_risk_handler.call_count == model.epochs * (len(x) // model.batch_size)
    assert mock_penalty_handler.call_count == mock_logistic_risk_handler.call_count