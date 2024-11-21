from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from linear.regression.logistic_regression_gd import LogisticRegressionGD


@pytest.fixture
def model():
    return LogisticRegressionGD(alpha=0.1, iterations=10, penalty='l2', risk='logistic')


def test_initialization(model):
    assert model.alpha == 0.1
    assert model.iterations == 10
    assert model.penalty == 'l2'
    assert model.risk == 'logistic'
    assert model.w is None
