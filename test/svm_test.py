import pytest
import numpy as np

from linear.classifier.svm_classifier import SupportVectorMachine


@pytest.fixture
def svm():
    return SupportVectorMachine(c=1.0, alpha=0.1, iterations=10, kernel='linear', degree=3, gamma=0.1)


def test_initialization(svm):
    assert svm.c == 1.0
    assert svm.alpha == 0.1
    assert svm.iterations == 10
    assert svm.kernel == 'linear'
    assert svm.degree == 3
    assert svm.gamma == 0.1
    assert svm.w is None
    assert svm.b == 0


def test_kernel_handler_invalid(svm):
    svm.kernel = 'invalid_kernel'
    x = np.array([1, 2])
    with pytest.raises(ValueError, match="incorrect : invalid_kernel"):
        svm._kernel_handler(x)


def test_predict(svm):
    svm.w = np.array([0.5, -0.2])
    svm.b = 0.1
    x = np.array([[1, 2], [3, -1], [0, 0]])
    predictions = svm.predict(x)
    expected = np.sign(np.dot(x, svm.w) + svm.b)
    np.testing.assert_array_equal(predictions, expected)
