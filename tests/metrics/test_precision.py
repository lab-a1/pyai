from pyai import metrics
import numpy as np


def test_precision_1():
    y_true = np.array([1, 1, 0, 1, 0, 0])
    y_hat = np.array([1, 1, 1, 1, 1, 1])
    precision = metrics.precision(y_true, y_hat)

    assert round(precision, 3) == 0.5


def test_precision_2():
    y_true = np.array([1, 1, 0, 1, 0, 0])
    y_hat = np.array([1, 1, 1, 0, 0, 0])
    precision = metrics.precision(y_true, y_hat)

    assert round(precision, 3) == 0.667


def test_precision_3():
    y_true = np.array([1, 1, 0, 1, 0, 0])
    y_hat = np.array([0, 0, 0, 0, 0, 0])
    precision = metrics.precision(y_true, y_hat)

    assert round(precision, 3) == 0
