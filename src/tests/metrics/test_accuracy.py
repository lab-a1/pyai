from pyai import metrics
import numpy as np


def test_accuracy_1():
    y_true = np.array([1, 1, 0, 1, 0, 0])
    y_hat = np.array([1, 1, 0, 0, 0, 0])
    accuracy = metrics.accuracy(y_true, y_hat)

    assert round(accuracy, 3) == 0.833


def test_accuracy_2():
    y_true = np.array([1, 1, 0, 1, 0, 0])
    y_hat = np.array([1, 1, 1, 0, 0, 0])
    accuracy = metrics.accuracy(y_true, y_hat)

    assert round(accuracy, 3) == 0.667


def test_accuracy_3():
    y_true = np.array([1, 1, 0, 1, 0, 0])
    y_hat = np.array([0, 0, 0, 0, 0, 0])
    accuracy = metrics.accuracy(y_true, y_hat)

    assert round(accuracy, 3) == 0.5
