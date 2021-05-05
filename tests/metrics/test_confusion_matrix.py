from pyai import metrics
import numpy as np


def test_confusion_matrix():
    y_true = np.array([1, 1, 0, 1, 0, 0])
    y_hat = np.array([1, 1, 1, 1, 1, 1])
    matrix = metrics.confusion_matrix(y_true, y_hat).reshape(-1)

    assert (matrix == np.array([3, 3, 0, 0])).all()
