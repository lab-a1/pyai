from pyai import Tensor, metrics
import numpy as np


def precision(y_true: Tensor, y_hat: Tensor) -> float:
    (tp, fp, tn, fn) = metrics.confusion_matrix(
        y_true, y_hat, return_tuple=True
    )
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0
