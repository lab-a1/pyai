from pyai import Tensor
import numpy as np


def confusion_matrix(
    y_true: Tensor, y_hat: Tensor, return_tuple: bool = False
) -> Tensor:
    matrix = [[0, 0], [0, 0]]
    for yt, yh in zip(y_true, y_hat):
        if yt == yh == 1:
            matrix[0][0] += 1  # true positive
        if yh == 1 and yt != yh:
            matrix[0][1] += 1  # false positive
        if yt == yh == 0:
            matrix[1][1] += 1  # true negative
        if yh == 0 and yt != yh:
            matrix[1][0] += 1  # false negative
    if return_tuple:
        # tp, fp, tn, fn
        return (matrix[0][0], matrix[0][1], matrix[1][1], matrix[1][0])
    else:
        return np.array(matrix)
