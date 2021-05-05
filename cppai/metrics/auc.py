from pyai import Tensor
import numpy as np


def auc(tpr: Tensor, fpr: Tensor) -> float:
    return np.sum(np.trapz(tpr, fpr))


def auc_from_scratch(tpr: Tensor, fpr: Tensor) -> float:
    auc_score = 0
    for i in range(1, len(tpr)):
        auc_score += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1])
    return auc_score / 2
