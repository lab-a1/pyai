from pyai import metrics
import numpy as np


def test_auc_1():
    tpr = np.array([0, 0.3, 0.6, 0.6, 1, 1])
    fpr = np.array([0, 0.3, 0.3, 0.7, 0.7, 1])
    auc = metrics.auc(tpr, fpr)

    assert round(auc, 3) == 0.585


def test_auc_2():
    tpr = np.array([0, 0.3, 0.6, 0.78, 1, 1])
    fpr = np.array([0, 0.3, 0.6, 0.78, 1, 1])
    auc = metrics.auc_from_scratch(tpr, fpr)

    assert round(auc, 3) == 0.5
