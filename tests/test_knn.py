from pyai import Tensor, kNN
import numpy as np


def test_euclidean():
    knn = kNN()

    distance = knn.distance_function(np.array([3, 5]), np.array([2, 7]))

    assert round(distance, 3) == 2.236
