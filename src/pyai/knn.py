from pyai import Tensor
import numpy as np


class kNN:
    def __init__(self, distance_function=None):
        self.distance_function = distance_function or self.__euclidean_distance

    def __euclidean_distance(self, x1: Tensor, x2: Tensor) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X: Tensor, y: Tensor) -> None:
        # TODO: calculate distances.
        self.X = X
        self.y = y

    def predict(self, x: Tensor):
        pass
