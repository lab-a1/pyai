import numpy as np
from pyai import Tensor
from pyai.nn.activations.base import ActivationFunction


class Tanh(ActivationFunction):
    def __init__(self):
        super().__init__(self.tahn, self.tanh_prime)

    def tahn(self, x: Tensor) -> Tensor:
        return np.tanh(x)

    def tanh_prime(self, x: Tensor) -> Tensor:
        y = self.tahn(x)
        return 1 - y ** 2
