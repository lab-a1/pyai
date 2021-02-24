import numpy as np


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __str__(self):
        return str(self.value)

    def __add__(self, x):
        return Tensor(self.value + x.value)

    def __sub__(self, x):
        return Tensor(self.value - x.value)

    def __mul__(self, x):
        return Tensor(self.value * x.value)

    def __truediv__(self, x):
        return Tensor(self.value / x.value)

    def shape(self):
        return self.value.shape()
