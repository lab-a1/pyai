import numpy as np


class Tensor:
    """
    Tensor is a n-dimensional array.
    """

    def __init__(self, value):
        self.value = np.asarray(value)

    def __call__(self):
        return self.value

    def __str__(self):
        return str(self.value)

    def __add__(self, other):
        return Tensor(self.value + other.value)

    def __sub__(self, other):
        return Tensor(self.value - other.value)

    def __mul__(self, other):
        return Tensor(self.value * other.value)

    def __matmul__(self, other):
        return Tensor(self.value @ other.value)

    def __truediv__(self, other):
        return Tensor(self.value / other.value)

    def __getattribute__(self, key):
        if key == "__array__":
            return self
        elif key == "shape":
            return self.value.shape
        elif key == "T":
            return Tensor(self.value.T)
        return super(Tensor, self).__getattribute__(key)
