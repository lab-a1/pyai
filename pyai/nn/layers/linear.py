from pyai import Tensor
from pyai.nn.layers.base import Layer
import numpy as np


class Linear(Layer):
    """
    Equation: y = x*W + b

    Shapes:
        x: (batch_size, input_size)
        y: (batch_size, output_size)
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params["w"] = Tensor(np.random.rand(input_size, output_size))
        self.params["b"] = Tensor(np.random.rand(output_size))

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        return x @ self.params["w"] + self.params["b"]

    def backward(self, gradients: Tensor) -> Tensor:
        self.gradients["b"] = Tensor(np.sum(gradients, axis=0))
        self.gradients["w"] = Tensor(self.x.T @ gradients)
        return gradients @ self.params["w"].T
