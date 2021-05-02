from typing import Callable
from pyai.nn.layers.base import BaseLayer
from pyai import Tensor


F = Callable[[Tensor], Tensor]


class ActivationFunction(BaseLayer):
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        return self.f(x)

    def backward(self, gradients: Tensor) -> Tensor:
        return self.f_prime(self.x) * gradients
