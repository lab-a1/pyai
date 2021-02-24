from typing import Dict
from pyai import Tensor


class Layer:
    """
    Generic layer for neural network layers.
    """

    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, gradients: Tensor) -> Tensor:
        raise NotImplementedError
