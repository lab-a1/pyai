import numpy as np
from base import BaseActivationFunction


class Sigmoid(BaseActivationFunction):
    """
    https://en.wikipedia.org/wiki/Sigmoid_function

    Equation: f(x) = \frac{1}{1+e^{-x}}
    """

    def forward(self, x):
        resut = 1 / (1 + np.exp(-x))
        self.__set_forward_result__(result)
        return self.__get_forward_result__()

    def backward(self, previous):
        resut = (
            (1 - self.__get_forward_result__())
            * self.__get_forward_result__()
            * previous
        )
        self.__set_backward_result__(resut)
        return self.__get_backward_result__()
