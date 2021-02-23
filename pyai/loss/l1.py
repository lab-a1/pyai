import numpy as np
from base import BaseLoss


class L1Loss(BaseLoss):
    """
    Equation: loss = \sum_{i=1}^{n}|y_{true}-y_{predicted}|
    """

    def forward(self, y_true, y_predicted):
        return np.abs(np.sum(y_true - y_predicted))


if __name__ == "__main__":
    criterion = L1Loss()
    a = np.array([1, 2])
    b = np.array([2, 3])
    print(criterion(a, b))
