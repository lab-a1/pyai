import numpy as np
from base import BaseLoss


class L2Loss(BaseLoss):
    """
    Equation: loss = \sum_{i=1}^{n}(y_{true}-y_{predicted})^{2}
    """

    def forward(self, y_true, y_predicted):
        squared = np.power(y_true - y_predicted, 2)
        return np.abs(np.sum(squared))


if __name__ == "__main__":
    criterion = L2Loss()
    a = np.array([1, 2])
    b = np.array([2, 3])
    print(criterion(a, b))
