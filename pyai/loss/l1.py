import numpy as np


class L1Loss:
    """
    Equation: loss = \sum_{i=1}^{n}|y_{true}-y_{predicted}|
    """

    def __call__(self, y_true, y_predicted):
        return self.forward(y_true, y_predicted)

    def forward(self, y_true, y_predicted):
        return np.abs(np.sum(y_true - y_predicted))

    def backward(self):
        return 1


if __name__ == "__main__":
    criterion = L1Loss()
    a = np.array([1, 2])
    b = np.array([2, 3])
    print(criterion(a, b))
