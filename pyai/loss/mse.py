import numpy as np
from pyai.loss.base import BaseLoss
from pyai import Tensor


class MSELoss(BaseLoss):
    """
    Mean-Squared Error, or L2 loss.

    Equation: MSE = \frac{\sum_{i=1}^{n}(y_{true}-y_{predicted})^2}{n}
    """

    def loss(self, y_true: Tensor, y_predicted: Tensor) -> Tensor:
        return np.mean(np.square(y_true - y_predicted))

    def gradients(self, y_true: Tensor, y_predicted: Tensor) -> Tensor:
        """
        Equation: \frac{d_{loss}}{d_{y\_predicted}}
        """
        return (y_true - y_predicted) * 2


if __name__ == "__main__":
    criterion = MSELoss()
    a = np.array([2.2, 3])
    b = np.array([1.8, 4])
    loss = criterion(a, b)
    gradients = criterion.gradients(a, b)
    gradients_lim = criterion.gradients_lim(a, b)
    print(loss, loss.shape)
    print(gradients, gradients.shape)
    print(gradients_lim, gradients_lim.shape)
