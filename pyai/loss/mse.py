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
    a = np.array([[-0.5855, 0.4962, -0.7684], [0.0587, 0.5546, 0.9823]])
    b = np.array([[0.7184, -1.3773, 0.9070], [-0.1963, -3.2091, -0.8386]])
    loss = criterion(a, b)
    print(loss, loss.shape)
