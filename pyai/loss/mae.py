import numpy as np
from pyai.loss.base import BaseLoss
from pyai.tensor import Tensor


class MAELoss(BaseLoss):
    """
    Mean Absolute Error, or L1 loss.

    Equation: MAE = \frac{\sum_{i=1}^{n}|y_{true}-y_{predicted}|}{n}
    """

    def loss(self, y_true: Tensor, y_predicted: Tensor) -> Tensor:
        return Tensor(np.mean(np.abs(y_true - y_predicted)))


if __name__ == "__main__":
    criterion = MAELoss()
    a = Tensor([[-0.5855, 0.4962, -0.7684], [0.0587, 0.5546, 0.9823]])
    b = Tensor([[0.7184, -1.3773, 0.9070], [-0.1963, -3.2091, -0.8386]])
    loss = criterion(a, b)
    print(loss, loss.shape)
