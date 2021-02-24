import numpy as np
from base import BaseLoss


class MAELoss(BaseLoss):
    """
    Mean Absolute Error, or L1 loss.

    Equation: MAE = \frac{\sum_{i=1}^{n}|y_{true}-y_{predicted}|}{n}
    """

    def forward(self, y_true, y_predicted):
        return np.mean(np.abs(y_true - y_predicted))


if __name__ == "__main__":
    criterion = MAELoss()
    a = np.array([[-0.5855, 0.4962, -0.7684], [0.0587, 0.5546, 0.9823]])
    b = np.array([[0.7184, -1.3773, 0.9070], [-0.1963, -3.2091, -0.8386]])
    print(criterion(a, b))
