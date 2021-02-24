import numpy as np
from base import BaseLoss


class MSELoss(BaseLoss):
    """
    Mean-Squared Error, or L2 loss.

    Equation: MSE = \frac{\sum_{i=1}^{n}(y_{true}-y_{predicted})^2}{n}
    """

    def forward(self, y_true, y_predicted):
        squared = np.square(y_true - y_predicted)
        return np.mean(np.abs(squared))


if __name__ == "__main__":
    criterion = MSELoss()
    a = np.array([[-0.5855, 0.4962, -0.7684], [0.0587, 0.5546, 0.9823]])
    b = np.array([[0.7184, -1.3773, 0.9070], [-0.1963, -3.2091, -0.8386]])
    print(criterion(a, b))
