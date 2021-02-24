from pyai import Tensor


class BaseLoss:
    def __call__(self, *args, **keyword_args) -> None:
        return self.loss(*args, **keyword_args)

    def loss(self, y_true: Tensor, y_predicted: Tensor) -> Tensor:
        raise NotImplementedError

    def gradients(self, y_true: Tensor, y_predicted: Tensor) -> Tensor:
        raise NotImplementedError
