class BaseActivationFunction:
    def __call__(self, *args, **keyword_args):
        return self.forward(*args, **keyword_args)

    def __set_forward_result__(self, result):
        self._forward_result = result

    def __get_forward_result__(self):
        return self._forward_result

    def __set_backward_result__(self, result):
        self._backward_result = result

    def __get_backward_result__(self):
        return self._backward_result
