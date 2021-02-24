class BaseLoss:
    def __call__(self, *args, **keyword_args):
        return self.forward(*args, **keyword_args)
