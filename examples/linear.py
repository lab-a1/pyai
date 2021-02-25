import numpy as np
from pyai import Tensor
from pyai.loss import MSELoss
from pyai.nn.layers import Linear


criterion = MSELoss()

linear = Linear(input_size=10, output_size=1)
x = np.random.rand(16, 10)
target = np.random.rand(16, 1)
y = linear(x)
loss = criterion(y, target)
loss_gradients = criterion.gradients(y, target)
dy = linear.backward(loss_gradients)
print(dy.shape)
