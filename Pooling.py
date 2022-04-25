import numpy as np

from layers.BaseLayer import BaseLayer
from layers.utils import VALID_PADDING, img_to_col, col_to_image

# Defining pool layer class
class PoolingLayer(BaseLayer):
    def __init__(self, pool_shape_size=(2, 2), stride=2, padding=VALID_PADDING):
        self.shape_pool = pool_shape_size
        self.stride = pool_shape_size[0] if stride is None else stride
        self.pad_val = padding

    def front_flow(self, X, train=True):
        self.l_input = X
        b_size, channel, h, w = X.shape
        X = X.reshape(b_size * channel, 1, h, w)
        _, h_out, w_out = self.get_output()
        X_col = img_to_col(X, self.shape_pool, self.stride, self.pad_val)
        out = self._pool_forward(X_col)
        out = out.reshape(h_out, w_out, b_size, channel)
        out = out.transpose(2, 3, 0, 1)
        return out

    def back_flow(self, total_grad):
        b_size, _, _, _ = total_grad.shape
        channels, h, w = self.input_size
        total_grad = total_grad.transpose(2, 3, 0, 1).ravel()
        total_gradient_col = self._pool_backward(total_grad)
        total_grad = col_to_image(total_gradient_col, (b_size * channels, 1, h, w), self.shape_pool,
                                  self.stride, self.pad_val)
        total_grad = total_grad.reshape((b_size,) + self.input_size)
        return total_grad

    def get_output(self):
        channel, h, w = self.input_size
        h_out = (h - self.shape_pool[0]) // self.stride + 1
        w_out = (w - self.shape_pool[1]) // self.stride + 1
        return channel, int(h_out), int(w_out)

# Defining maxpool layer class
class MaxPooling2D(PoolingLayer):
    def _pool_forward(self, X_col):
        argument_maximum = np.argmax(X_col, axis=0).flatten()
        o = X_col[argument_maximum, range(argument_maximum.size)]
        self.cache = argument_maximum
        return o

    def _pool_backward(self, total_gradient):
        total_grad_col = np.zeros((np.prod(self.shape_pool), total_gradient.size))
        argument_maximum = self.cache
        total_grad_col[argument_maximum, range(total_gradient.size)] = total_gradient
        return total_grad_col