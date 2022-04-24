import numpy as np

from layers.BaseLayer import BaseLayer
from layers.utils import VALID_PADDING, convert_img_to_col, column_to_image


class PoolingLayer(BaseLayer):
    def __init__(self, pool_shape=(2, 2), stride=2, padding=VALID_PADDING):
        self.pool_shape = pool_shape
        self.stride = pool_shape[0] if stride is None else stride
        self.padding = padding

    def forward_flow(self, X, training=True):
        self.layer_input = X
        batch_size, channels, height, width = X.shape
        _, out_height, out_width = self.get_output()
        X = X.reshape(batch_size * channels, 1, height, width)
        X_col = convert_img_to_col(X, self.pool_shape, self.stride, self.padding)
        output = self._pool_forward(X_col)
        output = output.reshape(out_height, out_width, batch_size, channels)
        output = output.transpose(2, 3, 0, 1)
        return output

    def backward_flow(self, total_gradient):
        batch_size, _, _, _ = total_gradient.shape
        channels, height, width = self.input_shape
        total_gradient = total_gradient.transpose(2, 3, 0, 1).ravel()
        total_gradient_col = self._pool_backward(total_gradient)
        total_gradient = column_to_image(total_gradient_col, (batch_size * channels, 1, height, width), self.pool_shape,
                                         self.stride, self.padding)
        total_gradient = total_gradient.reshape((batch_size,) + self.input_shape)
        return total_gradient

    def get_output(self):
        channels, height, width = self.input_shape
        out_height = (height - self.pool_shape[0]) // self.stride + 1
        out_width = (width - self.pool_shape[1]) // self.stride + 1
        return channels, int(out_height), int(out_width)


class MaxPooling2D(PoolingLayer):
    def _pool_forward(self, X_col):
        arg_max = np.argmax(X_col, axis=0).flatten()
        output = X_col[arg_max, range(arg_max.size)]
        self.cache = arg_max
        return output

    def _pool_backward(self, total_gradient):
        total_gradient_col = np.zeros((np.prod(self.pool_shape), total_gradient.size))
        arg_max = self.cache
        total_gradient_col[arg_max, range(total_gradient.size)] = total_gradient
        return total_gradient_col
