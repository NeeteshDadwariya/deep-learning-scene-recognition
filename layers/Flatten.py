import numpy as np

from layers.BaseLayer import BaseLayer


class Flatten(BaseLayer):
    def __init__(self, input_shape=None):
        self.prev_shape = None
        self.input_shape = input_shape

    def get_output(self):
        return (np.prod(self.input_shape),)

    def backward_flow(self, total_gradient):
        return total_gradient.reshape(self.prev_shape)

    def forward_flow(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], -1))
