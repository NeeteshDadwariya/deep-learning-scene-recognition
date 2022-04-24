import numpy as np

from layers.BaseLayer import BaseLayer


class ReluActivation:
    def gradient(self, x):
        return np.where(x >= 0, 1, 0)

    def __call__(self, data):
        return np.where(data >= 0, data, 0)


class SoftmaxActivation:
    def gradient(self, data):
        p = self.__call__(data)
        return p * (1 - p)

    def __call__(self, data):
        e_x = np.exp(data - np.max(data, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


class Activation(BaseLayer):
    def __init__(self, func):
        self.layer_input = None
        self.func = func()

    def name(self):
        return "Activation ({})".format(self.func.__class__.__name__)

    def get_output(self):
        return self.input_shape

    def backward_flow(self, total_gradient):
        return total_gradient * self.func.gradient(self.layer_input)

    def forward_flow(self, X, training=True):
        self.layer_input = X
        return self.func(X)
