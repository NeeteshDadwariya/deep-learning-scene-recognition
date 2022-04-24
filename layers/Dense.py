import copy
import math

import numpy as np

from layers.BaseLayer import BaseLayer


class Dense(BaseLayer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.weight = None
        self.weight0 = None

    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[0])
        self.weight = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.weight0 = np.zeros((1, self.n_units))
        self.optim_w = copy.copy(optimizer)
        self.optim_w0 = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.weight.shape) + np.prod(self.weight0.shape)

    def forward_flow(self, X, training=True):
        self.layer_input = X
        return X.dot(self.weight) + self.weight0

    def backward_flow(self, total_gradient):
        W = self.weight
        grad_w = self.layer_input.T.dot(total_gradient)
        grad_w0 = np.sum(total_gradient, axis=0, keepdims=True)
        self.weight = self.optim_w.update(self.weight, grad_w)
        self.weight0 = self.optim_w0.update(self.weight0, grad_w0)
        total_gradient = total_gradient.dot(W.T)
        return total_gradient

    def get_output(self):
        return (self.n_units,)
