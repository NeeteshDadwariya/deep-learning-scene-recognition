import copy

import numpy as np

from layers.BaseLayer import BaseLayer


class BatchNormalization(BaseLayer):
    def __init__(self, momentum=0.99, axis=0):
        self.momentum = momentum
        self.eps = 0.01
        self.running_mean = None
        self.running_var = None
        self.axis = axis

    def initialize(self, optimizer):
        self.gamma = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)
        self.gamma_optimizer = copy.copy(optimizer)
        self.beta_optimizer = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)

    def forward_flow(self, X, training=True):
        if self.running_mean is None:
            self.running_mean = np.mean(X, self.axis)
            self.running_var = np.var(X, self.axis)

        mean = np.mean(X, self.axis)
        var = np.var(X, self.axis)
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

        self.X_bar = X - mean
        self.inverse_stddev = 1 / np.sqrt(var + self.eps)

        normalized_X = self.X_bar * self.inverse_stddev
        output = self.gamma * normalized_X + self.beta

        return output

    def backward_flow(self, total_gradient):
        gamma = self.gamma
        X_norm = self.X_bar * self.inverse_stddev
        grad_gamma = np.sum(total_gradient * X_norm, self.axis)
        grad_beta = np.sum(total_gradient, self.axis)

        self.gamma = self.gamma_optimizer.update(self.gamma, grad_gamma)
        self.beta = self.beta_optimizer.update(self.beta, grad_beta)

        batch_size = total_gradient.shape[0]

        total_gradient = (1 / batch_size) * gamma * self.inverse_stddev * (
                batch_size * total_gradient
                - np.sum(total_gradient, self.axis)
                - self.X_bar * self.inverse_stddev ** 2 * np.sum(total_gradient * self.X_bar, self.axis)
        )

        return total_gradient

    def get_output(self):
        return self.input_shape
