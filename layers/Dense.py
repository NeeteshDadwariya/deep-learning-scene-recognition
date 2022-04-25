import copy
import math

import numpy as np

from layers.BaseLayer import BaseLayer


class Dense(BaseLayer):
    def __init__(self, number_of_units, inp_size=None):
        self.l_input = None
        self.w = None
        self.w0 = None
        self.number_of_units = number_of_units
        self.inp_size = inp_size

    # Initializing values
    def initialize_value(self, optimizer):
        val = 1 / math.sqrt(self.inp_size[0])
        self.w = np.random.uniform(-val, val, (self.inp_size[0], self.number_of_units))
        self.w0 = np.zeros((1, self.number_of_units))
        self.optimal_w = copy.copy(optimizer)
        self.optimal_w0 = copy.copy(optimizer)

    # Calculating the total number of parameters
    def params(self):
        return np.prod(self.w.shape) + np.prod(self.w0.shape)

    # Defining the forward flow function
    def front_flow(self, X, training=True):
        self.l_input = X
        return self.w0 + X.dot(self.w)

    # Defining the backward flow function
    def back_flow(self, total_grad):
        W = self.w
        gradient_w = self.l_input.T.dot(total_grad)
        gradient_w0 = np.sum(total_grad, axis=0, keepdims=True)
        self.w = self.optimal_w.update(self.w, gradient_w)
        self.w0 = self.optimal_w0.update(self.w0, gradient_w0)
        total_grad = total_grad.dot(W.T)
        return total_grad

    # Returning the output units
    def get_output(self):
        return (self.number_of_units,)
