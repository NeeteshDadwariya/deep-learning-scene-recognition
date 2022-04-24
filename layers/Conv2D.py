import copy
import math

import numpy as np

from layers.BaseLayer import BaseLayer
from layers.utils import get_padding_value, convert_img_to_col, column_to_image


class Conv2D(BaseLayer):
    def __init__(self, n_filters, filter_shape, input_shape=None, stride=1, padding='same'):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.padding = padding
        self.stride = stride

    def initialize(self, optimizer):
        filter_height, filter_width = self.filter_shape
        channels = self.input_shape[0]
        limit = 1 / math.sqrt(np.prod(self.filter_shape))
        self.weight = np.random.uniform(-limit, limit, size=(self.n_filters, channels, filter_height, filter_width))
        self.weight0 = np.zeros((self.n_filters, 1))
        self.optim_w = copy.copy(optimizer)
        self.optim_w0 = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.weight.shape) + np.prod(self.weight0.shape)

    def forward_flow(self, X, training=True):
        batch_size, channels, height, width = X.shape
        self.layer_input = X
        self.X_col = convert_img_to_col(X, self.filter_shape, stride=self.stride, output=self.padding)
        self.W_col = self.weight.reshape((self.n_filters, -1))
        output = self.W_col.dot(self.X_col) + self.weight0
        output = output.reshape(self.get_output() + (batch_size,))
        return output.transpose(3, 0, 1, 2)

    def get_output(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = get_padding_value(self.filter_shape, padding=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width)

    def backward_flow(self, total_gradient):
        total_gradient = total_gradient.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)
        grad_w = total_gradient.dot(self.X_col.T).reshape(self.weight.shape)
        grad_w0 = np.sum(total_gradient, axis=1, keepdims=True)
        self.weight = self.optim_w.update(self.weight, grad_w)
        self.weight0 = self.optim_w0.update(self.weight0, grad_w0)
        total_gradient = self.W_col.T.dot(total_gradient)
        total_gradient = column_to_image(total_gradient,
                                         self.layer_input.shape,
                                         self.filter_shape,
                                         stride=self.stride,
                                         output_shape=self.padding)

        return total_gradient
