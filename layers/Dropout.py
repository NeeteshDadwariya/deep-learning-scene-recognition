import numpy as np

from layers.BaseLayer import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, p=0.2):
        self.p = p
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True

    def get_output(self):
        return self.input_shape

    def backward_flow(self, total_gradient):
        return total_gradient * self._mask

    def forward_flow(self, X, training=True):
        c = (1 - self.p)
        if training:
            self._mask = np.random.uniform(size=X.shape) > self.p
            c = self._mask
        return X * c
