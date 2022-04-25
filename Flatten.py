import numpy as np

from layers.BaseLayer import BaseLayer


class Flatten(BaseLayer):
    def __init__(self, input_shape_val=None):
        self.previous_shape = None
        self.in_shape = input_shape_val

    # Defining output shape
    def get_output(self):
        return (np.prod(self.in_shape),)

    # Defining backward flow
    def back_flow(self, total_gradient):
        return total_gradient.reshape(self.previous_shape)

    # Defining forward flow
    def front_flow(self, X, training=True):
        self.previous_shape = X.shape
        return X.reshape((X.shape[0], -1))
