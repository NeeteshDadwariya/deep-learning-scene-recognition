import numpy as np

from layers.BaseLayer import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, p_val=0.2):
        self.p_val = p_val
        self.number_of_units = None
        self.pass_through = True
        self._mask_val = None
        self.inp_size = None

    # Defining output shape function
    def get_output(self):
        return self.inp_size

    # Defining backward flow
    def back_flow(self, total_gradient):
        return total_gradient * self._mask_val

    # Defining forward flow
    def front_flow(self, X, training=True):
        c_val = (1 - self.p_val)
        if training:
            self._mask_val = np.random.uniform(size=X.shape) > self.p_val
            c_val = self._mask_val
        return X * c_val
