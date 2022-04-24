class BaseLayer(object):

    def __init__(self):
        self.input_shape = None

    def set_input(self, shape):
        self.input_shape = shape

    def name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def get_output(self):
        raise NotImplementedError()

    def forward_flow(self, X, training):
        raise NotImplementedError()

    def backward_flow(self, total_gradient):
        raise NotImplementedError()

