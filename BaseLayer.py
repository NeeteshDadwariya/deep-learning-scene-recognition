# Defining Base Layer class
class BaseLayer(object):

    def __init__(self):
        self.input_size = None

    def set_input(self, shape):
        self.input_size = shape

    def name(self):
        return self.__class__.__name__

    def params(self):
        return 0

    def get_output(self):
        raise NotImplementedError()

    def front_flow(self, X, training):
        raise NotImplementedError()

    def back_flow(self, total_gradient):
        raise NotImplementedError()

