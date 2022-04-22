class Dropout:
    def __init__(self, prob=0.5):
        self.input_shape = None
        self.output_shape = None
        self.input_data = None
        self.output = None
        self.isbias = False
        self.activation = None
        self.parameters = 0
        self.delta = 0
        self.weights = 0
        self.bias = 0
        self.prob = prob
        self.delta_weights = 0
        self.delta_biases = 0

    def set_output_shape(self):
        self.output_shape = self.input_shape
        self.weights = 0

    def apply_activation(self, x, train=True):
        if train:
            self.input_data = x
            flat = np.array(self.input_data).flatten()
            random_indices = np.random.randint(0, len(flat), int(self.prob * len(flat)))
            flat[random_indices] = 0
            self.output = flat.reshape(x.shape)
            return self.output
        else:
            self.input_data = x
            self.output = x / self.prob
            return self.output

    def activation_dfn(self, x):
        return x

    def backpropagate(self, nx_layer):
        if type(nx_layer).__name__ != "Conv2d":
            self.error = np.dot(nx_layer.weights, nx_layer.delta)
            self.delta = self.error * self.activation_dfn(self.out)
        else:
            self.delta = nx_layer.delta
        self.delta[self.output == 0] = 0
