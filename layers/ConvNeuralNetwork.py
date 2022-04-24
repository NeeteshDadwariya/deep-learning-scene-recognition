import numpy as np
from terminaltables import AsciiTable

from layers.utils import batch_iterator


class NeuralNetwork:

    def __init__(self, optimizer, loss, val_data=None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()
        self.val_set = None
        if val_data:
            X, y = val_data
            self.val_set = {"X": X, "y": y}

    def add(self, layer):
        if self.layers:
            print(self.layers[-1].name(), "output", self.layers[-1].get_output())
            layer.set_input(shape=self.layers[-1].get_output())
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)
        self.layers.append(layer)

    def test_on_batch(self, X, y):
        y_pred = self._forward_pass(X, training=False)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.calc_accuracy(y, y_pred)

        return loss, acc

    def train_on_batch(self, X, y):
        y_pred = self._forward_pass(X)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        accuracy = self.loss_function.calc_accuracy(y, y_pred)
        loss_grad = self.loss_function.gradient(y, y_pred)
        self._backward_pass(loss_grad=loss_grad)

        return loss, accuracy

    def fit(self, X, y, n_epochs, batch_size):
        for i in range(n_epochs):
            batch_error = []
            batch_accuracy = []
            batch = 0
            for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size):
                loss, accuracy = self.train_on_batch(X_batch, y_batch)
                batch_error.append(loss)
                print("Training for epoch:{} batch:{} | loss={}, accuracy={}".format(i, batch, loss, accuracy))
                batch += 1

            self.errors["training"].append(np.mean(batch_error))

            if self.val_set is not None:
                val_loss, accuracy = self.test_on_batch(self.val_set["X"], self.val_set["y"])
                self.errors["validation"].append(val_loss)
                batch_accuracy.append(accuracy)

        return self.errors["training"], self.errors["validation"]

    def _forward_pass(self, X, training=True):
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_flow(layer_output, training)

        return layer_output

    def _backward_pass(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward_flow(loss_grad)

    def summary(self, name="Model Summary"):
        print(AsciiTable([[name]]).table)
        print("Input Shape: %s" % str(self.layers[0].input_shape))
        table_data = [["Layer Name", "Parameters", "Output Shape"]]
        tot_params = 0
        for layer in self.layers:
            layer_name = layer.name()
            params = layer.parameters()
            out_shape = layer.get_output()
            table_data.append([layer_name, str(params), str(out_shape)])
            tot_params += params
        print(AsciiTable(table_data).table)
        print("Total Parameters: %d\n" % tot_params)

    def predict(self, X):
        return self._forward_pass(X, training=False)
