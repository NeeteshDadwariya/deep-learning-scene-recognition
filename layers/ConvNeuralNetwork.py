import numpy as np

from datetime import datetime
from terminaltables import AsciiTable

from layers.utils import batch_iterator, get_time_diff


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
        train_acc = []
        val_acc = []
        total_start_time = datetime.now()
        for i in range(n_epochs):
            batch_error = []
            batch_train_accuracy = []
            val_accuracy = 0
            batch = 1
            epoch_start_time = datetime.now()
            for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size):
                loss, train_accuracy = self.train_on_batch(X_batch, y_batch)
                batch_error.append(loss)
                batch_train_accuracy.append(train_accuracy)
                print("Training for epoch:{} batch:{} in time:{} | loss={:.2f}, accuracy={:.2f}"
                      .format(i, batch, get_time_diff(epoch_start_time), loss, train_accuracy), end='\r')
                batch += 1
            print("")

            if self.val_set is not None:
                val_loss, val_accuracy = self.test_on_batch(self.val_set["X"], self.val_set["y"])
                self.errors["validation"].append(val_loss)

            mean_training_loss = np.mean(batch_error)
            mean_training_accuracy = np.mean(batch_train_accuracy)
            train_acc.append(mean_training_accuracy)
            val_acc.append(val_accuracy)

            self.errors["training"].append(mean_training_loss)
            print(
                "Training loop complete for epoch:{} in time:{} | train_loss:{:.2f} train_accuracy:{:.2f} | val_loss:{:.2f} val_accuracy:{:.2f}"
                    .format(i, get_time_diff(epoch_start_time), mean_training_loss, mean_training_accuracy, val_loss, val_accuracy))

        print("Final accuracy:{:.2f} | Time taken:{}".format(val_acc[-1], get_time_diff(total_start_time)))
        return self.errors["training"], self.errors["validation"], train_acc, val_acc

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
