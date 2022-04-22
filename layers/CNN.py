import keras
import pandas as pd
from keras.models import Sequential

from Conv2D import Conv2d
from Pool2d import Pool2d
from Flatten import Flatten
from Dropout import Dropout


class CNN:
    def __init__(self):
        self.layers = []
        self.info_df = {}
        self.column = ["LName", "Input Shape", "Output Shape", "Activation", "Bias"]
        self.parameters = []
        self.optimizer = ""
        self.loss = "mse"
        self.lr = 0.01
        self.mr = 0.0001
        self.metrics = []
        self.av_optimizers = ["sgd", "momentum", "adam"]
        self.av_metrics = ["mse", "accuracy", "cse"]
        self.av_loss = ["mse", "cse"]
        self.iscompiled = False
        self.model_dict = None
        self.out = []
        self.eps = 1e-15
        self.train_loss = {}
        self.val_loss = {}
        self.train_acc = {}
        self.val_acc = {}

    def add(self, layer):
        if len(self.layers) > 0:
            prev_layer = self.layers[-1]
            if prev_layer.name != "Input Layer":
                prev_layer.name = f"{type(prev_layer).__name__}{len(self.layers) - 1}"
            if layer.input_shape is None:
                if type(layer).__name__ == "Flatten":
                    ops = prev_layer.output_shape[:]
                    if type(prev_layer).__name__ == "Pool2d":
                        ops = prev_layer.output_shape[:]
                elif type(layer).__name__ == "Conv2d":
                    ops = prev_layer.output_shape[:]
                    conv_layers = list(filter(lambda l : type(l).__name__ == "Conv2d", self.layers))
                    prev_conv_layer = (conv_layers or [None])[-1]
                    if prev_conv_layer is not None:
                        layer.prev_filters = prev_conv_layer.filters
                        layer.set_variables()
                    if type(prev_layer).__name__ == "Pool2d":
                        ops = prev_layer.output_shape
                elif type(layer).__name__ == "Pool2d":
                    ops = prev_layer.output_shape[:]
                    if type(prev_layer).__name__ == "Pool2d":
                        ops = prev_layer.output_shape[:]
                else:
                    ops = prev_layer.output_shape
                layer.input_shape = ops
                layer.set_output_shape()
            layer.name = f"Out Layer({type(layer).__name__})"
        else:
            layer.name = "Input Layer"
        if type(layer).__name__ == "Conv2d":
            print(layer.filters)
            if layer.output_shape[0] <= 0 or layer.output_shape[1] <= 0:
                raise ValueError(
                    f"The output shape became invalid [i.e. {layer.output_shape}]. Reduce filter size or increase "
                    f"image size.")
        self.layers.append(layer)
        self.parameters.append(layer.parameters)

    def summary(self):
        lname = []
        linput = []
        loutput = []
        lactivation = []
        lisbias = []
        lparam = []
        for layer in self.layers:
            lname.append(layer.name)
            linput.append(layer.input_shape)
            loutput.append(layer.output_shape)
            lactivation.append(layer.activation)
            lisbias.append(layer.isbias)
            lparam.append(layer.parameters)
        model_dict = {"Layer Name": lname, "Input": linput, "Output Shape": loutput,
                      "Activation": lactivation, "Bias": lisbias, "Parameters": lparam}
        model_df = pd.DataFrame(model_dict).set_index("Layer Name")
        print(model_df)
        print(f"Total Parameters: {sum(lparam)}")


input_shape = (28, 28, 1)

m = CNN()
m.add(Conv2d(filters=2, padding=None, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
m.add(Conv2d(filters=4, kernel_size=(3, 3), padding=None, activation="relu"))
m.add(Pool2d(kernel_size=(2, 2)))
m.add(Conv2d(filters=6, kernel_size=(3, 3), padding=None, activation="relu"))
m.add(Conv2d(filters=8, kernel_size=(3, 3), padding=None, activation="relu"))
m.add(Pool2d(kernel_size=(2, 2)))
m.add(Dropout(0.1))
m.add(Flatten())
m.summary()

import tensorflow as tf

model = Sequential()
model.add(tf.keras.layers.Conv2D(filters=2, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Flatten())
model.summary()
