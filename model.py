from layers1.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D
from layers1.loss_functions import SquareLoss
from layers1.neural_network import NeuralNetwork
from layers1.optimizers import Adam


class Model:
    # lf.add(Conv2D(n_filters=16, filter_shape=(3, 3), stride=1, input_shape=(1, 8, 8), padding='same'))
    # clf.add(Activation('relu'))
    # clf.add(Dropout(0.25))
    # clf.add(BatchNormalization())
    # clf.add(Conv2D(n_filters=32, filter_shape=(3, 3), stride=1, padding='same'))
    # clf.add(Activation('relu'))
    # clf.add(Dropout(0.25))
    # clf.add(BatchNormalization())
    # clf.add(Flatten())
    # clf.add(Dense(256))
    # clf.add(Activation('relu'))
    # clf.add(Dropout(0.4))
    # clf.add(BatchNormalization())
    # clf.add(Dense(10))
    # clf.add(Activation('softmax'))

    def __init__(self, n_inputs, n_outputs):
        model = NeuralNetwork(optimizer=Adam(), loss=SquareLoss)
        model.add(ZeroPadding2D((2, 2)))
        model.add(Conv2D(n_filters=16, filter_shape=(2, 2), stride=1, padding='same'))
        model.add(BatchNormalization()) #Axis=3
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_shape=(2, 2)))

        # X = MaxPooling2D((2, 2), padding='same', name='max_pool0')(X)



