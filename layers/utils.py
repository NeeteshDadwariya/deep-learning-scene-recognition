import math

import numpy as np

SAME_PADDING = "same"
VALID_PADDING = "valid"


def batch_iterator(X, y=None, batch_size=64):
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]


def create_diagonal_matrix(x):
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] = x[i]
    return m


def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def get_padding_value(filter, padding=SAME_PADDING):
    if padding == "valid":
        return (0, 0), (0, 0)
    elif padding == SAME_PADDING:
        filter_h, filter_w = filter
        h1 = int(math.floor((filter_h - 1) / 2))
        h2 = int(math.ceil((filter_h - 1) / 2))
        w1 = int(math.floor((filter_w - 1) / 2))
        w2 = int(math.ceil((filter_w - 1) / 2))

        return (h1, h2), (w1, w2)


def find_column_values(image, filter, padding, stride=1):
    batch_size, channels, height, width = image
    filter_h, filter_w = filter
    pad_h, pad_w = padding
    output_height = int((height + np.sum(pad_h) - filter_h) / stride + 1)
    output_width = int((width + np.sum(pad_w) - filter_w) / stride + 1)

    i0 = np.repeat(np.arange(filter_h), filter_w)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(output_height), output_width)
    j0 = np.tile(np.arange(filter_w), filter_h * channels)
    j1 = stride * np.tile(np.arange(output_width), output_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(channels), filter_h * filter_w).reshape(-1, 1)
    return (k, i, j)


def convert_img_to_col(images, filter, stride, output=SAME_PADDING):
    pad_h, pad_w = get_padding_value(filter, output)
    images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')
    k, i, j = find_column_values(images.shape, filter, (pad_h, pad_w), stride)
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    filter_height, filter_width = filter
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    return cols


def column_to_image(cols, images_shape, filter, stride, output_shape=SAME_PADDING):
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = get_padding_value(filter, output_shape)
    padded_h = height + np.sum(pad_h)
    padded_w = width + np.sum(pad_w)
    padded_i = np.zeros((batch_size, channels, padded_h, padded_w))
    k, i, j = find_column_values(images_shape, filter, (pad_h, pad_w), stride)
    cols = cols.reshape(channels * np.prod(filter), -1, batch_size)
    cols = cols.transpose(2, 0, 1)
    np.add.at(padded_i, (slice(None), k, i, j), cols)
    return padded_i[:, :, pad_h[0]:height + pad_h[0], pad_w[0]:width + pad_w[0]]


class AdamOptimizer:
    def __init__(self, rate=0.001, decay_rate1=0.9, decay_rate2=0.999):
        self.delta = None
        self.rate = rate
        self.eps = 1e-8
        self.momentum = None
        self.velocity = None
        self.decay_rate1 = decay_rate1
        self.decay_rate2 = decay_rate2

    def update(self, original_weight, weight_grad):
        if self.momentum is None:
            self.momentum = np.zeros(np.shape(weight_grad))
            self.velocity = np.zeros(np.shape(weight_grad))

        self.momentum = self.decay_rate1 * self.momentum + (1 - self.decay_rate1) * weight_grad
        self.velocity = self.decay_rate2 * self.velocity + (1 - self.decay_rate2) * np.power(weight_grad, 2)

        updated_velocity = self.velocity / (1 - self.decay_rate2)
        updated_momentum = self.momentum / (1 - self.decay_rate1)

        self.delta = self.rate * updated_momentum / (np.sqrt(updated_velocity) + self.eps)
        return original_weight - self.delta


def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred, axis=0) / len(y_true)


class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def calc_accuracy(self, y, y_pred):
        return 0


class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)


class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Clipping probability to avoid divide by zero error
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def calc_accuracy(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Clipping probability to avoid divide by zero error
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)
