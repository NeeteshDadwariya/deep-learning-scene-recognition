import tensorflow as tf
import numpy as np
import glob

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from layers.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from layers.loss_functions import SquareLoss, CrossEntropy
from layers.neural_network import NeuralNetwork
from layers.optimizers import Adam


class DeepLearningModel:

    def __init__(self, n_inputs, n_outputs):
        model = NeuralNetwork(optimizer=Adam(), loss=CrossEntropy)
        model.add(Conv2D(input_shape=n_inputs, n_filters=16, filter_shape=(2, 2), stride=1, padding='same'))
        # model.add(BatchNormalization(axis=0))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_shape=(2, 2), stride=2, padding='same'))

        model.add(Conv2D(n_filters=32, filter_shape=(2, 2), stride=1, padding='same'))
        # model.add(BatchNormalization(axis=0))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_shape=(2, 2)))  # Valid padding

        model.add(Conv2D(n_filters=64, filter_shape=(2, 2), stride=1, padding='same'))
        # model.add(BatchNormalization(axis=0))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_shape=(2, 2)))  # Valid padding

        model.add(Conv2D(n_filters=128, filter_shape=(2, 2), stride=1, padding='same'))
        # model.add(BatchNormalization(axis=0))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_shape=(2, 2)))  # Valid padding

        model.add(Flatten())
        model.add(Dense(256))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(256))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(n_outputs))
        model.add(Activation('softmax'))

        self.model = model

    def get_model(self):
        return self.model


BATCH_SIZE = 32
IMG_SIZE = (154, 154)

train_dir = './seg_train'
val_dir = './seg_val'

train_dir = './seg_train/seg_train'
val_dir = './seg_val'

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

train_ds = train_ds.map(lambda x, y: (tf.keras.layers.Rescaling(1. / 255)(x), y))

files = glob.glob("./seg_train/seg_train/**/*.*", recursive=True)
print(len(files))

image_batch, labels_batch = next(iter(train_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

X = []
y = []

for image_batch, label_batch in train_ds:
    for i in range(BATCH_SIZE):
        if i < image_batch.shape[0]:
            X.append(image_batch[i].numpy())
            y.append(label_batch[i].numpy())

# MAX_SIZE = 10000
# X = np.array(X)[:MAX_SIZE]
# y = np.array(y)[:MAX_SIZE]

X = np.array(X)
y = np.array(y)

X = np.moveaxis(X, -1, 1)
# Convert to one-hot encoding
y = to_categorical(y.astype("int"))

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = DeepLearningModel(n_inputs=(3, 154, 154), n_outputs=6).get_model()
model.summary(name="DeepLearningModel")

train_err, val_err = model.fit(X_train, y_train, n_epochs=1, batch_size=BATCH_SIZE)

# Training and validation error plot
n = len(train_err)
# training, = plt.plot(range(n), train_err, label="Training Error")
# validation, = plt.plot(range(n), val_err, label="Validation Error")
# plt.legend(handles=[training, validation])
# plt.title("Error Plot")
# plt.ylabel('Error')
# plt.xlabel('Iterations')
# plt.show()

_, accuracy = model.test_on_batch(X_test, y_test)
print ("Accuracy:", accuracy)
