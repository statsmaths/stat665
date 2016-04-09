import numpy as np

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

# set this to false once you have tested your code!
TEST = True

# function to read in and process the cifar-10 data; set the
# number of classes you want
def load_data(nclass):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # down-sample to three classes
    X_train = X_train[(y_train < nclass).reshape(50000)]
    y_train = y_train[(y_train < nclass).reshape(50000)]
    X_test = X_test[(y_test < nclass).reshape(10000)]
    y_test = y_test[(y_test < nclass).reshape(10000)]
    # create responses
    Y_train = np_utils.to_categorical(y_train, nclass)
    Y_test = np_utils.to_categorical(y_test, nclass)
    if TEST:
        X_train = X_train[:1000]
        Y_train = Y_train[:1000]
        X_test = X_test[:1000]
        Y_test = Y_test[:1000]
    return X_train, Y_train, X_test, Y_test


# Note: You'll need to do this manipulation to construct the
# output of the autoencoder. This is because the autoencoder
# will have a flattend dense layer on the output, and you need
# to give Keras a flatted version of X_train
X_train_auto_output = X_train.reshape(X_train.shape[0], 3072)


