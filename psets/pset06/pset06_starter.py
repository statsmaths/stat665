import numpy as np

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

# function to read in and process the cifar-10 data
def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape(50000, 3072)
    X_test = X_test.reshape(10000, 3072)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # take only first 3 classes
    X_train = X_train[(y_train < 3).reshape(50000)]
    y_train = y_train[(y_train < 3).reshape(50000)]
    X_test = X_test[(y_test < 3).reshape(10000)]
    y_test = y_test[(y_test < 3).reshape(10000)]
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 3)
    Y_test = np_utils.to_categorical(y_test, 3)
    return X_train, Y_train, X_test, Y_test

# copy the first nlayers of model 'model' and freeze
# them. Note that these are layers in the keras sense,
# that is, it counts a drop-out layer or activation layer
# as an actual layer
def copy_freeze_model(model, nlayers = 1):
    new_model = Sequential()
    for l in model.layers[:nlayers]:
      l.trainable = False
      new_model.add(l)
    return new_model

# read in the dataset
(X_train, Y_train, X_test, Y_test) = load_data()

# for testing your code, you can downsample the data.
# for example, here we use just the first 1000 observations
X_train = X_train[:1000]
X_test = X_test[:1000]
Y_train = Y_train[:1000]
Y_test = Y_test[:1000]

# simple example: one hidden layer with 16 hidden nodes
model = Sequential()
model.add(Dense(16, input_shape=(3072,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)
model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1,
          show_accuracy=True, validation_split=0.2)



print('Classifcation rate %02.3f' % model.evaluate(X_test, Y_test, show_accuracy=True)[1])



