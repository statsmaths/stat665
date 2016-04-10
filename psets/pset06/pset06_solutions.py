#!/usr/bin/env python

import numpy as np

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

VERBOSE = 1
TEST = False

# function to read in and process the cifar-10 data, with only
# the first three categories for the purposes of speed
def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape(50000, 3072)
    X_test = X_test.reshape(10000, 3072)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_train = X_train[(y_train < 3).reshape(50000)]
    y_train = y_train[(y_train < 3).reshape(50000)]
    X_test = X_test[(y_test < 3).reshape(10000)]
    y_test = y_test[(y_test < 3).reshape(10000)]
    Y_train = np_utils.to_categorical(y_train, 3)
    Y_test = np_utils.to_categorical(y_test, 3)
    if TEST:
        X_train = X_train[:1000]
        Y_train = Y_train[:1000]
        X_test = X_test[:1000]
        Y_test = Y_test[:1000]
    return X_train, Y_train, X_test, Y_test


def build_model(width, depth, auto=False):
    model = Sequential()
    for d in range(depth):
        if d == 0:
            model.add(Dense(width, input_shape=(3072,)))
        else:
            model.add(Dense(width))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
    if auto:
        model.add(Dense(3072))
        model.compile(loss='mean_squared_error', optimizer=RMSprop())
    else:
        model.add(Dense(3))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop())
    return model


def copy_freeze_model(model, nlayers = 1):
    new_model = Sequential()
    for l in model.layers[:nlayers]:
        l.trainable = False
        new_model.add(l)
    return new_model


def add_top_layer(model):
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop())
    return model


def fit_model(model, auto=False):
    if auto:
        model.fit(X_train, X_train, batch_size=32, nb_epoch=25,
            verbose=VERBOSE, show_accuracy=True,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
    else:
        model.fit(X_train, Y_train, batch_size=32, nb_epoch=25,
            verbose=VERBOSE, show_accuracy=True,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
    return model


def eval_model(model, auto=False):
    if auto:
        return model.evaluate(X_test, X_test, show_accuracy=True)[0]
    else:
        return model.evaluate(X_test, Y_test, show_accuracy=True)[1]


def print_output(scores, vals, name, auto=False):
    print("\n\n %s\n" % name)
    for s, v in zip(scores, vals):
        if auto:
            print("  type: %4d   mse: %02.5f" % (v, s) )
        else:
            print("  type: %4d   classification rate: %02.5f" % (v, s) )
    print("\n\n")


# I - Changing layer size (width)
def part_I():
    scores = []
    widths = (2, 8, 32, 128, 512)
    for w in widths:
        model = build_model(w,1)
        model = fit_model(model)
        scores.append(eval_model(model))

    print_output(scores, widths, "Part I")


# II - Changing number of layers (depth)
def part_II():
    scores = []
    for depth in range(1,6):
        model = build_model(512, depth)
        model = fit_model(model)
        scores.append(eval_model(model))

    print_output(scores, range(1,6), "Part II")


# III - Build layers in sequence freezing each as you go
def part_III():
    scores = []
    for depth in range(5):
        if (depth == 0):
            model = build_model(512,1)
        else:
            model = copy_freeze_model(model, 3*depth)
            model = add_top_layer(model)
        model = fit_model(model)
        scores.append(eval_model(model))

    print_output(scores, range(5), "Part III")


# IV - Autoencoder: changing layer size (width)
def part_IV():
    scores = []
    widths = (32, 128, 512, 1024)
    for w in widths:
        model = build_model(w, 1, auto=True)
        model = fit_model(model, auto=True)
        scores.append(eval_model(model, auto=True))

    print_output(scores, widths, "Part IV", auto=True)


# V - Autoencoder as pre-training
def part_V():
    scores = []
    for depth in range(5):
        if (depth == 0):
            model = build_model(1024, 1, auto=True)
            model = fit_model(model, auto=True)
        else:
            model = copy_freeze_model(model, 3*depth)
            model = add_top_layer(model)
            model = fit_model(model)
            model = fit_model(model)
            scores.append(eval_model(model))

    print_output(scores, range(1,6), "Part V")


if __name__ == "__main__":
    (X_train, Y_train, X_test, Y_test) = load_data()
    part_I()
    part_II()
    part_III()
    part_IV()
    part_V()
