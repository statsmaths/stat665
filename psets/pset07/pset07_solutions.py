import numpy as np

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D

VERBOSE = 1
TEST = False

# function to read in and process the cifar-10 data
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


def create_cnn_layers(kern, nclass, nlayer = 1, auto = False):
    model = Sequential()
    # feature layers
    for l in range(nlayer):
        model.add(Convolution2D(32, kern, kern, border_mode='same', input_shape = (3, 32, 32)))
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # prediction layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nclass))
    if not auto:
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop())
    else:
        model.compile(loss='mean_squared_error', optimizer=RMSprop())
    return model


def copy_freeze_model(model, nlayers = 1):
    new_model = Sequential()
    for l in model.layers[:nlayers]:
        l.trainable = False
        new_model.add(l)
    return new_model


def freeze_features(model, nclass, nlayers = 1):
    new_model = Sequential()
    # copy feature layers
    for l in model.layers[:(nlayers*2 + 2)]:
        l.trainable = False
        new_model.add(l)
    # prediction layers
    new_model.add(Flatten())
    new_model.add(Dense(512))
    new_model.add(Activation('relu'))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(nclass))
    new_model.add(Activation('softmax'))
    # compile the model
    new_model.compile(loss='categorical_crossentropy', optimizer=RMSprop())
    return new_model


def fit_model(model, X_train, Y_train, auto=False):
    if auto:
        Y_train_use = Y_train.reshape(Y_train.shape[0], 3072)
        model.fit(X_train, Y_train_use, batch_size=32, nb_epoch=25 - TEST*23,
            verbose=VERBOSE, show_accuracy=True,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
    else:
        model.fit(X_train, Y_train, batch_size=32, nb_epoch=25 - TEST*23,
            verbose=VERBOSE, show_accuracy=True,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
    return model


def eval_model(model, X_test, Y_test, auto=False):
    if auto:
        Y_test_use = Y_test.reshape(Y_test.shape[0], 3072)
        return model.evaluate(X_test, Y_test_use, show_accuracy=True)[0]
    else:
        return model.evaluate(X_test, Y_test, show_accuracy=True)[1]


def print_output(scores, vals, name, auto=False):
    print("\n\n %s\n" % name)
    for s, v in zip(scores, vals):
        if auto:
            print("  value: %4d   mse: %02.5f" % (v, s) )
        else:
            print("  value: %4d   classification rate: %02.5f" % (v, s) )
    print("\n\n")


def part_I():
    (X_train, Y_train, X_test, Y_test) = load_data(2)
    scores = []
    kernel_sizes = (1, 3, 5)
    for k in kernel_sizes:
        model = create_cnn_layers(k,3,1)
        fit_model(model, X_train, Y_train)
        scores.append(eval_model(model, X_test, Y_test))

    print_output(scores, kernel_sizes, "Part I")


def part_II():
    (X_train, Y_train, X_test, Y_test) = load_data(2)
    scores = []
    kernel_sizes = (1, 3, 5)
    for k in kernel_sizes:
        model = create_cnn_layers(k,3072,1,auto=True)
        fit_model(model, X_train, X_train, auto=True)
        scores.append(eval_model(model, X_test, X_test, auto=True))

    print_output(scores, kernel_sizes, "Part II", auto=True)


def part_III():
    (X_train, Y_train, X_test, Y_test) = load_data(2)
    scores = []

    model = create_cnn_layers(3,2,1)
    fit_model(model, X_train, Y_train)
    model = freeze_features(model, 2)
    fit_model(model, X_train, Y_train)
    scores.append(eval_model(model, X_test, Y_test))

    model = create_cnn_layers(3,3072,1,auto=True)
    fit_model(model, X_train, X_train, auto=True)
    model = freeze_features(model, 2)
    fit_model(model, X_train, Y_train)
    scores.append(eval_model(model, X_test, Y_test))

    print_output(scores, [3.0,3072], "Part III")


def part_IV():
    (X_train, Y_train, X_test, Y_test) = load_data(2)
    (X_train_10, Y_train_10, X_test_10, Y_test_10) = load_data(10)
    scores = []

    model = create_cnn_layers(3,2,1)
    fit_model(model, X_train, Y_train)
    model = freeze_features(model, 10)
    fit_model(model, X_train_10, Y_train_10)
    scores.append(eval_model(model, X_test_10, Y_test_10))

    model = create_cnn_layers(3,3072,1,auto=True)
    fit_model(model, X_train, X_train, auto=True)
    model = freeze_features(model, 10)
    fit_model(model, X_train_10, Y_train_10)
    scores.append(eval_model(model, X_test_10, Y_test_10))

    print_output(scores, [3.0,3072], "Part IV")


def part_V():
    (X_train, Y_train, X_test, Y_test) = load_data(2)
    (X_train_10, Y_train_10, X_test_10, Y_test_10) = load_data(10)
    scores = []

    model = create_cnn_layers(3,2,2)
    fit_model(model, X_train, Y_train)
    scores.append(eval_model(model, X_test, Y_train))

    model = freeze_features(model, 10, 2)
    fit_model(model, X_train_10, Y_train_10)
    scores.append(eval_model(model, X_test_10, Y_test_10))

    print_output(scores, [0.0, 1.0], "Part V")


if __name__ == "__main__":
    #part_I()
    #part_II()
    #part_III()
    part_IV()
    part_V()



