#!/usr/bin/env python

# NOTE: As said in the problem set, you do *not* have to
#  run this code! It is just to give you some context as
#  to where the datasets for Part I come from

import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from keras.datasets import mnist, cifar10
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization

###### load the STL-10 dataset
with open("stl10_binary/train_X.bin", 'rb') as f:
    everything = np.fromfile(f, dtype=np.uint8)
    images = np.reshape(everything, (-1, 3, 96, 96))
    X_train = np.transpose(images, (0, 3, 2, 1))

with open("stl10_binary/test_X.bin", 'rb') as f:
    everything = np.fromfile(f, dtype=np.uint8)
    images = np.reshape(everything, (-1, 3, 96, 96))
    X_test = np.transpose(images, (0, 3, 2, 1))

with open("stl10_binary/train_y.bin", 'rb') as f:
    y_train = np.fromfile(f, dtype=np.uint8) - 1
    Y_train = np_utils.to_categorical(y_train, 10)

with open("stl10_binary/test_y.bin", 'rb') as f:
    y_test = np.fromfile(f, dtype=np.uint8) - 1
    Y_test = np_utils.to_categorical(y_test, 10)


# save the responses
np.savetxt("Y_train.csv", Y_train, delimiter=",")
np.savetxt("Y_test.csv", Y_test, delimiter=",")

# the input data have to be resized to be 224x224,
# and need the proper offsets to be applied
X_train_input = np.zeros((X_train.shape[0], 3, 224, 224))
X_test_input = np.zeros((X_test.shape[0], 3, 224, 224))

for i, val in enumerate(X_train):
    im = Image.fromarray(val, 'RGB').resize((224, 224), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    X_train_input[i] = im

for i, val in enumerate(X_test):
    im = Image.fromarray(val, 'RGB').resize((224, 224), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    X_test_input[i] = im

###### construct VGG-19 model
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

model.load_weights("vgg19_weights.h5")

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

######## Make predictions using all but the last hidden layer
model2 = copy.copy(model)
model2.layers = model2.layers[:39]
model2.compile(loss='categorical_crossentropy', optimizer=RMSprop())

X_train_new = model2.predict(X_train_input, verbose=1)
X_test_new = model2.predict(X_test_input, verbose=1)

np.savetxt("X_train_new.csv", X_train_new, delimiter=",")
np.savetxt("X_test_new.csv", X_test_new, delimiter=",")
