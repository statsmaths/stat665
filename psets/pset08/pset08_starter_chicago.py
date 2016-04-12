import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.layers.normalization import BatchNormalization

# load the STL-10 crime data into Python. you need to first
# download these from here:
#    http://euler.stat.yale.edu/~tba3/class_data/stl10

X_train = np.genfromtxt('X_train_new.csv', delimiter=',')
Y_train = np.genfromtxt('Y_train.csv', delimiter=',')
X_test = np.genfromtxt('X_test_new.csv', delimiter=',')
Y_test = np.genfromtxt('Y_test.csv', delimiter=',')

