import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.layers.normalization import BatchNormalization

# load the chicago crime data into Python. you need to first
# download these from here:
#    http://euler.stat.yale.edu/~tba3/class_data/chi_python

X_train = np.genfromtxt('chiCrimeMat_X_train.csv', delimiter=',')
Y_train = np.genfromtxt('chiCrimeMat_Y_train.csv', delimiter=',')
X_test = np.genfromtxt('chiCrimeMat_X_test.csv', delimiter=',')
Y_test = np.genfromtxt('chiCrimeMat_Y_test.csv', delimiter=',')
