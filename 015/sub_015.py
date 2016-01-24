"""
Submission 15. Full training set.

Accuracy score: blind
"""
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import *
from sklearn.cross_validation import *
from sklearn.metrics import *

TRAIN_FILE = 'data/train.csv'
TEST_FILE = 'data/test.csv'

train_data = np.loadtxt(TRAIN_FILE, skiprows = 1, delimiter = ',', dtype = 'float')
X = train_data[:, 1:]
X = X.reshape((X.shape[0], 1, 28, 28))
X = X/255
raw_Y = train_data[:, 0].reshape(-1, 1)

X_test = np.loadtxt(TEST_FILE, skiprows = 1, delimiter = ',', dtype = 'float')
X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))
X_test = X_test/255

Y_expander = OneHotEncoder().fit(raw_Y)
Y = Y_expander.transform(raw_Y).astype(int).toarray()

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode = 'valid', input_shape = (1, 28, 28)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode = 'valid', input_shape = (1, 28, 28)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.35))

model.add(Convolution2D(32, 3, 3, border_mode = 'valid', input_shape = (1, 28, 28)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode = 'valid', input_shape = (1, 28, 28)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.35))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.35))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.03, decay=1e-7, momentum=0.1, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

batch_size = 30

model.fit(X, Y, nb_epoch = 40, batch_size = batch_size, verbose = 1)

Y_test_pred = model.predict_classes(X_test, batch_size = batch_size, verbose = 1)

id_col = np.arange(1, Y_test_pred.shape[0] + 1)
output = np.vstack((id_col, Y_test_pred)).transpose()
np.savetxt('output.csv', output, fmt = '%d', delimiter = ',', header = 'ImageId,Label', comments = '')

