"""
Keras MLP Implementation.

Accuracy score:  0.97619047619
"""

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import *
from sklearn.cross_validation import *
from sklearn.metrics import *

TRAIN_FILE = 'data/train.csv'
TEST_FILE = 'data/test.csv'

train_data = np.loadtxt(TRAIN_FILE, skiprows = 1, delimiter = ',', dtype = 'float')
X = train_data[:, 1:]
X = X/255
raw_Y = train_data[:, 0].reshape(-1, 1)

X_test = np.loadtxt(TEST_FILE, skiprows = 1, delimiter = ',', dtype = 'float')
X_test = X_test/255

X_train, X_cv, raw_Y_train, raw_Y_cv = train_test_split(X, raw_Y, test_size = 0.10)
Y_expander = OneHotEncoder().fit(raw_Y)
Y_train = Y_expander.transform(raw_Y_train).astype(int).toarray()
Y_cv = Y_expander.transform(raw_Y_cv).astype(int).toarray()

model = Sequential()
model.add(Dense(input_dim = X.shape[1], output_dim = 512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim = 512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim = 10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-7, momentum=0.1, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train, nb_epoch = 20, batch_size = 10, show_accuracy = True, verbose = 1, validation_split = 0.05)
Y_cv_pred = model.predict_classes(X_cv, batch_size = 10, verbose = 1)

score = accuracy_score(raw_Y_cv, Y_cv_pred)
print('Accuracy score: ', score)

Y_test_pred = model.predict_classes(X_test, batch_size = 10, verbose = 1)

id_col = np.arange(1, Y_test_pred.shape[0] + 1)
output = np.vstack((id_col, Y_test_pred)).transpose()
np.savetxt('output.csv', output, fmt = '%d', delimiter = ',', header = 'ImageId,Label', comments = '')
